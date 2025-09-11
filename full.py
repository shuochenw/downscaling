exp_name = 'runs/dann_full'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Function
import torch
from torch import nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Create TensorBoard writer
writer = SummaryWriter(log_dir=exp_name)

# tas
var = 'tas'
BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE = 1e-4
SCALING_FACTOR = 4
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
NUM_FEATURES = 64

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# In[3]:


# read tensor
X = torch.load('/projects/sds-lab/Shuochen/downscaling/canesm2/gcm_2deg_conus.pth')
y = torch.load('/projects/sds-lab/Shuochen/downscaling/cordex_canesm2/rcm_0.5deg_conus.pth')

# 1951-2005 historical 55 years; 2006-2100 rcp85
X_train = X[:365*55,:,:,:]
X_test = X[365*55:,:,:,:]
y_train = y[:365*55,:,:,:]
y_test = y[365*55:,:,:,:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# create datasets
training_set = TensorDataset(X_train, y_train)
testing_set = TensorDataset(X_test, y_test)
# create dataloaders
train_dataloader = DataLoader(training_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
test_dataloader = DataLoader(testing_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

from models import Encoder, Decoder, VGGDomainClassifier, Decoder_Identity, VGG19PerceptualLoss
# define model (generator)
model_Enc = Encoder().to(device)
model_Dec_Id = Decoder_Identity().to(device)
model_Dec_SR = Decoder().to(device)
# define model (discriminator)
model_Disc_feat = VGGDomainClassifier(in_channels=64).to(device)
model_Disc_img_LR = VGGDomainClassifier(in_channels=1).to(device)
model_Disc_img_HR = VGGDomainClassifier(in_channels=1).to(device)

loss_L1 = nn.L1Loss().to(device)
loss_MSE = nn.MSELoss().to(device)
loss_adversarial = nn.BCEWithLogitsLoss().to(device)
loss_percept = VGG19PerceptualLoss().to(device)

# optimizer 
params_G = list(model_Enc.parameters()) + list(model_Dec_Id.parameters()) + list(model_Dec_SR.parameters())
optimizer_G = torch.optim.AdamW(params_G, lr=LEARNING_RATE, weight_decay=1e-7)
params_D = list(model_Disc_feat.parameters()) + list(model_Disc_img_LR.parameters()) + list(model_Disc_img_HR.parameters())
optimizer_D = torch.optim.AdamW(params_D, lr=LEARNING_RATE, weight_decay=1e-7)
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=EPOCHS)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=EPOCHS)


for epoch in range(EPOCHS):
    
    # generator
    model_Enc.train()
    model_Dec_Id.train()
    model_Dec_SR.train()
    # discriminator
    model_Disc_feat.train()
    model_Disc_img_LR.train()
    model_Disc_img_HR.train()

    running_loss_D_total = 0.0
    running_loss_G_total = 0.0

    running_loss_align = 0.0
    running_loss_rec = 0.0
    running_loss_res = 0.0
    running_loss_sty = 0.0
    running_loss_idt = 0.0
    running_loss_cyc = 0.0
    
    for (X_s, Y_s), (X_t, _) in zip(cycle(train_dataloader), test_dataloader):
        X_s, Y_s, X_t = X_s.to(device), Y_s.to(device), X_t.to(device)
        # real label and fake label
        batch_size = X_t.size(0)
        real_label = torch.full((batch_size, 1), 1, dtype=X_t.dtype).cuda(non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=X_t.dtype).cuda(non_blocking=True)
        ########################
        # (1) Update D network #
        ########################
        model_Disc_feat.zero_grad()
        model_Disc_img_LR.zero_grad()
        model_Disc_img_HR.zero_grad()

        F_t = model_Enc(X_t)
        F_s = model_Enc(X_s)
        # 1. feature aligment loss (discriminator)
        # output of discriminator (feature domain) (b x c(=1) x h x w)
        output_Disc_F_t = model_Disc_feat(F_t.detach())
        output_Disc_F_s = model_Disc_feat(F_s.detach())
        # discriminator loss (feature domain)
        loss_Disc_F_t = loss_MSE(output_Disc_F_t, fake_label)
        loss_Disc_F_s = loss_MSE(output_Disc_F_s, real_label)
        loss_Disc_feat_align = (loss_Disc_F_t + loss_Disc_F_s) / 2

        # 2. SR reconstruction loss (discriminator)
        # generator output (image domain)
        Y_s_s = model_Dec_SR(F_s)
        # output of discriminator (image domain)
        output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s.detach())
        output_Disc_Y_s = model_Disc_img_HR(Y_s)
        # discriminator loss (image domain)
        loss_Disc_Y_s_s = loss_MSE(output_Disc_Y_s_s, fake_label)
        loss_Disc_Y_s = loss_MSE(output_Disc_Y_s, real_label)
        loss_Disc_img_rec = (loss_Disc_Y_s_s + loss_Disc_Y_s) / 2

        # 4. Target degradation style loss
        # generator output (image domain)
        X_s_t = model_Dec_Id(F_s)
        # output of discriminator (image domain)
        output_Disc_X_s_t = model_Disc_img_LR(X_s_t.detach())
        output_Disc_X_t = model_Disc_img_LR(X_t)
        # discriminator loss (image domain)
        loss_Disc_X_s_t = loss_MSE(output_Disc_X_s_t, fake_label)
        loss_Disc_X_t = loss_MSE(output_Disc_X_t, real_label)
        loss_Disc_img_sty = (loss_Disc_X_s_t + loss_Disc_X_t) / 2

        # 6. Cycle loss
        # generator output (image domain)
        Y_s_t_s = model_Dec_SR(model_Enc(model_Dec_Id(F_s)))
        # output of discriminator (image domain)
        output_Disc_Y_s_t_s = model_Disc_img_HR(Y_s_t_s.detach())
        output_Disc_Y_s = model_Disc_img_HR(Y_s)
        # discriminator loss (image domain)
        loss_Disc_Y_s_t_s = loss_MSE(output_Disc_Y_s_t_s, fake_label)
        loss_Disc_Y_s = loss_MSE(output_Disc_Y_s, real_label)
        loss_Disc_img_cyc = (loss_Disc_Y_s_t_s + loss_Disc_Y_s) / 2

        # discriminator weight update
        loss_D_total = loss_Disc_feat_align + loss_Disc_img_rec + loss_Disc_img_sty + loss_Disc_img_cyc
        loss_D_total.backward()
        optimizer_D.step()
    
        ########################
        # (2) Update G network #
        ########################
        model_Enc.zero_grad()
        model_Dec_SR.zero_grad()
        model_Dec_Id.zero_grad()

        # generator output (feature domain)
        F_t = model_Enc(X_t)
        F_s = model_Enc(X_s)

        # 1. feature alignment loss (generator)
        # output of discriminator (feature domain)
        output_Disc_F_t = model_Disc_feat(F_t)
        output_Disc_F_s = model_Disc_feat(F_s)
        # generator loss (feature domain)
        loss_G_F_t = loss_MSE(output_Disc_F_t, (real_label + fake_label)/2)
        loss_G_F_s = loss_MSE(output_Disc_F_s, (real_label + fake_label)/2)
        L_align_E = loss_G_F_t + loss_G_F_s

        # 2. SR reconstruction loss
        # generator output (image domain)
        Y_s_s = model_Dec_SR(F_s)
        # output of discriminator (image domain)
        output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s)
        # L1 loss
        loss_L1_rec = loss_L1(Y_s.detach(), Y_s_s)
        # perceptual loss
        # loss_percept_rec = loss_percept(Y_s.detach(), Y_s_s)
        loss_percept_rec = 0.0
        # adversatial loss
        loss_G_Y_s_s = loss_MSE(output_Disc_Y_s_s, real_label)
        L_rec_G_SR = loss_L1_rec + 0.01*loss_percept_rec + 0.01*loss_G_Y_s_s

        # 3. Target LR restoration loss
        X_t_t = model_Dec_Id(F_t)
        L_res_G_t = loss_L1(X_t, X_t_t)

        # 4. Target degredation style loss
        # generator output (image domain)
        X_s_t = model_Dec_Id(F_s)
        # output of discriminator (img domain)
        output_Disc_X_s_t = model_Disc_img_LR(X_s_t)
        # generator loss (feature domain)
        loss_G_X_s_t = loss_MSE(output_Disc_X_s_t, real_label)
        L_sty_G_t = loss_G_X_s_t

        # 5. Feature identity loss
        F_s_tilda = model_Enc(model_Dec_Id(F_s))
        L_idt_G_t = loss_L1(F_s, F_s_tilda)

        # 6. Cycle loss
        # generator output (image domain)
        Y_s_t_s = model_Dec_SR(model_Enc(model_Dec_Id(F_s)))
        # output of discriminator (image domain)
        output_Disc_Y_s_t_s = model_Disc_img_HR(Y_s_t_s)
        # L1 loss
        loss_L1_cyc = loss_L1(Y_s.detach(), Y_s_t_s)
        # perceptual loss
        # loss_percept_cyc = loss_percept(Y_s.detach(), Y_s_t_s)
        loss_percept_cyc = 0.0
        # adversarial loss 
        loss_Y_s_t_s = loss_MSE(output_Disc_Y_s_t_s, real_label)
        L_cyc_G_t_G_SR = loss_L1_cyc + 0.01*loss_percept_cyc + 0.01*loss_Y_s_t_s

        # generator weight update
        loss_G_total = 0.01*L_align_E + 1.0*L_rec_G_SR + 1.0*L_res_G_t + 0.01*L_sty_G_t + 0.01*L_idt_G_t + 1.0*L_cyc_G_t_G_SR
        loss_G_total.backward()
        optimizer_G.step()

    scheduler_D.step()
    scheduler_G.step()
    
    ########################
    #     compute loss     #
    ########################
    running_loss_D_total += loss_D_total.item()
    running_loss_G_total += loss_G_total.item()

    running_loss_align += L_align_E.item()
    running_loss_rec += L_rec_G_SR.item()
    running_loss_res += L_res_G_t.item()
    running_loss_sty += L_sty_G_t.item()
    running_loss_idt += L_idt_G_t.item()
    running_loss_cyc += L_cyc_G_t_G_SR.item()

    # Log epoch-wise training and test loss
    writer.add_scalar('Loss/running_loss_D_total', running_loss_D_total, epoch)
    writer.add_scalar('Loss/running_loss_G_total', running_loss_G_total, epoch)
    writer.add_scalar('Loss/running_loss_align', running_loss_align, epoch)
    writer.add_scalar('Loss/running_loss_rec', running_loss_rec, epoch)
    writer.add_scalar('Loss/running_loss_res', running_loss_res, epoch)
    writer.add_scalar('Loss/running_loss_sty', running_loss_sty, epoch)
    writer.add_scalar('Loss/running_loss_idt', running_loss_idt, epoch)
    writer.add_scalar('Loss/running_loss_cyc', running_loss_cyc, epoch)
    
    with torch.no_grad():
        model_Enc.eval()
        model_Dec_SR.eval()

        test_loss = 0
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_enc = model_Enc(X)
            test_pred = model_Dec_SR(y_enc)
            test_loss += loss_MSE(test_pred, y).item() # accumulatively add up the loss per epoch
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        
    writer.add_scalar('Loss/test_loss', test_loss, epoch)
    
    print(f"{epoch} | D: {running_loss_D_total:.4f} | G: {running_loss_G_total:.4f} | Align: {running_loss_align:.4f} | Rec: {running_loss_rec:.4f} | Res: {running_loss_res:.4f} | Sty: {running_loss_sty:.4f} | Idt: {running_loss_idt:.4f} | Cyc: {running_loss_cyc:.4f}")
    print(test_loss)

