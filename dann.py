import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

# ======================
# Device setup
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# ======================
# Config
# ======================
exp_name = 'runs/dann_grl_test'
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

# ======================
# Load Data
# ======================
X = torch.load('/projects/sds-lab/Shuochen/downscaling/canesm2/gcm_2deg_conus.pth')
y = torch.load('/projects/sds-lab/Shuochen/downscaling/cordex_canesm2/rcm_0.5deg_conus.pth')

# 1951-2005 historical 55 years; 2006-2100 rcp85
X_train = X[:365*55,:,:,:]
X_test  = X[365*55:,:,:,:]
y_train = y[:365*55,:,:,:]
y_test  = y[365*55:,:,:,:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Datasets and loaders
training_set = TensorDataset(X_train, y_train)
testing_set  = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader  = DataLoader(testing_set,  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ======================
# Models
# ======================
from models import Encoder, Decoder, VGGDomainClassifier

model_Enc       = Encoder().to(device)
model_Dec_SR    = Decoder().to(device)
model_Disc_feat = VGGDomainClassifier().to(device)

# ======================
# Gradient Reversal Layer
# ======================
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.λ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, λ=1.0):
        super().__init__()
        self.λ = λ
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.λ)

grl = GradientReversalLayer(λ=1.0)

# ======================
# Optimizers & schedulers
# ======================
optimizer_Enc       = torch.optim.AdamW(model_Enc.parameters(),       lr=LEARNING_RATE, weight_decay=1e-4)
optimizer_Dec_SR    = torch.optim.AdamW(model_Dec_SR.parameters(),    lr=LEARNING_RATE, weight_decay=1e-4)
optimizer_Disc_feat = torch.optim.AdamW(model_Disc_feat.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

scheduler_Enc       = CosineAnnealingLR(optimizer_Enc,       T_max=EPOCHS)
scheduler_Dec_SR    = CosineAnnealingLR(optimizer_Dec_SR,    T_max=EPOCHS)
scheduler_Disc_feat = CosineAnnealingLR(optimizer_Disc_feat, T_max=EPOCHS)

# ======================
# Loss functions
# ======================
loss_MSE = nn.MSELoss()
loss_BCE_logits = nn.BCEWithLogitsLoss()

# ======================
# TensorBoard
# ======================
writer = SummaryWriter(log_dir=exp_name)

# ======================
# Training
# ======================
L_rec_G_SR_list = []
loss_Disc_feat_align_list = []
test_loss_list = []

for epoch in range(EPOCHS):
    model_Enc.train()
    model_Dec_SR.train()
    model_Disc_feat.train()

    L_rec_G_SR_sum, loss_Disc_feat_align_sum = 0.0, 0.0

    for (xs, ys), (xt, _) in zip(train_dataloader, cycle(test_dataloader)):
        xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
        bs = xs.shape[0]

        # ========= Forward =========
        F_s = model_Enc(xs)   # source features
        F_t = model_Enc(xt)   # target features

        # SR reconstruction
        y_pred = model_Dec_SR(F_s)
        L_rec_G_SR = loss_MSE(y_pred, ys)

        # Domain classification with GRL
        F_s_rev = grl(F_s)
        F_t_rev = grl(F_t)
        out_s = model_Disc_feat(F_s_rev).view(-1, 1)
        out_t = model_Disc_feat(F_t_rev).view(-1, 1)

        real_label = torch.ones(bs, 1, device=device)
        fake_label = torch.zeros(bs, 1, device=device)

        loss_D_s = loss_BCE_logits(out_s, real_label)
        loss_D_t = loss_BCE_logits(out_t, fake_label)
        loss_D = 0.5 * (loss_D_s + loss_D_t)

        # ========= Total Loss =========
        loss_total = L_rec_G_SR + 0.01 * loss_D

        optimizer_Enc.zero_grad()
        optimizer_Dec_SR.zero_grad()
        optimizer_Disc_feat.zero_grad()
        loss_total.backward()
        optimizer_Enc.step()
        optimizer_Dec_SR.step()
        optimizer_Disc_feat.step()

        L_rec_G_SR_sum += L_rec_G_SR.item()
        loss_Disc_feat_align_sum += loss_D.item()

    # Average losses
    L_rec_G_SR_sum /= len(train_dataloader)
    loss_Disc_feat_align_sum /= len(train_dataloader)
    L_rec_G_SR_list.append(L_rec_G_SR_sum)
    loss_Disc_feat_align_list.append(loss_Disc_feat_align_sum)

    # ======================
    # Test evaluation
    # ======================
    with torch.no_grad():
        model_Enc.eval()
        model_Dec_SR.eval()
        model_Disc_feat.eval()

        test_loss = 0.0
        for Xb, yb in test_dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            y_enc = model_Enc(Xb)
            test_pred = model_Dec_SR(y_enc)
            test_loss += loss_MSE(test_pred, yb).item()
        test_loss /= len(test_dataloader)
        test_loss_list.append(test_loss)

    # LR schedulers
    scheduler_Enc.step()
    scheduler_Dec_SR.step()
    scheduler_Disc_feat.step()

    # Log to TensorBoard
    writer.add_scalar('Loss/L_rec_G_SR', L_rec_G_SR_sum, epoch)
    writer.add_scalar('Loss/loss_Disc_feat_align', loss_Disc_feat_align_sum, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)

    print(f"Epoch {epoch:03d} | SR Loss: {L_rec_G_SR_sum:.5f} | "
          f"Domain Loss: {loss_Disc_feat_align_sum:.5f} | Test Loss: {test_loss:.5f}")

# ======================
# Final test prediction
# ======================
preds, y_true = [], []
with torch.no_grad():
    for Xb, yb in test_dataloader:
        Xb, yb = Xb.to(device), yb.to(device)
        y_enc = model_Enc(Xb)
        y_pred = model_Dec_SR(y_enc)
        preds.append(y_pred.cpu())
        y_true.append(yb.cpu())
test_pred = torch.cat(preds, dim=0)
y_true = torch.cat(y_true, dim=0)
final_loss = loss_MSE(test_pred, y_true)
print("Final Test MSE:", final_loss.item())
