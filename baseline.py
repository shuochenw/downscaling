exp_name = 'runs/baseline AdamW cosineannealing shuffle test random weidec 1e-4 400'

# # split UNet to an encoder and a decoder

# In[1]:


import torch
from torch import nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


# In[2]:


# tas
var = 'tas'
BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE = 1e-4
SCALING_FACTOR = 4
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
NUM_FEATURES = 64

# seed = 0
# import random
# random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# from torch.backends import cudnn
# cudnn.benchmark = False
# cudnn.deterministic = True

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
train_dataloader = DataLoader(training_set,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(testing_set,batch_size=BATCH_SIZE,shuffle=True)


from models import Encoder, Decoder
# import torch.nn.init as init

# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         # Example: Xavier Uniform Initialization
#         init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)
model_enc = Encoder().to(device)
model_dec = Decoder().to(device)
# model_enc.apply(initialize_weights)
# model_dec.apply(initialize_weights)

loss_fn = nn.MSELoss()
optimizer_enc = torch.optim.AdamW(model_enc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
optimizer_dec = torch.optim.AdamW(model_dec.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
scheduler_enc = CosineAnnealingLR(optimizer_enc, T_max=EPOCHS)
scheduler_dec = CosineAnnealingLR(optimizer_dec, T_max=EPOCHS)


# Create TensorBoard writer
writer = SummaryWriter(log_dir=exp_name)

train_loss_list = []
test_loss_list = []
for epoch in range(EPOCHS):
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model_enc.train()
        model_dec.train()
        
        X, y = X.to(device), y.to(device)
        y_enc = model_enc(X)
        y_pred = model_dec(y_enc)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() # accumulatively add up the loss per epoch
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()
        
    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    train_loss_list.append(train_loss)

    
    model_enc.eval()
    model_dec.eval()
    with torch.no_grad():
        test_loss = 0
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_enc = model_enc(X)
            test_pred = model_dec(y_enc)
            test_loss += loss_fn(test_pred, y).item() # accumulatively add up the loss per epoch
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        test_loss_list.append(test_loss)
        
    scheduler_enc.step()
    scheduler_dec.step()

    # Log epoch-wise training and test loss
    writer.add_scalar('Loss/Train_epoch', train_loss, epoch)
    writer.add_scalar('Loss/Test_epoch', test_loss, epoch)
    
    print(f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

# Close the writer when done
writer.close()

preds = []
y_test = []
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_enc = model_enc(X)
        y_pred = model_dec(y_enc)
        preds.append(y_pred.cpu())
        # re-combine test data in the order of the dataloader, since shuffle=True
        y_test.append(y.cpu())
# Concatenate back into one tensor
test_pred = torch.cat(preds, dim=0)
y_test = torch.cat(y_test, dim=0)
loss = nn.MSELoss()
print(loss(test_pred, y_test))
