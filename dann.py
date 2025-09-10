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

writer = SummaryWriter(log_dir=f"runs/coral")

# In[2]:


# tas
var = 'tas'
BATCH_SIZE = 32
EPOCHS = 200
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
train_dl = DataLoader(training_set, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)
test_dl = DataLoader(testing_set,
    batch_size=BATCH_SIZE,
    shuffle=True # don't necessarily have to shuffle the testing data
)

from models import Encoder, Decoder
encoder = Encoder().to(device)
decoder = Decoder().to(device)
opt_enc = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
opt_dec = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE)
sch_enc = CosineAnnealingLR(opt_enc, T_max=EPOCHS)
sch_dec = CosineAnnealingLR(opt_dec, T_max=EPOCHS)
loss_fn = nn.MSELoss()

def coral_loss(source, target):
    source = source.view(source.size(0), -1)
    target = target.view(target.size(0), -1)

    source = source - source.mean(dim=0)
    target = target - target.mean(dim=0)

    source_cov = (source.T @ source) / (source.size(0) - 1)
    target_cov = (target.T @ target) / (target.size(0) - 1)

    loss = torch.mean((source_cov - target_cov) ** 2)
    return loss

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    decoder.train()

    running_sr = 0.0
    running_coral = 0.0

    for (x_src, y_src), (x_tgt, _) in zip(cycle(train_dl), test_dl):
        x_src, y_src = x_src.to(device), y_src.to(device)
        x_tgt = x_tgt.to(device)

        # Forward
        f_src = encoder(x_src)
        f_tgt = encoder(x_tgt)
        y_pred = decoder(f_src)

        # Losses
        loss_sr = loss_fn(y_pred, y_src)
        loss_coral = coral_loss(f_src, f_tgt)
        lambda_coral = epoch / EPOCHS
        loss = loss_sr + lambda_coral * loss_coral

        # Backward
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        loss.backward()
        opt_enc.step()
        opt_dec.step()

        running_sr += loss_sr.item()
        running_coral += loss_coral.item()

    # === Validation ===
    encoder.eval()
    decoder.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y_hr in test_dl:
            x = x.to(device)
            y_hr = y_hr.to(device)

            feat = encoder(x)
            y_pred = decoder(feat)
            loss = loss_fn(y_pred, y_hr)
            test_loss += loss.item()

    # Scheduler step
    sch_enc.step()
    sch_dec.step()

    print(f"Epoch {epoch:03d} | SR Loss: {running_sr/len(train_dl):.6f} | "
          f"CORAL Loss: {running_coral/len(train_dl):.6f} | "
          f"Test Loss: {test_loss/len(test_dl):.6f}")
    # === Logging ===
    avg_sr = running_sr / len(train_dl)
    avg_coral = running_coral / len(train_dl)
    avg_test = test_loss / len(test_dl)

    writer.add_scalar('Loss/SR', avg_sr, epoch)
    writer.add_scalar('Loss/CORAL', avg_coral, epoch)
    writer.add_scalar('Loss/Test', avg_test, epoch)
    writer.add_scalar('Lambda/CORAL', lambda_coral, epoch)