import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torchvision
from torch.autograd import Function


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)        # -> [B, 64, 14, 30]
        self.pool1 = nn.MaxPool2d(2)                        # -> [B, 64, 7, 15]
        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # -> [B, 128, 7, 15]
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)        # -> [B, 128, 4, 8]

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)  # -> [B, 256, 4, 8]

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)  # -> [B, 128, 8, 16]
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)  # skip + up -> [B, 128, 7, 15]
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)  # -> [B, 64, 14, 30]
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        
    def forward(self, x):
        x1 = self.enc1(x)       # [B, 64, 14, 30]
        x2 = self.enc2(self.pool1(x1))  # [B, 128, 7, 15]
        x3 = self.bottleneck(self.pool2(x2))  # [B, 256, 4, 8]

        x = self.up2(x3)        # [B, 128, 8, 16]
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))  # [B, 128, 7, 15]

        x = self.up1(x)         # [B, 64, 14, 30]
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x1], dim=1))  # [B, 64, 14, 30]
        
        return  x

class Decoder(nn.Module):
    def __init__(self,base_ch=64, out_channels=1):
        super().__init__()
        # super-resolution
        self.transconv1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.transconv2 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_ch, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.transconv1(x)
        x = self.conv1(x)
        x = self.transconv2(x)
        x = self.conv2(x)
        x = self.conv3(x)      
        return x
        
class VGGDomainClassifier(nn.Module):
    def __init__(self, in_channels=64):
        super(VGGDomainClassifier, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 7, 15]

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # [B, 128, 4, 8]

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # [B, 256, 2, 4]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Flatten(),             # [B, 256]
            nn.Linear(256, 1)         # [B, 1] — domain prediction
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
               
class Decoder_Identity(nn.Module):
    def __init__(self):
        super(Decoder_Identity, self).__init__()

        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, bias=True),            
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, bias=True),
            nn.ReLU()
        )

    def forward(self, feat):
        featmap_2 = self.conv_up_2(feat)
        featmap_1 = self.conv_up_1(featmap_2)
        out = self.conv_last(featmap_1)

        return out

class VGG19PerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layer=35):
        super(VGG19PerceptualLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()
        # Freeze parameters
        for name, param in self.features.named_parameters():
            param.requires_grad = False
    
    def forward(self, source, target):
        vgg_loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))

        return vgg_loss