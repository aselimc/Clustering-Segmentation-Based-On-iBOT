import numpy as np
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = self.pool(out)

        return out


class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest'):
        super(UpSampleBlock, self).__init__()

        self.scale_factor = scale_factor
        if mode == 'nearest':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = torch.relu(self.conv1(out))
        out = torch.relu(self.conv2(out))

        return out


class ConvLinearClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes, upsample_mode='nearest'):
        super(ConvLinearClassifier, self).__init__()
        self.n_classes = n_classes
        self.chn = [256, 128, 64, 32]

        self.block1 = UpSampleBlock(embed_dim, self.chn[0], scale_factor=2, mode=upsample_mode)
        self.block2 = UpSampleBlock(self.chn[0], self.chn[1], scale_factor=2, mode=upsample_mode)
        self.block3 = UpSampleBlock(self.chn[1], self.chn[2], scale_factor=2, mode=upsample_mode)
        self.block4 = UpSampleBlock(self.chn[2], self.chn[3], scale_factor=2, mode=upsample_mode)

        self.conv_mask = nn.Conv2d(self.chn[3], n_classes, kernel_size=1, stride=1)

    def forward(self,x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.conv_mask(out)

        return out


class UNetClassifier(nn.Module):
    def __init__(self, n_classes, upsample_mode='nearest'):
        super(UNetClassifier, self).__init__()

        self.n_classes = n_classes
        self.chn = [32, 64, 128, 256, 512]

        self.conv_input = nn.Conv2d(3, self.chn[0], kernel_size=1, stride=1)

        self.downsample1 = DownSampleBlock(self.chn[0], self.chn[1])
        self.downsample2 = DownSampleBlock(self.chn[1], self.chn[2])
        self.downsample3 = DownSampleBlock(self.chn[2], self.chn[3])
        self.downsample4 = DownSampleBlock(self.chn[3], self.chn[4])

        self.conv_feat = nn.Conv2d(self.chn[4], self.chn[4], kernel_size=3, stride=1, padding=1)

        self.upsample1 = UpSampleBlock(self.chn[4], self.chn[3], mode=upsample_mode)
        self.upsample2 = UpSampleBlock(self.chn[3], self.chn[2], mode=upsample_mode)
        self.upsample3 = UpSampleBlock(self.chn[2], self.chn[1], mode=upsample_mode)
        self.upsample4 = UpSampleBlock(self.chn[1], self.chn[0], mode=upsample_mode)

        self.conv_mask = nn.Conv2d(self.chn[0], n_classes, kernel_size=1, stride=1)
    
    def forward(self, x):
        enc = self.conv_input(x)
        enc = self.downsample1(enc)
        enc = self.downsample2(enc)
        enc = self.downsample3(enc)
        enc = self.downsample4(enc)
        enc = self.conv_feat(enc)

        dec = self.upsample1(enc)
        dec = self.upsample2(dec)
        dec = self.upsample3(dec)
        dec = self.upsample4(dec)
        dec = self.conv_mask(dec)

        return dec

class ConvSingleLinearClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes, patch_size, upsample_mode='bilinear'):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, n_classes, kernel_size=1)
        self.patch_size = patch_size
        self.mode = upsample_mode

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=self.patch_size, mode= self.mode, align_corners=False, recompute_scale_factor=False)
        return x