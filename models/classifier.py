import numpy as np
import torch
import torch.nn as nn


class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest'):
        super(UpSampleBlock, self).__init__()

        self.scale_factor = scale_factor
        if mode == 'nearest':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.upconv(out)

        return out


class ConvLinearClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes, upsample_mode='nearest'):
        super().__init__()
        self.n_classes = n_classes
        self.chn = [256, 128, 64]

        self.block1 = UpSampleBlock(embed_dim, self.chn[0], scale_factor=2, mode=upsample_mode)
        self.block2 = UpSampleBlock(self.chn[0], self.chn[1], scale_factor=2, mode=upsample_mode)
        self.block3 = UpSampleBlock(self.chn[1], self.chn[2], scale_factor=2, mode=upsample_mode)
        self.block4 = UpSampleBlock(self.chn[2], n_classes, scale_factor=2, mode=upsample_mode)

    def forward(self,x):
        bs, h_sqrt , ch= x.shape
        h = int(np.sqrt(h_sqrt))
        x = x.contiguous().view(bs,ch, h, h)
        
        out = torch.relu(self.block1(x))
        out = torch.relu(self.block2(out))
        out = torch.relu(self.block3(out))
        out = self.block4(out)
    
        return out
