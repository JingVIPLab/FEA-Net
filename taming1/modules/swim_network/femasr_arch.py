import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math


from .network_swinir import RSTB


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()

        # Global average pooling
        self.global_pooling = nn.AdaptiveAvgPool1d(256)

        # First convolution layer
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # ReLU activation
        self.relu = nn.ReLU()

        # Second convolution layer
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        b, hw, c = x.shape
        out = self.global_pooling(x.permute(0, 2, 1))  # Transpose for 1D global pooling

        # Reshape for convolution
        out = out.view(b,hw, c)

        # First convolution layer
        out = self.conv1(out)

        # ReLU activation
        out = self.relu(out)

        # Second convolution layer
        out = self.conv2(out)

        # Sigmoid activation
        out = self.sigmoid(out)

        # Multiply attention weights with input
        out = x * out.permute(0, 2, 1)  # Transpose back to the original shape

        return out



class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(3):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
        self.channel_attention_block = ChannelAttentionBlock(embed_dim)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2) # b h*w c

        for m in self.swin_blks:
            x = m(x, (h, w))
            # x = self.channel_attention_block(x)
        x = x.transpose(1, 2).reshape(b, c, h, w) 
        return x

