import torch
import torch.nn as nn
from models.blocks.conv_block import ConvBlock

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)      # output for skip-connection
        x_pool = self.pool(x_conv) # downsampled output
        return x_conv, x_pool
