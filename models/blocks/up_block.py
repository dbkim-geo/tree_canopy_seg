import torch
import torch.nn as nn
from models.blocks.conv_block import ConvBlock

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels=out_channels + skip_channels, out_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)  # upsampled

        # resize if needed (in case of mismatch due to rounding)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)  # concat on channel dim

        x = self.conv(x)
        return x
