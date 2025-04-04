import torch
import torch.nn as nn
from models.blocks.up_block import UpBlock
from models.blocks.conv_block import ConvBlock

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # UpBlock(in_ch, skip_ch, out_ch)
        self.up1 = UpBlock(256, 0, 128)     # no skip
        self.up2 = UpBlock(128, 0, 64)      # no skip
        self.up3 = UpBlock(64, 0, 64)       # no skip

        self.up4 = UpBlock(64, 256, 64)     # with skip (MaxPool3)
        self.up5 = UpBlock(64, 128, 64)     # with skip (MaxPool2)
        self.up6 = UpBlock(64, 64, 64)      # with skip (MaxPool1)

        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.Sigmoid()

    def forward(self, x, fc_skips):
        # UpBlock1~3: no skip
        x = self.up1(x)                        # 256 → 128 @ 8×8
        x = self.up2(x)                        # 128 → 64 @ 16×16
        x = self.up3(x)                        # 64 → 64 @ 32×32

        # UpBlock4~6: with skip from FakeColor encoder
        x = self.up4(x, fc_skips[2])           # skip: MaxPool3 → 64×64
        x = self.up5(x, fc_skips[1])           # skip: MaxPool2 → 128×128
        x = self.up6(x, fc_skips[0])           # skip: MaxPool1 → 256×256

        # Final projection + resolution match
        x = self.final_conv(x)                 # 64 → 1
        x = self.upsample(x)                   # 256×256 → 512×512
        x = self.activation(x)                 # probability mask

        return x
