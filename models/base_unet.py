import torch
import torch.nn as nn

from models.encoder.encoder_block import EncoderBlock
from models.decoder.decoder import Decoder

class BaseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder (3단)
        self.enc1 = EncoderBlock(in_channels, 64)     # 512 → 256
        self.enc2 = EncoderBlock(64, 128)             # 256 → 128
        self.enc3 = EncoderBlock(128, 256)            # 128 → 64

        # Bottom (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder (UpBlock 3단 + Final)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 64 → 128
        self.dec1 = EncoderBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 128 → 256
        self.dec2 = EncoderBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 256 → 512
        self.dec3 = EncoderBlock(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1, p1 = self.enc1(x)  # e1: B×64×512×512, p1: B×64×256×256
        e2, p2 = self.enc2(p1)
        e3, p3 = self.enc3(p2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        x = self.up1(b)
        x = torch.cat([x, e3], dim=1)
        _, x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        _, x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, e1], dim=1)
        _, x = self.dec3(x)

        return self.final(x)  # B × 1 × 512 × 512
