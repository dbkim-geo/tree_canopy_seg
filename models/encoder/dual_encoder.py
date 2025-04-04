import torch
import torch.nn as nn
from models.encoder.encoder_block import EncoderBlock

class DualEncoder(nn.Module):
    def __init__(self, in_channels_rgb=3, in_channels_fc=3):
        super().__init__()

        # RGB encoder
        self.rgb_enc1 = EncoderBlock(in_channels_rgb, 64)
        self.rgb_enc2 = EncoderBlock(64, 128)
        self.rgb_enc3 = EncoderBlock(128, 256)

        # FakeColor encoder
        self.fc_enc1 = EncoderBlock(in_channels_fc, 64)
        self.fc_enc2 = EncoderBlock(64, 128)
        self.fc_enc3 = EncoderBlock(128, 256)

    def forward(self, x_rgb, x_fc):
        # RGB path
        rgb1, rgb1_pool = self.rgb_enc1(x_rgb)  # B×64×512×512, B×64×256×256
        rgb2, rgb2_pool = self.rgb_enc2(rgb1_pool)
        rgb3, rgb3_pool = self.rgb_enc3(rgb2_pool)

        # FakeColor path
        fc1, fc1_pool = self.fc_enc1(x_fc)
        fc2, fc2_pool = self.fc_enc2(fc1_pool)
        fc3, fc3_pool = self.fc_enc3(fc2_pool)

        # return feature maps needed
        return {
            "rgb_skips": [rgb1, rgb2, rgb3],
            "fc_skips": [fc1, fc2, fc3],
            "rgb_out": rgb3_pool,   # B × 256 × 64 × 64
            "fc_out": fc3_pool      # B × 256 × 64 × 64
        }
