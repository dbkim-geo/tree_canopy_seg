import torch
import torch.nn as nn

from models.encoder.dual_encoder import DualEncoder
from models.decoder.decoder import Decoder
from models.blocks.feature_fusion import FeatureFusion
from models.blocks.patch_embedding import PatchEmbedding
from models.blocks.feature_reconstruction import FeatureReconstruction
from models.blocks.transformer_encoder import TransformerEncoder

class DualStreamTransformerUNet(nn.Module):
    def __init__(
        self,
        in_channels_rgb=3,
        in_channels_fc=3,
        patch_size=16,
        embed_dim=256,
        transformer_depth=4,
        num_heads=4,
        out_channels=1
    ):
        super().__init__()

        self.encoder = DualEncoder(in_channels_rgb, in_channels_fc)
        self.fusion = FeatureFusion()
        self.patch_embed = PatchEmbedding(in_channels=512, embed_dim=embed_dim, patch_size=patch_size)
        self.transformer = TransformerEncoder(depth=transformer_depth, embed_dim=embed_dim, num_heads=num_heads)
        self.reconstruct = FeatureReconstruction(embed_dim=embed_dim, out_channels=256, patch_size=patch_size)
        self.decoder = Decoder()

    def forward(self, rgb, fake):
        enc = self.encoder(rgb, fake)
        fused = self.fusion(enc["rgb_out"], enc["fc_out"])  # B × 512 × 64 × 64

        tokens = self.patch_embed(fused)                    # B × 16 × 256
        encoded = self.transformer(tokens)                  # B × 16 × 256
        feat = self.reconstruct(encoded)                    # B × 256 × 4 × 4

        out = self.decoder(feat, enc["fc_skips"])           # B × 1 × 512 × 512
        return out
