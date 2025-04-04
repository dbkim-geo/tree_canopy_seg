import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb_feat, fc_feat):
        # concat along channel dimension
        return torch.cat([rgb_feat, fc_feat], dim=1)  # B × (C1 + C2) × H × W
