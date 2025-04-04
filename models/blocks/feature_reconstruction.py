import torch
import torch.nn as nn

class FeatureReconstruction(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.linear = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        # x: B × N × D
        B, N, D = x.shape
        H = W = int(N ** 0.5)  # assuming square patch grid
        x = self.linear(x)             # → B × N × out_channels
        x = x.transpose(1, 2)          # → B × out_channels × N
        x = x.reshape(B, self.out_channels, H, W)  # → B × C × H × W
        return x
