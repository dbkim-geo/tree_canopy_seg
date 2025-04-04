import torch
import torch.nn as nn
from models.blocks.transformer_block import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self, depth, embed_dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
