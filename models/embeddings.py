import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        device = timestep.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RegimeEmbedding(nn.Module):
    def __init__(self, num_regimes=3, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_regimes, emb_dim)

    def forward(self, regime):
        return self.embedding(regime)
