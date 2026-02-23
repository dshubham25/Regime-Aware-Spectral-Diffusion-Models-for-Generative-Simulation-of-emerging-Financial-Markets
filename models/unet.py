import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        emb_out = self.emb_proj(emb)
        h = h + emb_out[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.down1 = ResBlock(1, 64, emb_dim)
        self.down2 = ResBlock(64, 128, emb_dim)

        self.mid = ResBlock(128, 128, emb_dim)

        self.up1 = ResBlock(128, 64, emb_dim)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, emb):
        d1 = self.down1(x, emb)
        d2 = self.down2(d1, emb)

        mid = self.mid(d2, emb)

        up = self.up1(mid, emb)

        return self.out(up)
