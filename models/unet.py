import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# BASIC BLOCK
# =========================
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_layer = nn.Linear(emb_dim, out_ch)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # inject embedding
        emb_out = self.emb_layer(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h


# =========================
# U-NET
# =========================
class SimpleUNet(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        self.emb_dim = emb_dim

        # DOWN
        self.down1 = Block(1, 64, emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = Block(64, 128, emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = Block(128, 256, emb_dim)
        self.pool3 = nn.MaxPool2d(2)

        # BOTTLENECK
        self.mid = Block(256, 256, emb_dim)

        # UP
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block1 = Block(256, 128, emb_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block2 = Block(128, 64, emb_dim)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.block3 = Block(64, 32, emb_dim)

        # OUTPUT
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, emb):
        # DOWN
        d1 = self.down1(x, emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, emb)
        p2 = self.pool2(d2)

        d3 = self.down3(p2, emb)
        p3 = self.pool3(d3)

        # MID
        m = self.mid(p3, emb)

        # UP
        u1 = self.up1(m)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.block1(u1, emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.block2(u2, emb)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.block3(u3, emb)

        return self.out(u3)
