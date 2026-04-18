import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# RESIDUAL BLOCK with FiLM conditioning
# =========================
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # ✅ FIXED: FiLM scale + shift instead of plain additive embedding
        self.emb_scale = nn.Linear(emb_dim, out_ch)
        self.emb_shift = nn.Linear(emb_dim, out_ch)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        # ✅ FIXED: Residual projection when channels differ
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        residual = self.residual(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # FiLM: modulate features with scale and shift from embedding
        scale = self.emb_scale(emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.emb_shift(emb).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + scale) + shift

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + residual   # residual connection


# =========================
# UNET
# =========================
class SimpleUNet(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        # DOWN
        self.down1 = Block(1, 64, emb_dim)
        self.pool1 = nn.AvgPool2d(2)       # ✅ AvgPool smoother than MaxPool for spectral maps

        self.down2 = Block(64, 128, emb_dim)
        self.pool2 = nn.AvgPool2d(2)

        self.down3 = Block(128, 256, emb_dim)
        self.pool3 = nn.AvgPool2d(2)

        # MID — two blocks for better bottleneck capacity
        self.mid1 = Block(256, 256, emb_dim)
        self.mid2 = Block(256, 256, emb_dim)

        # UP
        self.up1   = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.block1 = Block(256 + 256, 256, emb_dim)

        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block2 = Block(128 + 128, 128, emb_dim)

        self.up3   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block3 = Block(64 + 64, 64, emb_dim)

        # OUT
        self.out = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x, emb):
        # DOWN
        d1 = self.down1(x, emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, emb)
        p2 = self.pool2(d2)

        d3 = self.down3(p2, emb)
        p3 = self.pool3(d3)

        # MID
        m = self.mid1(p3, emb)
        m = self.mid2(m, emb)

        # UP — ✅ FIXED: pad to match skip connection sizes exactly
        u1 = self.up1(m)
        u1 = _pad_to_match(u1, d3)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.block1(u1, emb)

        u2 = self.up2(u1)
        u2 = _pad_to_match(u2, d2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.block2(u2, emb)

        u3 = self.up3(u2)
        u3 = _pad_to_match(u3, d1)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.block3(u3, emb)

        return self.out(u3)


def _pad_to_match(x, target):
    """Pad x spatially to match target (H, W) — fixes odd-dimension mismatches."""
    dh = target.shape[2] - x.shape[2]
    dw = target.shape[3] - x.shape[3]
    if dh != 0 or dw != 0:
        x = F.pad(x, [0, dw, 0, dh])
    return x
