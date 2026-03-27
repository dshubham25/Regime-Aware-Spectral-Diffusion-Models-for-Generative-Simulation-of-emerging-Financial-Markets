import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from models.unet import SimpleUNet
from models.scheduler import CosineScheduler

from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt
from data.regime import compute_volatility, compute_drawdown, assign_regimes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(0, half) / half
    ).to(t.device)

    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def normalize(x):
    x = np.abs(x)
    x = np.log1p(x)
    return x


def plot_sample(sample, title, filename):
    os.makedirs("generated", exist_ok=True)

    plt.imshow(sample, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"generated/{filename}", dpi=300)
    plt.close()


# =========================
# MODEL
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler()
T = scheduler.timesteps

regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)

ckpt = torch.load("checkpoints/ema_epoch_5.pt", map_location=DEVICE)

model.load_state_dict(ckpt["model"])
regime_embed.load_state_dict(ckpt["r_embed"])

model.eval()


# =========================
# SHAPE FROM DATA
# =========================
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

sample = compute_cwt(normalize(windows[0]))
shape = torch.tensor(sample).unsqueeze(0).unsqueeze(0).shape


# =========================
# SAMPLING
# =========================
@torch.no_grad()
def sample(shape, regime_label):
    x = torch.randn(shape).to(DEVICE)

    alphas = scheduler.alphas.to(DEVICE)
    alpha_hat = scheduler.alpha_hat.to(DEVICE)
    betas = scheduler.betas.to(DEVICE)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=DEVICE)

        emb = torch.cat([
            get_timestep_embedding(t_tensor, 128),
            regime_embed(torch.tensor([regime_label], device=DEVICE))
        ], dim=1)

        noise_pred = model(x, emb)

        a = alphas[t]
        ah = alpha_hat[t]
        b = betas[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(a)) * (
            x - ((1 - a) / torch.sqrt(1 - ah)) * noise_pred
        ) + torch.sqrt(b) * noise

    return x.squeeze().cpu().numpy()


stable = sample(shape, 0)
crisis = sample(shape, 2)

plot_sample(stable, "Stable", "stableUpdated1.png")
plot_sample(crisis, "Crisis", "crisisUpdated1.png")

