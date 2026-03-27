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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# EMBEDDING
# =========================
def get_timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) *
        torch.arange(0, half) / half
    ).to(t.device)

    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# =========================
# SAME NORMALIZATION
# =========================
def normalize(x):
    x = np.abs(x)
    x = np.log1p(x)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x


# =========================
# FIXED PLOT
# =========================
def plot_sample(sample, title, filename):
    os.makedirs("generated", exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.imshow(sample, aspect='auto', cmap='viridis', vmin=-2, vmax=2)
    plt.colorbar()
    plt.title(title)

    plt.savefig(f"generated/{filename}", dpi=300)
    plt.close()


# =========================
# MODEL
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=200)

model.load_state_dict(torch.load("checkpoints/debug_epoch_3.pt", map_location=DEVICE))
model.eval()

T = scheduler.timesteps


# =========================
# GET SHAPE
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
def sample(shape):
    x = torch.randn(shape).to(DEVICE)

    alphas = scheduler.alphas.to(DEVICE)
    alpha_hat = scheduler.alpha_hat.to(DEVICE)
    betas = scheduler.betas.to(DEVICE)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=DEVICE)

        emb = get_timestep_embedding(t_tensor, 256)

        noise_pred = model(x, emb)

        a = alphas[t]
        ah = alpha_hat[t]
        b = betas[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(a)) * (
            x - ((1 - a) / torch.sqrt(1 - ah)) * noise_pred
        ) + torch.sqrt(b) * noise

    return x.squeeze().cpu().numpy()


# =========================
# GENERATE
# =========================
sample_out = sample(shape)

plot_sample(sample_out, "Generated (Fixed)", "stable_fixed1.png")

print("Done.")
