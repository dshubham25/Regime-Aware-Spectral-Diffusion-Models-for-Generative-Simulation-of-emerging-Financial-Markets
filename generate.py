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
# EMBEDDINGS
# =========================
def get_timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) *
        torch.arange(0, half) / half
    ).to(t.device)

    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)


# =========================
# NORMALIZATION
# =========================
def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-6)


# =========================
# PLOT
# =========================
def plot_sample(sample, title, filename):
    os.makedirs("generated", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(sample, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)

    plt.savefig(f"generated/{filename}", dpi=300)
    plt.close()


# =========================
# LOAD MODEL
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=200)

model.load_state_dict(torch.load("checkpoints/final_epoch_30.pt", map_location=DEVICE))
model.eval()

T = scheduler.timesteps


# =========================
# SHAPE
# =========================
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

sample = compute_cwt(normalize(windows[0]))

if sample.ndim == 3:
    sample = sample.mean(axis=0)

sample = sample / (np.max(np.abs(sample)) + 1e-6)

shape = torch.tensor(sample).unsqueeze(0).unsqueeze(0).shape


# =========================
# SAMPLING
# =========================
@torch.no_grad()
def sample(shape, regime_label=0):
    x = torch.randn(shape).to(DEVICE)

    alphas = scheduler.alphas.to(DEVICE)
    alpha_hat = scheduler.alpha_hat.to(DEVICE)
    betas = scheduler.betas.to(DEVICE)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=DEVICE)

        t_emb = get_timestep_embedding(t_tensor, 128)
        r_emb = regime_embed(torch.tensor([regime_label], device=DEVICE))

        emb = torch.cat([t_emb, r_emb], dim=1)

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
# GENERATE BOTH REGIMES
# =========================
stable = sample(shape, regime_label=0)
crisis = sample(shape, regime_label=2)

plot_sample(stable, "Stable Regime", "stable_final2.png")
plot_sample(crisis, "Crisis Regime", "crisis_final2.png")

print("Generation complete.")
