import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from models.unet import SimpleUNet
from models.scheduler import CosineScheduler
from train import get_timestep_embedding, regime_embed, normalize

def plot_sample(sample, title, filename):
    os.makedirs("figures/generated", exist_ok=True)
    plt.imshow(sample, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"figures/generated/{filename}", dpi=300)
    plt.close()

# =========================
# LOAD MODEL
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=300)

# Load latest checkpoint
ckpts = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
latest_ckpt = ckpts[-1]
state = torch.load(f"checkpoints/{latest_ckpt}", map_location=DEVICE)
model.load_state_dict(state['model'])
regime_embed.load_state_dict(state['regime_embed'])
model.eval()
regime_embed.eval()
T = scheduler.timesteps

# =========================
# SHAPE
# =========================
from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt

df = load_nifty("data_files/Nifty50(2008-2025).csv")
returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)
sample = compute_cwt(normalize(windows[0]))
if sample.ndim == 3:
    sample = sample.mean(axis=0)
sample = (sample - sample.mean()) / (sample.std() + 1e-6)
shape = torch.tensor(sample).unsqueeze(0).unsqueeze(0).shape

# =========================
# SAMPLING FUNCTION
# =========================
@torch.no_grad()
def sample_diffusion(shape, regime_label=0, n_samples=4):
    samples = []
    alphas = scheduler.alphas.to(DEVICE)
    alpha_hat = scheduler.alpha_hat.to(DEVICE)
    betas = scheduler.betas.to(DEVICE)
    for _ in range(n_samples):
        x = torch.randn(shape).to(DEVICE)
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
            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ah)) * noise_pred) + torch.sqrt(b) * noise
        samples.append(x.squeeze().cpu().numpy())
    return np.stack(samples)

# =========================
# GENERATE & PLOT SAMPLES FOR EACH REGIME
# =========================
regime_names = {0: "Stable", 1: "Transition", 2: "Crisis"}
evaluation_summary = {
    "checkpoint": latest_ckpt,
    "timesteps": T,
    "regimes": {}
}

for regime in [0, 1, 2]:
    gen_samples = sample_diffusion(shape, regime, n_samples=4)
    for i, s in enumerate(gen_samples):
        plot_sample(s, f"{regime_names[regime]} Regime (Sample {i+1})", f"regime{regime}_sample{i+1}.png")

    regime_stats = {
        "label": regime,
        "name": regime_names[regime],
        "mean": float(gen_samples.mean()),
        "std": float(gen_samples.std()),
        "min": float(gen_samples.min()),
        "max": float(gen_samples.max()),
        "n_samples": int(gen_samples.shape[0])
    }
    evaluation_summary["regimes"][str(regime)] = regime_stats

    print(
        f"Regime {regime} ({regime_names[regime]}): "
        f"mean={regime_stats['mean']:.4f}, std={regime_stats['std']:.4f}, "
        f"min={regime_stats['min']:.4f}, max={regime_stats['max']:.4f}"
    )

print("All regime samples generated and saved in figures/generated/")

os.makedirs("results", exist_ok=True)
with open("results/evaluation_summary.json", "w", encoding="ascii") as f:
    json.dump(evaluation_summary, f, indent=2)

with open("results/evaluation_summary.txt", "w", encoding="ascii") as f:
    f.write(f"Checkpoint: {latest_ckpt}\n")
    f.write(f"Timesteps: {T}\n")
    for regime in [0, 1, 2]:
        stats = evaluation_summary["regimes"][str(regime)]
        f.write(
            f"Regime {regime} ({stats['name']}): "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"n_samples={stats['n_samples']}\n"
        )

print("Saved evaluation summary to results/evaluation_summary.json and .txt")
