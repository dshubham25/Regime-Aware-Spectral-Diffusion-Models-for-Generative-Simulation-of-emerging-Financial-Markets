# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# from models.unet import SimpleUNet
# from models.scheduler import CosineScheduler

# from data.load_data import load_nifty
# from data.features import compute_log_returns
# from data.windowing import create_windows
# from data.wavelet import compute_cwt


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# def get_timestep_embedding(t, dim=128):
#     half = dim // 2
#     freqs = torch.exp(
#         -torch.log(torch.tensor(10000.0)) *
#         torch.arange(0, half) / half
#     ).to(t.device)

#     args = t[:, None] * freqs[None]
#     return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)


# def normalize(x):
#     x = x - np.mean(x)
#     x = x / (np.std(x) + 1e-6)
#     return x


# def plot_sample(sample, title, filename):
#     os.makedirs("generated", exist_ok=True)

#     plt.imshow(sample, aspect='auto', cmap='viridis')
#     plt.colorbar()
#     plt.title(title)

#     plt.savefig(f"generated/{filename}", dpi=300)
#     plt.close()


# # =========================
# # LOAD MODEL
# # =========================
# model = SimpleUNet(emb_dim=256).to(DEVICE)
# scheduler = CosineScheduler(timesteps=300)

# model.load_state_dict(torch.load("checkpoints/final_epoch_40.pt", map_location=DEVICE))
# model.eval()

# T = scheduler.timesteps


# # =========================
# # SHAPE
# # =========================
# df = load_nifty("data_files/Nifty50(2008-2025).csv")

# returns = compute_log_returns(df["Close"].values)
# windows = create_windows(returns)

# sample = compute_cwt(normalize(windows[0]))

# if sample.ndim == 3:
#     sample = sample.mean(axis=0)

# sample = (sample - sample.mean()) / (sample.std() + 1e-6)

# shape = torch.tensor(sample).unsqueeze(0).unsqueeze(0).shape


# # =========================
# # SAMPLING
# # =========================
# @torch.no_grad()
# def sample(shape, regime_label=0):
#     x = torch.randn(shape).to(DEVICE)

#     alphas = scheduler.alphas.to(DEVICE)
#     alpha_hat = scheduler.alpha_hat.to(DEVICE)
#     betas = scheduler.betas.to(DEVICE)

#     for t in reversed(range(T)):
#         t_tensor = torch.tensor([t], device=DEVICE)

#         t_emb = get_timestep_embedding(t_tensor, 128)
#         r_emb = regime_embed(torch.tensor([regime_label], device=DEVICE))

#         emb = torch.cat([t_emb, r_emb], dim=1)

#         noise_pred = model(x, emb)

#         a = alphas[t]
#         ah = alpha_hat[t]
#         b = betas[t]

#         noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

#         x = (1 / torch.sqrt(a)) * (
#             x - ((1 - a) / torch.sqrt(1 - ah)) * noise_pred
#         ) + torch.sqrt(b) * noise

#     return x.squeeze().cpu().numpy()


# # =========================
# # GENERATE
# # =========================
# stable = sample(shape, 0)
# crisis = sample(shape, 2)

# plot_sample(stable, "Stable Regime", "stable_don.png")
# plot_sample(crisis, "Crisis Regime", "crisis_don.png")

# print("DONE")

#updated code

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
print(f"Using device: {DEVICE}")


def get_timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) *
        torch.arange(0, half) / half
    ).to(t.device)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)


def normalize(x):
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-6)
    return x


def plot_sample(sample, title, filename):
    os.makedirs("generated", exist_ok=True)

    s_min, s_max, s_std = sample.min(), sample.max(), sample.std()
    print(f"  [{title}] min={s_min:.4f}  max={s_max:.4f}  std={s_std:.4f}")

    if s_std < 1e-4:
        print(f"  ⚠ WARNING: [{title}] output is nearly flat — model has likely collapsed!")

    # ✅ Always stretch contrast so image is never blank
    if s_max - s_min > 1e-6:
        sample_display = (sample - s_min) / (s_max - s_min)
    else:
        sample_display = np.zeros_like(sample)

    plt.figure(figsize=(12, 5))
    im = plt.imshow(sample_display, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    cbar = plt.colorbar(im)
    cbar.set_label(f"Normalized intensity\n(raw range: [{s_min:.3f}, {s_max:.3f}])")
    plt.title(title, fontsize=14)
    plt.xlabel("Time Steps")
    plt.ylabel("Frequency Scale")
    plt.tight_layout()
    plt.savefig(f"generated/{filename}", dpi=300)
    plt.close()
    print(f"  Saved → generated/{filename}")


# =========================
# LOAD MODEL
# =========================
# ✅ FIXED: timesteps=300 matches train.py
model     = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=300)

checkpoint_path = "checkpoints/final_epoch_60.pt"
if not os.path.exists(checkpoint_path):
    # Fall back to epoch 40 if 60-epoch checkpoint not available
    checkpoint_path = "checkpoints/final_epoch_40.pt"
    print(f"60-epoch checkpoint not found, falling back to: {checkpoint_path}")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
regime_embed.load_state_dict(checkpoint['regime_embed'])

if 'avg_loss' in checkpoint:
    print(f"Checkpoint trained to epoch {checkpoint['epoch']} | avg_loss: {checkpoint['avg_loss']:.4f}")

model.eval()
T = scheduler.timesteps


# =========================
# DIAGNOSTIC — check model output is non-trivial
# =========================
print("\n=== MODEL DIAGNOSTIC ===")
with torch.no_grad():
    dummy = torch.randn(1, 1, 64, 160).to(DEVICE)
    t_test  = torch.tensor([150], device=DEVICE)
    t_emb   = get_timestep_embedding(t_test, 128)
    r_emb   = regime_embed(torch.tensor([2], device=DEVICE))
    emb     = torch.cat([t_emb, r_emb], dim=1)
    out     = model(dummy, emb)
    out_std = out.std().item()
    print(f"Single forward pass — min: {out.min().item():.4f}  max: {out.max().item():.4f}  std: {out_std:.4f}")
    if out_std < 0.05:
        print("⚠ CRITICAL: Model output std < 0.05 — model has collapsed!")
        print("  → You MUST retrain using the new train.py before generating.")
    else:
        print("✓ Model output looks healthy — proceeding with generation.")
print("========================\n")


# =========================
# GET SHAPE FROM REAL DATA
# =========================
df      = load_nifty("data_files/Nifty50(2008-2025).csv")
returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

cwt_sample = compute_cwt(normalize(windows[0]))
if cwt_sample.ndim == 3:
    cwt_sample = cwt_sample.mean(axis=0)
cwt_sample = (cwt_sample - cwt_sample.mean()) / (cwt_sample.std() + 1e-6)

shape = torch.tensor(cwt_sample).unsqueeze(0).unsqueeze(0).shape
print(f"Generation shape: {shape}")


# =========================
# SAMPLING LOOP
# =========================
@torch.no_grad()
def sample(shape, regime_label=0):
    x = torch.randn(shape).to(DEVICE)

    alphas    = scheduler.alphas.to(DEVICE)
    alpha_hat = scheduler.alpha_hat.to(DEVICE)
    betas     = scheduler.betas.to(DEVICE)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=DEVICE)

        t_emb = get_timestep_embedding(t_tensor, 128)
        r_emb = regime_embed(torch.tensor([regime_label], device=DEVICE))
        emb   = torch.cat([t_emb, r_emb], dim=1)

        noise_pred = model(x, emb)

        a  = alphas[t]
        ah = alpha_hat[t]
        b  = betas[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(a)) * (
            x - ((1 - a) / torch.sqrt(1 - ah)) * noise_pred
        ) + torch.sqrt(b) * noise

    return x.squeeze().cpu().numpy()


# =========================
# GENERATE
# =========================
print("Generating Stable Regime (label=0)...")
stable = sample(shape, regime_label=0)

print("Generating Crisis Regime (label=2)...")
crisis = sample(shape, regime_label=2)

print("\n=== GENERATION STATS ===")
print(f"Stable — min: {stable.min():.4f}  max: {stable.max():.4f}  std: {stable.std():.4f}")
print(f"Crisis — min: {crisis.min():.4f}  max: {crisis.max():.4f}  std: {crisis.std():.4f}")

plot_sample(stable, "Stable Regime", "stable_don.png")
plot_sample(crisis, "Crisis Regime", "crisis_don.png")

print("\nDONE")
