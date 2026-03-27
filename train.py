import torch
import numpy as np
from tqdm import tqdm
import os

from models.unet import SimpleUNet
from models.diffusion import DiffusionModel
from models.scheduler import CosineScheduler

from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt
from data.regime import compute_volatility, compute_drawdown, assign_regimes


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 16
LR = 2e-4

os.makedirs("checkpoints", exist_ok=True)


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
# NORMALIZATION (FIXED)
# =========================
def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-6)


# =========================
# LOAD DATA
# =========================
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

volatility = compute_volatility(returns)
drawdown = compute_drawdown(df["Close"].values[1:])
regimes = assign_regimes(volatility, drawdown)

window_regimes = regimes[-len(windows):]

# ✅ USE FULL DATA (NO FILTERING)
windows = windows[:1000]
window_regimes = window_regimes[:1000]

print("Training samples:", len(windows))


# =========================
# CREATE SPECTRAL DATA
# =========================
spectral_data = []

for w in windows:
    w = normalize(w)

    spec = compute_cwt(w)

    # FIX SHAPE
    if spec.ndim == 3:
        spec = spec.mean(axis=0)

    # FIX SCALE
    spec = spec / (np.max(np.abs(spec)) + 1e-6)

    spectral_data.append(spec)

spectral_data = np.array(spectral_data)

print("DATA STD:", np.std(spectral_data))  # DEBUG


# =========================
# MODEL
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=200)
diffusion = DiffusionModel(model, scheduler, DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(regime_embed.parameters()),
    lr=LR
)

T = scheduler.timesteps


# =========================
# TRAIN LOOP
# =========================
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    indices = np.random.permutation(len(spectral_data))

    for i in tqdm(range(0, len(indices), BATCH_SIZE)):
        batch_idx = indices[i:i+BATCH_SIZE]

        batch = spectral_data[batch_idx]
        regimes_batch = window_regimes[batch_idx]

        x = torch.tensor(batch).unsqueeze(1).float().to(DEVICE)

        t = torch.randint(0, T, (x.size(0),), device=DEVICE)

        # embeddings
        t_emb = get_timestep_embedding(t, 128)
        r_emb = regime_embed(torch.tensor(regimes_batch, device=DEVICE))

        emb = torch.cat([t_emb, r_emb], dim=1)

        loss = diffusion.loss(x, emb, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints/final_epoch_{epoch+1}.pt")
