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
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4

os.makedirs("checkpoints", exist_ok=True)

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


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


spectral_data = []
for w in windows:
    w = normalize(w)
    spec = compute_cwt(w)
    spectral_data.append(spec)

spectral_data = np.array(spectral_data)


# =========================
# MODEL + DIFFUSION
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler()
diffusion = DiffusionModel(model, scheduler, DEVICE)

T = scheduler.timesteps

# 🔥 TRAINABLE EMBEDDINGS
timestep_embed = torch.nn.Embedding(T, 128).to(DEVICE)
regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)

# 🔥 FIX: include embeddings in optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) +
    list(timestep_embed.parameters()) +
    list(regime_embed.parameters()),
    lr=LR
)


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
        regime_batch = window_regimes[batch_idx]

        x = torch.tensor(batch).unsqueeze(1).float().to(DEVICE)
        regime_batch = torch.tensor(regime_batch).long().to(DEVICE)

        t = torch.randint(0, T, (x.size(0),), device=DEVICE)

        # 🔥 CORRECT EMBEDDING
        t_emb = timestep_embed(t)
        r_emb = regime_embed(regime_batch)

        emb = torch.cat([t_emb, r_emb], dim=1)  # 256

        loss = diffusion.loss(x, emb, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(indices) // BATCH_SIZE)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")

    # 🔥 SAVE EVERYTHING (MODEL + EMBEDDINGS)
    torch.save({
        "model": model.state_dict(),
        "t_embed": timestep_embed.state_dict(),
        "r_embed": regime_embed.state_dict()
    }, f"checkpoints/ema_epoch_{epoch+1}.pt")
