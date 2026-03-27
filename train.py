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
EPOCHS = 5   # start small
BATCH_SIZE = 16
LR = 3e-4    # ✅ increased

os.makedirs("checkpoints", exist_ok=True)


# =========================
# SINUSOIDAL EMBEDDING
# =========================
def get_timestep_embedding(t, dim=128):
    half = dim // 2

    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) *
        torch.arange(0, half) / half
    ).to(t.device)

    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# =========================
# NORMALIZATION (CRITICAL FIX)
# =========================
def normalize(x):
    x = np.abs(x)
    x = np.log1p(x)   # ✅ fixes collapse
    return x


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


spectral_data = []
for w in windows:
    w = normalize(w)
    spec = compute_cwt(w)
    spectral_data.append(spec)

spectral_data = np.array(spectral_data)

print("Dataset size:", len(spectral_data))


# =========================
# MODEL
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler()   # 200 steps
diffusion = DiffusionModel(model, scheduler, DEVICE)

T = scheduler.timesteps

regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(regime_embed.parameters()),
    lr=LR
)


# =========================
# TRAIN
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

        t_emb = get_timestep_embedding(t, 128)
        r_emb = regime_embed(regime_batch)
        emb = torch.cat([t_emb, r_emb], dim=1)

        # DEBUG
        noisy, noise = diffusion.add_noise(x, t)
        noise_pred = model(noisy, emb)

        if i == 0:
            print("noise mean:", noise.mean().item())
            print("pred mean:", noise_pred.mean().item())

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save({
        "model": model.state_dict(),
        "r_embed": regime_embed.state_dict()
    }, f"checkpoints/ema_epoch_{epoch+1}.pt")
