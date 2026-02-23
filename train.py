import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.regime import compute_volatility, compute_drawdown, assign_regimes
from data.dataset import SpectralDataset

from models.scheduler import CosineScheduler
from models.unet import SimpleUNet
from models.diffusion import DiffusionModel
from models.embeddings import SinusoidalPositionEmbeddings, RegimeEmbedding
from config import WINDOW_TOTAL, STEP_SIZE


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 16
LR = 2e-4

# DATA PREP
df = load_nifty("data_files/Nifty50(2008-2025).csv")
returns = compute_log_returns(df['Close'].values)
windows = create_windows(returns)

volatility = compute_volatility(returns)
drawdown = compute_drawdown(df['Close'].values[1:])
regimes = assign_regimes(volatility, drawdown)

window_regimes = []
for i in range(0, len(returns) - WINDOW_TOTAL, STEP_SIZE):
    window_regimes.append(regimes[i + WINDOW_TOTAL - 1])

window_regimes = torch.tensor(window_regimes)

dataset = SpectralDataset(windows, window_regimes)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# MODEL SETUP
scheduler = CosineScheduler()
model = SimpleUNet(emb_dim=128).to(DEVICE)

timestep_embed = SinusoidalPositionEmbeddings(128).to(DEVICE)
regime_embed = RegimeEmbedding(num_regimes=3, emb_dim=128).to(DEVICE)

diffusion = DiffusionModel(model, scheduler, DEVICE)

optimizer = torch.optim.AdamW(
    list(model.parameters()) +
    list(timestep_embed.parameters()) +
    list(regime_embed.parameters()),
    lr=LR
)


# TRAINING LOOP
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for spectral, regime in tqdm(dataloader):
        spectral = spectral.to(DEVICE)
        regime = regime.to(DEVICE)

        t = torch.randint(0, scheduler.timesteps, (spectral.size(0),), device=DEVICE)

        t_emb = timestep_embed(t)
        r_emb = regime_embed(regime)

        emb = t_emb + r_emb

        loss = diffusion.loss(spectral, emb, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")
