import torch
import numpy as np

from models.unet import SimpleUNet
from models.diffusion import Diffusion
from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt
from data.regime import compute_volatility, compute_drawdown, assign_regimes
from visualization.plots import plot_sample
from config import DEVICE

# =========================
# Load Model
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/ema_epoch_20.pt", map_location=DEVICE))
model.eval()

diffusion = Diffusion(device=DEVICE)

# =========================
# Embeddings
# =========================
timestep_embed = torch.nn.Embedding(1000, 128).to(DEVICE)
regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)

# =========================
# Load Data
# =========================
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

# =========================
# Regimes
# =========================
volatility = compute_volatility(returns)
drawdown = compute_drawdown(df["Close"].values[1:])
regimes = assign_regimes(volatility, drawdown)

# Align with windows
window_regimes = regimes[-len(windows):]

# =========================
# Select Different Past Windows
# =========================
stable_idx = np.where(window_regimes == 0)[0][0]
crisis_idx = np.where(window_regimes == 2)[0][0]

stable_window = windows[stable_idx]
crisis_window = windows[crisis_idx]

# Normalize
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)

stable_window = normalize(stable_window)
crisis_window = normalize(crisis_window)

# Convert to spectral
stable_spectral = compute_cwt(stable_window)
crisis_spectral = compute_cwt(crisis_window)

# Convert to tensor
stable_spectral = torch.tensor(stable_spectral, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
crisis_spectral = torch.tensor(crisis_spectral, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

# =========================
# Sampling Function
# =========================
@torch.no_grad()
def sample_conditioned(x_init, regime_label):
    x = x_init.clone()

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t], device=DEVICE)

        t_emb = timestep_embed(t_tensor)
        r_emb = regime_embed(torch.tensor([regime_label], device=DEVICE))

        # 🔥 IMPORTANT: CONCAT (256-dim)
        emb = torch.cat([t_emb, r_emb], dim=1)

        noise_pred = model(x, emb)

        x = diffusion.step(x, noise_pred, t)

    return x.squeeze().cpu().numpy()

# =========================
# Generate Samples
# =========================
stable_sample = sample_conditioned(stable_spectral, regime_label=0)
crisis_sample = sample_conditioned(crisis_spectral, regime_label=2)

# =========================
# Save Outputs
# =========================
plot_sample(stable_sample, "Generated Stable Spectral Map", "stable4.png")
plot_sample(crisis_sample, "Generated Crisis Spectral Map", "crisis4.png")
