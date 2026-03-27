import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from models.unet import SimpleUNet
from models.diffusion import DiffusionModel
from models.scheduler import CosineScheduler

from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt
from data.regime import compute_volatility, compute_drawdown, assign_regimes

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# PLOT FUNCTION
def plot_sample(sample, title, filename):
    os.makedirs("generated", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(sample), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.savefig(f"generated/{filename}", dpi=300)
    plt.close()



# MODEL + SCHEDULER
model = SimpleUNet(emb_dim=256).to(DEVICE)

# FIX PATH IF NEEDED
model.load_state_dict(torch.load("checkpoints/ema_epoch_20.pt", map_location=DEVICE))

model.eval()

scheduler = CosineScheduler()
diffusion = DiffusionModel(model, scheduler, DEVICE)

T = scheduler.timesteps


# EMBEDDINGS 
timestep_embed = torch.nn.Embedding(T, 128).to(DEVICE)
regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)



# LOAD DATA
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

volatility = compute_volatility(returns)
drawdown = compute_drawdown(df["Close"].values[1:])
regimes = assign_regimes(volatility, drawdown)

window_regimes = regimes[-len(windows):]


# PICK WINDOWS
stable_idx = np.where(window_regimes == 0)[0][0]
crisis_idx = np.where(window_regimes == 2)[0][0]


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


stable_window = normalize(windows[stable_idx])
crisis_window = normalize(windows[crisis_idx])

stable_spec = compute_cwt(stable_window)
crisis_spec = compute_cwt(crisis_window)

stable_spec = torch.tensor(stable_spec).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
crisis_spec = torch.tensor(crisis_spec).unsqueeze(0).unsqueeze(0).float().to(DEVICE)


# FIXED SAMPLING FUNCTION

@torch.no_grad()
def sample_conditioned(shape, regime_label):
    x = torch.randn(shape).to(DEVICE)

    alphas = scheduler.alphas.to(DEVICE)
    alpha_hat = scheduler.alpha_hat.to(DEVICE)
    betas = scheduler.betas.to(DEVICE)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=DEVICE)

        t_emb = timestep_embed(t_tensor)
        r_emb = regime_embed(torch.tensor([regime_label], device=DEVICE))

        emb = torch.cat([t_emb, r_emb], dim=1)

        noise_pred = model(x, emb)

        a = alphas[t]
        ah = alpha_hat[t]
        b = betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(a)) * (
            x - ((1 - a) / torch.sqrt(1 - ah)) * noise_pred
        ) + torch.sqrt(b) * noise

    return x.squeeze().cpu().numpy()



# GENERATE

shape = stable_spec.shape

stable_sample = sample_conditioned(shape, 0)
crisis_sample = sample_conditioned(shape, 2)


# SAVE OUTPUT

plot_sample(stable_sample, "Generated Stable Spectral Map", "stable7.png")
plot_sample(crisis_sample, "Generated Crisis Spectral Map", "crisis7.png")


print("✅ Generation complete. Check 'generated/' folder.")
