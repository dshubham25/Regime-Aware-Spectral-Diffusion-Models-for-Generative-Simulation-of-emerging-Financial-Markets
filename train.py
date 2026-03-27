import torch
import numpy as np

from models.unet import SimpleUNet
from models.scheduler import CosineScheduler
from models.diffusion import DiffusionModel

from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(0, half) / half
    ).to(t.device)

    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def normalize(x):
    x = np.abs(x)
    x = np.log1p(x)
    return x


# =========================
# LOAD ONLY ONE SAMPLE
# =========================
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

sample = normalize(windows[0])
spec = compute_cwt(sample)

x = torch.tensor(spec).unsqueeze(0).unsqueeze(0).float().to(DEVICE)


# =========================
# MODEL
# =========================
model = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=50)  # VERY SMALL
diffusion = DiffusionModel(model, scheduler, DEVICE)

regime_embed = torch.nn.Embedding(3, 128).to(DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(regime_embed.parameters()),
    lr=1e-3   # HIGH LR for overfit
)

T = scheduler.timesteps


# =========================
# OVERFIT LOOP
# =========================
for step in range(1000):
    t = torch.randint(0, T, (1,), device=DEVICE)

    t_emb = get_timestep_embedding(t, 128)
    r_emb = regime_embed(torch.tensor([0], device=DEVICE))

    emb = torch.cat([t_emb, r_emb], dim=1)

    loss = diffusion.loss(x, emb, t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} | Loss {loss.item():.6f}")
