import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.unet import SimpleUNet
from models.scheduler import CosineScheduler
from models.embeddings import SinusoidalPositionEmbeddings, RegimeEmbedding
from data.wavelet import compute_cwt
from config import NUM_SCALES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# LOAD MODEL (EMA)
model = SimpleUNet(emb_dim=128).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/ema_epoch_20.pt", map_location=DEVICE))
model.eval()

scheduler = CosineScheduler()

timestep_embed = SinusoidalPositionEmbeddings(128).to(DEVICE)
regime_embed = RegimeEmbedding(num_regimes=3, emb_dim=128).to(DEVICE)


# SAMPLING FUNCTION
@torch.no_grad()
def sample(regime_label, steps=1000):
    x = torch.randn(1, 1, 64, 160).to(DEVICE)

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t], device=DEVICE)

        t_emb = timestep_embed(t_tensor)
        r_emb = regime_embed(torch.tensor([regime_label], device=DEVICE))
        emb = t_emb + r_emb

        noise_pred = model(x, emb)

        alpha = scheduler.alphas[t].to(DEVICE)
        alpha_hat = scheduler.alpha_hat[t].to(DEVICE)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred
        ) + torch.sqrt(1 - alpha) * noise

    return x


# GENERATE SAMPLES
stable_sample = sample(regime_label=0)
crisis_sample = sample(regime_label=2)


# VISUALIZE
def plot_sample(sample, title, filename):
    sample = sample.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(sample), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    os.makedirs("generated", exist_ok=True)
    plt.savefig(f"generated/{filename}", dpi=300)
    plt.close()

plot_sample(stable_sample, "Generated Stable Spectral Map")
plot_sample(crisis_sample, "Generated Crisis Spectral Map")