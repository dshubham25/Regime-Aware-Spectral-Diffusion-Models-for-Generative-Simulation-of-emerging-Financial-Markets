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
EPOCHS = 60            # ✅ increased from 40 → 60
BATCH_SIZE = 16
LR = 3e-4              # ✅ bumped from 1e-4
GRAD_CLIP = 1.0        # ✅ gradient clipping

os.makedirs("checkpoints", exist_ok=True)
print(f"Using device: {DEVICE}")


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
# NORMALIZATION
# =========================
def normalize(x):
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-6)
    return x


# =========================
# LOAD DATA
# =========================
df = load_nifty("data_files/Nifty50(2008-2025).csv")

returns = compute_log_returns(df["Close"].values)
windows = create_windows(returns)

volatility = compute_volatility(returns)
drawdown   = compute_drawdown(df["Close"].values[1:])
regimes    = assign_regimes(volatility, drawdown)

window_regimes = regimes[-len(windows):]

windows        = windows[:1000]
window_regimes = window_regimes[:1000]

print(f"Training samples : {len(windows)}")
print(f"Regime counts    : {dict(zip(*np.unique(window_regimes, return_counts=True)))}")


# =========================
# CREATE SPECTRAL DATA
# =========================
spectral_data = []

for w in windows:
    w    = normalize(w)
    spec = compute_cwt(w)

    if spec.ndim == 3:
        spec = spec.mean(axis=0)

    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spectral_data.append(spec)

spectral_data = np.array(spectral_data, dtype=np.float32)

print(f"Spectral data shape : {spectral_data.shape}")
print(f"DATA STD            : {np.std(spectral_data):.4f}")
print(f"MIN / MAX           : {spectral_data.min():.4f} / {spectral_data.max():.4f}")

# ✅ Sanity check — abort early if data is degenerate
assert np.std(spectral_data) > 0.1, "ERROR: Spectral data std too low — check CWT pipeline"


# =========================
# MODEL
# =========================
model     = SimpleUNet(emb_dim=256).to(DEVICE)
scheduler = CosineScheduler(timesteps=300)   # ✅ must match generate.py
diffusion = DiffusionModel(model, scheduler, DEVICE)

# ✅ AdamW with weight decay
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(regime_embed.parameters()),
    lr=LR,
    weight_decay=1e-4
)

# ✅ Cosine LR schedule — decays LR smoothly over training
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

T = scheduler.timesteps

# ✅ Quick forward-pass sanity check before training starts
with torch.no_grad():
    dummy_x   = torch.randn(2, 1, spectral_data.shape[1], spectral_data.shape[2]).to(DEVICE)
    dummy_t   = torch.randint(0, T, (2,), device=DEVICE)
    dummy_t_emb = get_timestep_embedding(dummy_t, 128)
    dummy_r_emb = regime_embed(torch.zeros(2, dtype=torch.long, device=DEVICE))
    dummy_emb   = torch.cat([dummy_t_emb, dummy_r_emb], dim=1)
    dummy_out   = model(dummy_x, dummy_emb)
    print(f"Sanity check — input: {dummy_x.shape}  output: {dummy_out.shape}")
    assert dummy_out.shape == dummy_x.shape, "ERROR: UNet output shape mismatch!"
    print("UNet forward pass OK")


# =========================
# TRAIN LOOP
# =========================
model.train()
loss_history = []

for epoch in range(EPOCHS):
    total_loss  = 0
    num_batches = 0
    indices     = np.random.permutation(len(spectral_data))

    for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_idx      = indices[i:i+BATCH_SIZE]
        batch          = spectral_data[batch_idx]
        regimes_batch  = window_regimes[batch_idx]

        x = torch.tensor(batch).unsqueeze(1).float().to(DEVICE)
        t = torch.randint(0, T, (x.size(0),), device=DEVICE)
        r = torch.from_numpy(regimes_batch).long().to(DEVICE)

        t_emb = get_timestep_embedding(t, 128)
        r_emb = regime_embed(r)
        emb   = torch.cat([t_emb, r_emb], dim=1)

        loss = diffusion.loss(x, emb, t)

        optimizer.zero_grad()
        loss.backward()

        # ✅ Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(regime_embed.parameters()),
            max_norm=GRAD_CLIP
        )

        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    lr_scheduler.step()

    avg_loss = total_loss / max(num_batches, 1)
    loss_history.append(avg_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

    if epoch > 5 and avg_loss > 0.95:
        print(f"  ⚠ WARNING: Loss still high — model may not be learning")

    # Save checkpoint every 10 epochs + final
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        torch.save({
            'model':        model.state_dict(),
            'regime_embed': regime_embed.state_dict(),
            'epoch':        epoch + 1,
            'avg_loss':     avg_loss
        }, f"checkpoints/final_epoch_{epoch+1}.pt")
        print(f"  Checkpoint saved: checkpoints/final_epoch_{epoch+1}.pt")

# Final summary
print("\n=== TRAINING COMPLETE ===")
print(f"Epoch 1  loss : {loss_history[0]:.4f}")
print(f"Epoch {EPOCHS} loss : {loss_history[-1]:.4f}")
improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
print(f"Improvement   : {improvement:.1f}%")
if improvement < 20:
    print("⚠ WARNING: Loss barely improved — consider more epochs or check data pipeline")
else:
    print("✓ Training looks healthy")
