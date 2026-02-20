import numpy as np
import torch
from torch.utils.data import DataLoader

from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt
from data.regime import compute_volatility, compute_drawdown, assign_regimes
from data.dataset import SpectralDataset
from visualization.plots import plot_spectral_map
from config import WINDOW_TOTAL, STEP_SIZE
## Load Data
df = load_nifty("data_files/Nifty50(2008-2025).csv")

print("Data Shape:", df.shape)
print("First Date:", df['Date'].iloc[0])
print("Last Date:", df['Date'].iloc[-1])

## Compute Returns
returns = compute_log_returns(df['Close'].values)
print("Returns Shape:", returns.shape)

## Create Sliding Windows
windows = create_windows(returns)
print("Windows Shape:", windows.shape)


## SAMPLE SPECTRAL TRANSFORM

sample_window = windows[0]
sample_window = (sample_window - sample_window.mean()) / (sample_window.std() + 1e-6)

spectral = compute_cwt(sample_window)
print("Spectral Shape:", spectral.shape)

plot_spectral_map(spectral, title="Sample Spectral Map", save_path="sample_spectral.png")





# plot_spectral_map(spectral, title="Sample Spectral Map")

## Compute Daily Regimes
volatility = compute_volatility(returns)
drawdown = compute_drawdown(df['Close'].values[1:])
regimes = assign_regimes(volatility, drawdown)

print("\nDaily Regime Distribution:")
print("Stable:", (regimes == 0).sum())
print("Volatile:", (regimes == 1).sum())
print("Crisis:", (regimes == 2).sum())

## Convert to Window-Level Regimes
window_regimes = []

for i in range(0, len(returns) - WINDOW_TOTAL, STEP_SIZE):
    regime_label = regimes[i + WINDOW_TOTAL - 1]
    window_regimes.append(regime_label)

window_regimes = np.array(window_regimes)

print("\nWindow-Level Regime Distribution:")
print("Stable:", (window_regimes == 0).sum())
print("Volatile:", (window_regimes == 1).sum())
print("Crisis:", (window_regimes == 2).sum())

print("\nTotal Windows:", len(windows))
print("Total Window Regimes:", len(window_regimes))

## Sanity check
#print("\nTotal Windows:", len(windows))
# print("Total Window Regimes:", len(window_regimes))

## Crisis vs Stable spectral Comparison
# Find one crisis window
crisis_indices = np.where(window_regimes == 2)[0]
stable_indices = np.where(window_regimes == 0)[0]

crisis_idx = crisis_indices[0]
stable_idx = stable_indices[0]

crisis_window = windows[crisis_idx]
stable_window = windows[stable_idx]

crisis_window = (crisis_window - crisis_window.mean()) / (crisis_window.std() + 1e-6)
stable_window = (stable_window - stable_window.mean()) / (stable_window.std() + 1e-6)

crisis_spectral = compute_cwt(crisis_window)
stable_spectral = compute_cwt(stable_window)

plot_spectral_map(crisis_spectral, title="Crisis Spectral Map", save_path="crisis_spectral.png")
plot_spectral_map(stable_spectral, title="Stable Spectral Map", save_path="stable_spectral.png")


def spectral_energy_ratio(spectral):
    low_freq = spectral[:20, :]
    high_freq = spectral[20:, :]
    return np.mean(np.abs(low_freq)) / np.mean(np.abs(high_freq))


print("\nSpectral Energy Ratio:")
print("Stable:", spectral_energy_ratio(stable_spectral))
print("Crisis:", spectral_energy_ratio(crisis_spectral))

dataset = SpectralDataset(windows, window_regimes)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

spectral_batch, regime_batch = next(iter(dataloader))

print("\nDataset Test:")
print("Batch Spectral Shape:", spectral_batch.shape)
print("Batch Regime Shape:", regime_batch.shape)
