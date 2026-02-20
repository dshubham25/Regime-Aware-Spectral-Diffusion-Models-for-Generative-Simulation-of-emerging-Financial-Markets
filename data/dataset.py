import torch
from torch.utils.data import Dataset
import numpy as np
from data.wavelet import compute_cwt


class SpectralDataset(Dataset):
    def __init__(self, windows, window_regimes):
        self.windows = windows
        self.window_regimes = window_regimes

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        # Normalize per window (NO leakage)
        window = (window - window.mean()) / (window.std() + 1e-6)

        spectral = compute_cwt(window)
        spectral = torch.tensor(spectral, dtype=torch.float32)
        spectral = spectral.unsqueeze(0)  # (1, 64, 160)

        regime = torch.tensor(self.window_regimes[idx], dtype=torch.long)

        return spectral, regime
