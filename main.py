from data.load_data import load_nifty
from data.features import compute_log_returns
from data.windowing import create_windows
from data.wavelet import compute_cwt
from visualization.plots import plot_spectral_map
import numpy as np

# Load Data
df = load_nifty("data_files/Nifty50(2008-2025).csv")
print("Data Shape:", df.shape)
print("First Date:", df['Date'].iloc[0])
print("Last Date:", df['Date'].iloc[-1])

# Compute Returns
returns = compute_log_returns(df['Close'].values)
print("Returns Shape:", returns.shape)

# Create Sliding Windows
windows = create_windows(returns)
print("Windows Shape:", windows.shape)

# Take One Sample Window
sample_window = windows[0]

# Normalize per window
sample_window = (sample_window - sample_window.mean()) / (sample_window.std() + 1e-6)

# Compute Wavelet Transform
spectral = compute_cwt(sample_window)
print("Spectral Shape:", spectral.shape)

# Plot Spectral Map
plot_spectral_map(spectral, title="Sample Spectral Map")
