import matplotlib.pyplot as plt
import numpy as np

def plot_spectral_map(spectral, title="Spectral Map"):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(spectral), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency Scale")
    plt.show()
