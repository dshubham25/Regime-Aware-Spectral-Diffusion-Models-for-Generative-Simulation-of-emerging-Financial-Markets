import matplotlib.pyplot as plt
import numpy as np
import os

def plot_spectral_map(spectral, title="Spectral Map", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(spectral), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency Scale")
    plt.tight_layout()

    if save_path:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{save_path}", dpi=300)
    
    plt.close()
