import torch
import numpy as np

class CosineScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule()

        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def _cosine_beta_schedule(self):
        steps = self.timesteps + 1
        x = np.linspace(0, self.timesteps, steps)
        alphas_cumprod = np.cos(((x / self.timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.tensor(betas, dtype=torch.float32)
