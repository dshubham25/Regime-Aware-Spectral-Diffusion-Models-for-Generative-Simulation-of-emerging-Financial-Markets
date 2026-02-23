import torch
import torch.nn.functional as F


class DiffusionModel:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_hat = self.scheduler.alpha_hat[t].to(self.device)
        alpha_hat = alpha_hat[:, None, None, None]

        noisy = torch.sqrt(alpha_hat) * x + torch.sqrt(1 - alpha_hat) * noise
        return noisy, noise

    def loss(self, x, regime, t):
        noisy, noise = self.add_noise(x, t)

        emb = regime  # already embedded externally

        noise_pred = self.model(noisy, emb)

        # Mask: only future region (last 32 time steps)
        mask = torch.zeros_like(x)
        mask[:, :, :, -32:] = 1.0

        loss = F.mse_loss(noise_pred * mask, noise * mask)
        return loss
