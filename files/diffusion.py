import torch
import torch.nn.functional as F


class DiffusionModel:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    def add_noise(self, x, t):
        noise = torch.randn_like(x)

        alpha_hat = self.scheduler.alpha_hat.to(self.device)
        alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)

        noisy = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise
        return noisy, noise

    def loss(self, x, emb, t):
        noisy, noise = self.add_noise(x, t)
        noise_pred = self.model(noisy, emb)

        # ✅ Huber loss — more robust than MSE for outlier noise values
        loss = F.huber_loss(noise_pred, noise, delta=1.0)

        return loss
