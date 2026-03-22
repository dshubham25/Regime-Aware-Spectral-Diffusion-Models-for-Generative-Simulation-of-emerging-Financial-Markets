import torch
import copy

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay

        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                model.parameters()
            ):
                ema_param.data = (
                    self.decay * ema_param.data +
                    (1 - self.decay) * model_param.data
                )