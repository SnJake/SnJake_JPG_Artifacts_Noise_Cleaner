import math
from copy import deepcopy

import torch


class ModelEma:
    """Exponential Moving Average (EMA) of model weights for evaluation/inference.
    Keeps a separate copy of the model, updated after each optimizer step.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device: str | None = None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))

