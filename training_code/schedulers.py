import math
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, CosineAnnealingLR, StepLR


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps and self.warmup_steps > 0:
            scale = step / float(self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]


def build_scheduler(optimizer, name: str, total_steps: int, warmup_steps: int = 0, step_size: int = 0):
    name = name.lower()
    if name == "cosine":
        return WarmupCosineLR(optimizer, warmup_steps=warmup_steps, max_steps=total_steps)
    if name == "onecycle":
        # OneCycle needs epochs/steps: we set div_factor such that max lr ~= base lr
        return OneCycleLR(optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=total_steps)
    if name == "step":
        step_size = step_size or max(1, total_steps // 3)
        return StepLR(optimizer, step_size=step_size, gamma=0.5)
    raise ValueError(f"Unknown scheduler: {name}")

