import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    # Simplified SSIM for training; 11x11 Gaussian approx with avgpool
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()


class MixL1SSIM(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = torch.abs(pred - target).mean()
        s = ssim(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * (1 - s)


class GradientLoss(nn.Module):
    """Edge-preserving L1 on spatial gradients to discourage oversmoothing."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # N,C,H,W tensors. Compute forward differences.
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        loss = (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()
        return loss


class HFENLoss(nn.Module):
    def __init__(self, channels: int = 3, sigma: float = 1.5):
        super().__init__()
        size = int(2 * round(3 * sigma) + 1)
        coords = torch.arange(size) - size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        log = (xx**2 + yy**2 - 2 * sigma**2) * g / (sigma**4)
        log = log - log.mean()
        k = log.view(1, 1, size, size)
        self.register_buffer("kernel", k.repeat(channels, 1, 1, 1))
        self.groups = channels

    def forward(self, pred, target):
        k = self.kernel.to(dtype=pred.dtype, device=pred.device)
        p = F.conv2d(pred, k, padding="same", groups=self.groups)
        t = F.conv2d(target, k, padding="same", groups=self.groups)
        return (p - t).abs().mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG19 features. Используем стандартные слои для perceptual loss.
        # weights=True загрузит ImageNet веса (requires internet on first run)
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        # Нам нужны только feature extractor слои
        self.features = vgg.features
        # Замораживаем веса
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Индексы слоев после ReLU: relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        # Это классический набор для style/perceptual loss
        self.layer_indices = {'3': 0.1, '8': 0.1, '17': 1.0, '26': 1.0, '35': 1.0}
        
        # Нормализация ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.max_hw = 224  # clamp VGG input spatial size to save VRAM/compute

    def forward(self, pred, target):
        # Вход должен быть [0, 1]. Нормализуем под VGG
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        if max(pred_norm.shape[-2:]) > self.max_hw:
            target_size = (self.max_hw, self.max_hw)
            pred_norm = F.interpolate(pred_norm, size=target_size, mode="bilinear", align_corners=False)
            target_norm = F.interpolate(target_norm, size=target_size, mode="bilinear", align_corners=False)
        
        loss = 0.0
        x = pred_norm
        y = target_norm
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            if str(i) in self.layer_indices:
                w = self.layer_indices[str(i)]
                loss += w * F.l1_loss(x, y)
            
            # Ранний выход, если прошли последний нужный слой (35)
            if i >= 35:
                break
        return loss