# SnJakeArtifactsRemover — ComfyUI node for JPEG/noise artifacts removal
# Author: SnJake; Model: lightweight UNetRestorer
# Drop this folder into: ComfyUI/custom_nodes/SnJakeArtifactsRemover
# Weights default dir: ComfyUI/models/artifacts_remover
import os
import math
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


# -----------------------
#  Minimal UNetRestorer
# -----------------------

def conv3x3(in_ch, out_ch, stride=1, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)


class ResidualBlock(nn.Module):
    def __init__(self, ch, groups=1, expansion=2):
        super().__init__()
        mid = ch * expansion
        self.conv1 = conv3x3(ch, mid, groups=groups)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(mid, ch, groups=groups)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + identity


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.pixel_shuffle(x, 2)
        return self.act(x)


class CBAM(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // r, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(ch // r, 8), ch, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x))
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sa


class UNetBlock(nn.Module):
    def __init__(self, ch, n_blocks=2, use_cbam=False):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ch))
        self.body = nn.Sequential(*layers)
        self.cbam = CBAM(ch) if use_cbam else None

    def forward(self, x):
        x = self.body(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class UNetRestorer(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, depths=(2, 2, 4, 4), use_cbam=(False, True, True, False), out_ch=3):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Encoder
        self.enc0 = UNetBlock(chs[0], n_blocks=depths[0], use_cbam=use_cbam[0])
        self.down0 = Down(chs[0], chs[1])
        self.enc1 = UNetBlock(chs[1], n_blocks=depths[1], use_cbam=use_cbam[1])
        self.down1 = Down(chs[1], chs[2])
        self.enc2 = UNetBlock(chs[2], n_blocks=depths[2], use_cbam=use_cbam[2])
        self.down2 = Down(chs[2], chs[3])
        self.enc3 = UNetBlock(chs[3], n_blocks=depths[3], use_cbam=use_cbam[3])

        # Decoder
        self.up2 = Up(chs[3], chs[2])
        self.dec2 = UNetBlock(chs[2] + chs[2], n_blocks=2, use_cbam=False)
        self.up1 = Up(chs[2] + chs[2], chs[1])
        self.dec1 = UNetBlock(chs[1] + chs[1], n_blocks=2, use_cbam=False)
        self.up0 = Up(chs[1] + chs[1], chs[0])
        self.dec0 = UNetBlock(chs[0] + chs[0], n_blocks=2, use_cbam=False)

        self.exit = nn.Conv2d(chs[0] + chs[0], out_ch, 3, padding=1)

    def forward(self, x):
        x0 = self.entry(x)
        e0 = self.enc0(x0)
        x1 = self.down0(e0)
        e1 = self.enc1(x1)
        x2 = self.down1(e1)
        e2 = self.enc2(x2)
        x3 = self.down2(e2)
        e3 = self.enc3(x3)

        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        if d0.shape[-2:] != e0.shape[-2:]:
            d0 = F.interpolate(d0, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        out = self.exit(d0)
        return x + out  # residual

# -----------------------
#  Utilities
# -----------------------

_VALID_EXTS = (".pt", ".pth", ".ckpt", ".safetensors")

def make_hann_window(tile: int, device: torch.device):
    w = torch.hann_window(tile, device=device, periodic=False)
    eps = 1e-3
    w = eps + (1.0 - eps) * w
    w2d = torch.outer(w, w)
    return w2d



def _make_edge_aware_window(tile: int, overlap: int, x0: int, x1: int, y0: int, y1: int, W: int, H: int, device, eps: float = 1e-3):
    # Build per-tile window that only tapers on sides that actually overlap with neighbors
    wx = torch.ones(tile, device=device)
    wy = torch.ones(tile, device=device)
    ov = max(0, min(overlap, tile))
    if ov > 0:
        t = torch.linspace(0.0, math.pi / 2.0, steps=ov, device=device)
        ramp = eps + (1.0 - eps) * torch.sin(t) ** 2  # 0..1
        # X dimension
        if x0 > 0:  # has left neighbor
            wx[:ov] = ramp
        if x1 < W:  # has right neighbor
            wx[-ov:] = ramp.flip(0)
        # Y dimension
        if y0 > 0:  # has top neighbor
            wy[:ov] = ramp
        if y1 < H:  # has bottom neighbor
            wy[-ov:] = ramp.flip(0)
    w2d = wy.view(-1, 1) * wx.view(1, -1)
    return w2d


def _resolve_models_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        base = folder_paths.models_dir
    else:
        base = os.path.join(os.getcwd(), "models")
    path = os.path.join(base, "artifacts_remover")
    os.makedirs(path, exist_ok=True)
    try:
        if hasattr(folder_paths, "add_model_search_path"):
            folder_paths.add_model_search_path("artifacts_remover", path)
    except Exception:
        pass
    return path


def _list_artifact_models_local() -> List[str]:
    # List all filenames with allowed extensions from artifacts_remover dir (non-recursive)
    root = _resolve_models_dir()
    try:
        names = [n for n in os.listdir(root) if n.lower().endswith(_VALID_EXTS) and os.path.isfile(os.path.join(root, n))]
    except Exception:
        names = []
    names.sort()
    return names if names else ["<none found>"]


# -----------------------
#  Tiled inference
# -----------------------

@torch.no_grad()
def _tile_process(img_chw: torch.Tensor, model: nn.Module, tile: int, overlap: int, amp_dtype, edge_aware_window: bool):
    device = next(model.parameters()).device
    C, H, W = img_chw.shape
    amp_enabled = (amp_dtype is not None) and (device.type == 'cuda')

    if tile <= 0:
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
            out = model(img_chw.unsqueeze(0).to(device, memory_format=torch.channels_last))
        return out.squeeze(0).cpu()

    stride = tile - overlap
    out = torch.zeros(1, C, H, W, device=device)
    weight = torch.zeros_like(out)

    if not edge_aware_window:
        win = make_hann_window(tile, device).unsqueeze(0).unsqueeze(0)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0 = y
            x0 = x
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            y0 = max(0, y1 - tile)
            x0 = max(0, x1 - tile)

            patch = img_chw[:, y0:y1, x0:x1].unsqueeze(0).to(device, memory_format=torch.channels_last)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
                pred = model(patch)
            # приведение размеров предсказания к фактическому окну
            th, tw = (y1 - y0), (x1 - x0)
            ph, pw = pred.shape[-2], pred.shape[-1]
            use_h, use_w = min(th, ph), min(tw, pw)
            if (ph != th) or (pw != tw):
                pred = pred[:, :, :use_h, :use_w]

            if edge_aware_window:
                w2d = _make_edge_aware_window(tile, overlap, x0, x1, y0, y1, W, H, device)
                w = w2d.unsqueeze(0).unsqueeze(0)[:, :, :use_h, :use_w]
            else:
                w = win[:, :, :use_h, :use_w]

            out[:, :, y0:y0+use_h, x0:x0+use_w] += pred * w
            weight[:, :, y0:y0+use_h, x0:x0+use_w] += w

    out = out / weight.clamp(min=1e-8)
    return out.squeeze(0).cpu()


# -----------------------
#  Model cache/loader
# -----------------------

_MODEL_CACHE: Dict[Tuple[str, int, str], nn.Module] = {}

def _resolve_amp_dtype(mode: str):
    mode = mode.lower()
    if mode == "none":
        return None
    if not torch.cuda.is_available():
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        return torch.bfloat16
    return torch.float16  # "fp16"


def _load_checkpoint_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if safe_load_file is None:
            raise ImportError("safetensors not installed; install 'safetensors' to load .safetensors files")
        return safe_load_file(path)
    # torch native
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_model(weights_path: str, base_ch: int, device: torch.device) -> nn.Module:
    key = (weights_path, base_ch, str(device))
    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m
    model = UNetRestorer(in_ch=3, out_ch=3, base_ch=base_ch)
    ckpt = _load_checkpoint_any(weights_path)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    _MODEL_CACHE[key] = model
    return model


# -----------------------
#  ComfyUI Node
# -----------------------

class SnJakeArtifactsRemover:
    @classmethod
    def INPUT_TYPES(cls):
        models_dir = _resolve_models_dir()
        names = _list_artifact_models_local()
        default_name = names[0] if names else "<none found>"
        default_path = os.path.join(models_dir, default_name) if default_name not in ("<none found>",) else os.path.join(models_dir, "last.pt")
        return {
            "required": {
                "image": ("IMAGE",),
                "weights_name": (names, {"default": default_name}),
                "weights_path": ("STRING", {"default": default_path}),
                "base_ch": ("INT", {"default": 64, "min": 16, "max": 256, "step": 8}),
                "tile": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 16}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 4}),
                # <-- Новые поля ввода -->
                "edge_aware_window": ("BOOLEAN", {"default": True}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "amp_dtype": (["auto", "bf16", "fp16", "none"], {"default": "auto"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "😎 SnJake/JPG & Noise Remover"

    def _resolve_weights(self, weights_name: str, weights_path: str) -> str:
        # Resolve by dropdown name first (local dir), else manual path
        root = _resolve_models_dir()
        path = None
        if weights_name and weights_name not in ("<none found>",):
            cand = os.path.join(root, weights_name)
            if os.path.isfile(cand):
                path = cand
        if path is None and folder_paths is not None and hasattr(folder_paths, "get_full_path"):
            try:
                path = folder_paths.get_full_path("artifacts_remover", weights_name)
            except Exception:
                path = None
        if path is None or not os.path.isfile(path):
            path = weights_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weights not found. name='{weights_name}', path='{weights_path}'")
        return path
    
    # <-- Метод обновлен -->
    def apply(self, image, weights_name, weights_path, base_ch, tile, overlap, edge_aware_window, blend, amp_dtype, device):
        if device == "auto":
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(device)

        amp = _resolve_amp_dtype(amp_dtype)

        wp = self._resolve_weights(weights_name, weights_path)
        model = _load_model(wp, base_ch, device_t)

        b, h, w, c = image.shape
        if c != 3:
            raise ValueError("Only 3-channel RGB images are supported")

        out_list = []
        for i in range(b):
            img = image[i].permute(2, 0, 1).contiguous()  # CHW
            out_chw = _tile_process(img, model, int(tile), int(overlap), amp, edge_aware_window)

            if blend > 0:
                b_blend = max(0.0, min(1.0, blend))
                out_chw = (1.0 - b_blend) * out_chw + b_blend * img

            out_img = out_chw.permute(1, 2, 0).contiguous()
            out_list.append(out_img.unsqueeze(0))

        result = torch.cat(out_list, dim=0)
        result = result.clamp(0, 1).to(image.dtype)
        return (result,)