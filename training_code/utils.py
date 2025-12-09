import os
import io
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root: Path | str) -> List[Path]:
    root = Path(root)
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def read_image(path: Path | str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_torch(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return ten


def torch_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def random_crop_pair(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        scale = max(size / w, size / h)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = new_w, new_h
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    arr = np.array(img, dtype=np.uint8)[:, :, ::-1]
    result, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not result:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = dec[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(dec)


def add_gaussian_noise(img: Image.Image, std_min: float, std_max: float) -> Image.Image:
    if std_max <= 0:
        return img
    std = random.uniform(std_min, std_max)
    if std <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_blur_downscale(img: Image.Image, prob: float, scale_min: float = 0.5) -> Image.Image:
    """Random downscale->upscale blur. scale_min controls blur strength (lower = blurrier)."""
    if random.random() >= prob:
        return img

    w, h = img.size
    scale = random.uniform(scale_min, 0.95)

    new_w, new_h = int(w * scale), int(h * scale)
    small = img.resize((new_w, new_h), Image.BICUBIC)
    blurred = small.resize((w, h), Image.BICUBIC)
    return blurred


def degrade(
    img: Image.Image,
    jpeg_min: int,
    jpeg_max: int,
    noise_std: Tuple[float, float],
    blur_prob: float = 0.0,
    blur_scale_min: float = 0.5,
) -> Image.Image:
    # 1. Blur: random downscale->upscale before JPEG/noise
    img = apply_blur_downscale(img, prob=blur_prob, scale_min=blur_scale_min)

    # 2. Random JPEG/Noise order
    q = random.randint(jpeg_min, jpeg_max)
    if random.random() < 0.5:
        img = apply_jpeg(img, q)
        img = add_gaussian_noise(img, noise_std[0], noise_std[1])
    else:
        img = add_gaussian_noise(img, noise_std[0], noise_std[1])
        img = apply_jpeg(img, q)
    return img


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def make_hann_window(tile: int, device: torch.device):
    w = torch.hann_window(tile, device=device, periodic=False)
    eps = 1e-3
    w = eps + (1.0 - eps) * w
    w2d = torch.outer(w, w)
    return w2d