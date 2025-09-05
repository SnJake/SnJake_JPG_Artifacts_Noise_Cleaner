import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from models import UNetRestorer
from utils import list_images, read_image, pil_to_torch, torch_to_pil, make_hann_window
import math


def load_model(weights: str, device: torch.device, base_ch: int = 64):
    model = UNetRestorer(in_ch=3, out_ch=3, base_ch=base_ch)
    # Support .pt/.pth (pickle) and .safetensors
    if weights.lower().endswith('.safetensors'):
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except Exception as e:
            raise RuntimeError("Для загрузки .safetensors установите пакет safetensors: pip install safetensors") from e
        state_dict = safetensors_load_file(weights, device="cpu")
        model.load_state_dict(state_dict)
    else:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


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


def tile_process(img: torch.Tensor, model: torch.nn.Module, tile: int, overlap: int, amp_dtype, edge_aware_window: bool = False) -> torch.Tensor:
    # img: C,H,W in [0,1]
    device = next(model.parameters()).device
    C, H, W = img.shape
    amp_enabled = (amp_dtype is not None) and (device.type == 'cuda')
    if tile <= 0:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
            out = model(img.unsqueeze(0).to(device, memory_format=torch.channels_last))
        return out.squeeze(0).cpu()

    stride = tile - overlap
    out = torch.zeros(1, C, H, W, device=device)
    weight = torch.zeros_like(out)

    if not edge_aware_window:
        win = make_hann_window(tile, device)
        win = win.unsqueeze(0).unsqueeze(0)  # 1x1xTILExTILE

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y0 = y
                x0 = x
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)
                y0 = y1 - tile
                x0 = x1 - tile
                y0 = max(0, y0)
                x0 = max(0, x0)

                patch = img[:, y0:y1, x0:x1].unsqueeze(0).to(device, memory_format=torch.channels_last)
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
                    pred = model(patch)
                if edge_aware_window:
                    w2d = _make_edge_aware_window(tile, overlap, x0, x1, y0, y1, W, H, device)
                    w = w2d.unsqueeze(0).unsqueeze(0)
                else:
                    w = win[:, :, : (y1 - y0), : (x1 - x0)]
                out[:, :, y0:y1, x0:x1] += pred * w
                weight[:, :, y0:y1, x0:x1] += w

    out = out / weight.clamp(min=1e-8)
    return out.squeeze(0).cpu()


def main():
    ap = argparse.ArgumentParser(description="ArtifactsRemoverNN inference (tiled)")
    ap.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    ap.add_argument("--input", type=str, required=True, help="Input image or directory")
    ap.add_argument("--output", type=str, required=True, help="Output directory")
    ap.add_argument("--tile", type=int, default=512, help="Tile size (0 to disable tiling)")
    ap.add_argument("--overlap", type=int, default=64, help="Tile overlap for blending")
    ap.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "none"],
                    help="Autocast dtype for CUDA; auto prefers bf16")
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--blend", type=float, default=0.0, help="Blend factor with input [0..1] to restore fine details at inference")
    ap.add_argument("--edge-aware-window", action="store_true", help="Use edge-aware window so borders without neighbors are not tapered")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device, base_ch=args.base_ch)

    inp = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path]
    if inp.is_dir():
        files = list_images(inp)
    else:
        files = [inp]

    # Determine AMP dtype
    if torch.cuda.is_available() and device.type == 'cuda' and args.amp_dtype != 'none':
        if args.amp_dtype == 'auto':
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        elif args.amp_dtype == 'bf16':
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = None

    for p in tqdm(files, desc="Infer"):
        img = read_image(p)
        ten = pil_to_torch(img)  # C,H,W
        out = tile_process(ten, model, args.tile, args.overlap, amp_dtype, edge_aware_window=args.edge_aware_window)
        if args.blend > 0:
            b = max(0.0, min(1.0, args.blend))
            out = (1.0 - b) * out + b * ten
        res = torch_to_pil(out)
        save_path = out_dir / (p.stem + "_clean.png")
        res.save(save_path)


if __name__ == "__main__":
    main()
