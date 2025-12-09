import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

# Ensure project root is importable when running from utils/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer import load_model, tile_process  # type: ignore
import utils as u  # root-level utils.py


def _parse_amp_dtype(arg: str) -> torch.dtype | None:
    if torch.cuda.is_available() and arg != 'none':
        if arg == 'auto':
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if arg == 'bf16':
            return torch.bfloat16
        if arg == 'fp16':
            return torch.float16
    return None


def _choose_crop(img: Image.Image, mode: str, size: int, coords: Tuple[int, int, int, int] | None) -> Tuple[int, int, int, int]:
    W, H = img.size
    if mode == 'full':
        return 0, 0, W, H
    if mode == 'coords':
        if not coords:
            raise ValueError("Для режима coords укажите --x --y --w --h")
        x, y, w, h = coords
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return x, y, w, h
    # random / center
    size = max(1, min(size, min(W, H)))
    if mode == 'center':
        x = (W - size) // 2
        y = (H - size) // 2
        return x, y, size, size
    # random
    x = random.randint(0, W - size)
    y = random.randint(0, H - size)
    return x, y, size, size


def _hstack(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    W = img_left.width + img_right.width
    H = max(img_left.height, img_right.height)
    canvas = Image.new('RGB', (W, H))
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (img_left.width, 0))
    return canvas


def _vstack(img_top: Image.Image, img_bottom: Image.Image) -> Image.Image:
    W = max(img_top.width, img_bottom.width)
    H = img_top.height + img_bottom.height
    canvas = Image.new('RGB', (W, H))
    canvas.paste(img_top, (0, 0))
    canvas.paste(img_bottom, (0, img_top.height))
    return canvas


def process_file(p: Path, model, tile: int, overlap: int, amp_dtype, edge_aware_window: bool,
                 mode: str, size: int, coords: Tuple[int, int, int, int] | None,
                 orient: str, blend: float) -> Image.Image:
    img = u.read_image(p)

    if mode == 'full':
        ten = u.pil_to_torch(img)
        out = tile_process(ten, model, tile, overlap, amp_dtype, edge_aware_window=edge_aware_window)
        if blend > 0:
            b = max(0.0, min(1.0, blend))
            out = (1.0 - b) * out + b * ten
        res = u.torch_to_pil(out)
        return _hstack(img, res) if orient == 'h' else _vstack(img, res)

    x, y, w, h = _choose_crop(img, mode, size, coords)
    crop_in = img.crop((x, y, x + w, y + h))
    ten_crop = u.pil_to_torch(crop_in)
    out_crop = tile_process(ten_crop, model, tile, overlap, amp_dtype, edge_aware_window=edge_aware_window)
    if blend > 0:
        b = max(0.0, min(1.0, blend))
        out_crop = (1.0 - b) * out_crop + b * ten_crop
    crop_out = u.torch_to_pil(out_crop)

    return _hstack(crop_in, crop_out) if orient == 'h' else _vstack(crop_in, crop_out)


def main():
    ap = argparse.ArgumentParser(description="ArtifactsRemoverNN comparison (before/after)")
    ap.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    ap.add_argument("--input", type=str, required=True, help="Input image or directory")
    ap.add_argument("--output", type=str, required=True, help="Output directory")
    ap.add_argument("--tile", type=int, default=512, help="Tile size (0 to disable tiling)")
    ap.add_argument("--overlap", type=int, default=64, help="Tile overlap for blending")
    ap.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "none"],
                    help="Autocast dtype for CUDA; auto prefers bf16")
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--edge-aware-window", action="store_true", help="Use edge-aware window for tiles")
    ap.add_argument("--blend", type=float, default=0.0, help="Blend factor with input [0..1]")

    ap.add_argument("--mode", type=str, default="random", choices=["random", "center", "coords", "full"],
                    help="What to compare: random/center crop, specific coords, or full image")
    ap.add_argument("--crop-size", type=int, default=512, help="Crop size for random/center modes (pixels)")
    ap.add_argument("--x", type=int, default=0, help="Crop X (coords mode)")
    ap.add_argument("--y", type=int, default=0, help="Crop Y (coords mode)")
    ap.add_argument("--w", type=int, default=512, help="Crop width (coords mode)")
    ap.add_argument("--h", type=int, default=512, help="Crop height (coords mode)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible crops")
    ap.add_argument("--orient", type=str, default="h", choices=["h", "v"], help="Concatenate horizontally or vertically")

    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device, base_ch=args.base_ch)
    amp_dtype = _parse_amp_dtype(args.amp_dtype)

    inp = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path]
    if inp.is_dir():
        files = u.list_images(inp)
    else:
        files = [inp]

    for p in files:
        try:
            panel = process_file(
                p,
                model,
                tile=args.tile,
                overlap=args.overlap,
                amp_dtype=amp_dtype,
                edge_aware_window=args.edge_aware_window,
                mode=args.mode,
                size=args.crop_size,
                coords=(args.x, args.y, args.w, args.h) if args.mode == 'coords' else None,
                orient=args.orient,
                blend=args.blend,
            )
            suffix = f"_{args.mode}_cmp.png"
            save_path = out_dir / (p.stem + suffix)
            panel.save(save_path)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"[Ошибка] {p}: {e}")


if __name__ == "__main__":
    main()
