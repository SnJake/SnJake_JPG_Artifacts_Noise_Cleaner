import argparse
import math
import os
import random
from pathlib import Path
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import UNetRestorer
from dataset import DegradeDataset
from losses import CharbonnierLoss, MixL1SSIM, PerceptualLoss  # Import Perceptual
from losses import GradientLoss, HFENLoss  
from utils import psnr, list_images
from ema import ModelEma
from schedulers import build_scheduler


def parse_args():
    p = argparse.ArgumentParser(description="Train JPEG/noise artifacts removal network")
    p.add_argument("--data-root", type=str, required=False,
                   default=r"D:\\Models for Stable Diffusion\\Training\\TrainMultipleModelsPixelArt\\Non_pixel_art",
                   help="Path to clean images root")
    p.add_argument("--out-dir", type=str, default="runs/artifacts_remover", help="Output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--accum-steps", type=int, default=1)
    
    # Data Augmentation & Degradation
    p.add_argument("--clean-prob", type=float, default=0.0, help="Probability to use clean-as-noisy")
    p.add_argument("--blur-prob", type=float, default=0.0, help="Probability to blur input (for sharpening learning)")
    p.add_argument("--id-loss-w", type=float, default=0.0)

    p.add_argument("--jpeg-min", type=int, default=5)
    p.add_argument("--jpeg-max", type=int, default=75)
    p.add_argument("--noise-std", type=float, nargs=2, default=[0.0, 10.0], metavar=("MIN", "MAX"))
    p.add_argument("--blur-scale-min", type=float, default=0.5, help="Min scale for blur downscale (lower = blurrier)")

    # Optim
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "step", "none"])
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--step-size", type=int, default=0)

    # AMP/EMA
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "none"])
    p.add_argument("--ema-decay", type=float, default=0.999)

    # Losses
    p.add_argument("--mix-alpha", type=float, default=0.84)
    p.add_argument("--edge-loss-w", type=float, default=0.0)
    p.add_argument("--hfen-w", type=float, default=0.0)
    p.add_argument("--perceptual-w", type=float, default=0.0, help="Weight for VGG Perceptual Loss")

    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--resume", type=str, default="")

    return p.parse_args()


def build_dataloaders(args):
    files = list_images(args.data_root)
    n_total = len(files)
    n_val = max(1, int(n_total * args.val_ratio))

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(n_total, generator=g).tolist()
    val_idx = set(idx[:n_val])
    train_files = [files[i] for i in idx if i not in val_idx]
    val_files = [files[i] for i in idx if i in val_idx]

    train_ds = DegradeDataset(
        root=args.data_root,
        patch_size=args.patch_size,
        jpeg_min=args.jpeg_min,
        jpeg_max=args.jpeg_max,
        noise_std=tuple(args.noise_std),
        augment=True,
        seed=None,
        files=train_files,
        clean_prob=args.clean_prob,
        blur_prob=args.blur_prob, # <-- прокидываем
        blur_scale_min=args.blur_scale_min,
    )
    val_ds = DegradeDataset(
        root=args.data_root,
        patch_size=args.patch_size,
        jpeg_min=args.jpeg_min,
        jpeg_max=args.jpeg_max,
        noise_std=tuple(args.noise_std),
        augment=False,
        seed=args.seed + 123,
        files=val_files,
        clean_prob=0.0,
        blur_prob=args.blur_prob, # <-- Валидация тоже может быть размытой, чтобы чекать метрики
        blur_scale_min=args.blur_scale_min,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(min(2, args.num_workers) > 0),
    )
    return train_loader, val_loader


def save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_ckpt_if_any(model, optimizer, scaler, scheduler, ema, path: str):
    if not path:
        return 0, 0.0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "opt" in ckpt:
        optimizer.load_state_dict(ckpt["opt"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and "sched" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["sched"])
        except Exception:
            pass
    if ema is not None and "ema" in ckpt:
        ema.ema.load_state_dict(ckpt["ema"])
    start_epoch = ckpt.get("epoch", 0)
    best_psnr = ckpt.get("best_psnr", 0.0)
    return start_epoch, best_psnr


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    tb_dir = out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir.as_posix())

    train_loader, val_loader = build_dataloaders(args)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    model = UNetRestorer(in_ch=3, out_ch=3, base_ch=args.base_ch)
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    amp_enabled = (not args.no_amp) and (device.type == "cuda") and (args.amp_dtype.lower() != "none")
    if amp_enabled:
        if args.amp_dtype == "auto":
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        elif args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = None

    scaler = GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    if args.scheduler == "none":
        scheduler = None
    else:
        scheduler = build_scheduler(optimizer, args.scheduler, total_steps=total_steps, warmup_steps=args.warmup_steps, step_size=args.step_size)

    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # INIT LOSSES
    l_char = CharbonnierLoss().to(device)
    l_mix = MixL1SSIM(alpha=args.mix_alpha).to(device)
    
    l_edge = GradientLoss().to(device) if args.edge_loss_w > 0 else None
    l_hfen = HFENLoss().to(device) if args.hfen_w > 0 else None
    
    # Init Perceptual
    l_perc = None
    if args.perceptual_w > 0:
        print("Initializing VGG19 Perceptual Loss...")
        l_perc = PerceptualLoss().to(device).eval()

    start_epoch, best_psnr = load_ckpt_if_any(model, optimizer, scaler, scheduler, ema, args.resume)
    global_step = steps_per_epoch * start_epoch

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            if len(batch) == 3:
                noisy, clean, is_clean = batch
            else:
                noisy, clean = batch
                is_clean = torch.zeros((noisy.size(0),), dtype=torch.uint8)
            
            noisy = noisy.to(device, non_blocking=True, memory_format=torch.channels_last)
            clean = clean.to(device, non_blocking=True, memory_format=torch.channels_last)
            is_clean = is_clean.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=amp_dtype if amp_dtype is not None else torch.float16, enabled=amp_enabled):
                pred = model(noisy)
                
                # Base pixel loss
                loss = 0.7 * l_char(pred, clean) + 0.3 * l_mix(pred, clean)
                
                # Addons
                if l_edge is not None:
                    loss = loss + args.edge_loss_w * l_edge(pred, clean)
                if l_hfen is not None:
                    loss = loss + args.hfen_w * l_hfen(pred, clean)
                if l_perc is not None:
                    loss = loss + args.perceptual_w * l_perc(pred, clean)

                # Identity loss
                if args.id_loss_w > 0:
                    if is_clean.dtype != torch.float32:
                        m = is_clean.float()
                    else:
                        m = is_clean
                    if m.sum() > 0:
                        m = m.view(-1, 1, 1, 1)
                        id_l1 = torch.abs(pred - noisy)
                        denom = (m.sum() * id_l1.size(1) * id_l1.size(2) * id_l1.size(3)).clamp_min(1.0)
                        id_loss = (id_l1 * m).sum() / denom
                        loss = loss + args.id_loss_w * id_loss

            scaler.scale(loss / args.accum_steps).backward()

            if (i + 1) % args.accum_steps == 0:
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    scheduler.step()

            global_step += 1
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                writer.add_scalar("train/loss", running_loss / (i + 1), global_step)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_psnr = 0.0
            n = 0
            eval_model = ema.ema if ema is not None else model
            for batch in val_loader:
                if len(batch) == 3:
                    noisy, clean, _ = batch
                else:
                    noisy, clean = batch
                noisy = noisy.to(device, non_blocking=True, memory_format=torch.channels_last)
                clean = clean.to(device, non_blocking=True, memory_format=torch.channels_last)
                with autocast(device_type='cuda', dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
                    pred = eval_model(noisy)
                    # Simple validation metric without VGG to save time/mem
                    vloss = l_char(pred, clean)
                
                val_loss += vloss.item() * noisy.size(0)
                val_psnr += psnr(pred, clean) * noisy.size(0)
                n += noisy.size(0)

            val_loss /= max(1, n)
            val_psnr /= max(1, n)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/psnr", val_psnr, epoch)

        last_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
        }
        if scheduler is not None:
            try:
                last_state["sched"] = scheduler.state_dict()
            except Exception:
                pass
        if ema is not None:
            last_state["ema"] = ema.ema.state_dict()
        save_ckpt(last_state, Path(args.out_dir) / "last.pt")

        improved = val_psnr > best_psnr
        if improved:
            best_psnr = val_psnr
            save_ckpt(last_state, Path(args.out_dir) / "best.pt")
            if ema is not None:
                save_ckpt({"model": ema.ema.state_dict()}, Path(args.out_dir) / "best_ema.pt")

        tqdm.write(f"Epoch {epoch+1} done: val_psnr={val_psnr:.3f} (best={best_psnr:.3f})")

    writer.close()


if __name__ == "__main__":
    main()
