import argparse
import sys
from pathlib import Path
from typing import Dict, Any


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }


def _extract_state_dict(obj: Any):
    try:
        import torch
        from torch import nn
    except Exception as e:  # pragma: no cover
        print("[Ошибка] Требуется PyTorch: pip install torch", file=sys.stderr)
        raise

    # Direct module -> state_dict
    if isinstance(obj, nn.Module):
        return obj.state_dict()

    # Plain state dict (all tensors)
    if isinstance(obj, dict) and obj:
        if all(hasattr(v, "device") and hasattr(v, "dtype") for v in obj.values()):
            return obj

        # Common wrappers
        for key in (
            "state_dict",
            "model_state_dict",
            "ema_state_dict",
            "model",
            "module",
            "net",
            "generator",
            "ema",
        ):
            if key in obj:
                cand = obj[key]
                # unwrap recursively
                try:
                    sd = _extract_state_dict(cand)
                    if sd:
                        return sd
                except Exception:
                    pass

        # Fallback: choose largest tensor-mapping sub-dict
        best = None
        best_count = -1
        for v in obj.values():
            if isinstance(v, dict) and v:
                if all(hasattr(t, "device") and hasattr(t, "dtype") for t in v.values()):
                    c = len(v)
                    if c > best_count:
                        best, best_count = v, c
        if best is not None:
            return best

    raise ValueError("Не удалось извлечь state_dict из чекпоинта")


def convert_pt_to_safetensors(input_path: Path, output_path: Path) -> None:
    try:
        import torch
    except Exception:
        print("[Ошибка] Требуется PyTorch: pip install torch", file=sys.stderr)
        raise

    try:
        from safetensors.torch import save_file as safetensors_save_file
    except Exception:
        print("[Ошибка] Требуется safetensors: pip install safetensors", file=sys.stderr)
        raise

    # Load checkpoint
    ckpt = torch.load(str(input_path), map_location="cpu")

    # Extract state dict
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_module_prefix(state_dict)

    # Ensure tensors are on CPU and contiguous
    def to_cpu_contiguous(x):
        try:
            if hasattr(x, "detach"):
                x = x.detach()
            # move to cpu
            if hasattr(x, "to"):
                x = x.to("cpu")
            elif hasattr(x, "cpu"):
                x = x.cpu()
            # pack non-contiguous views
            if hasattr(x, "is_contiguous") and hasattr(x, "contiguous"):
                if not x.is_contiguous():
                    x = x.contiguous()
        except Exception:
            # If anything goes wrong, return as is so the caller sees the real error
            return x
        return x

    cpu_state_dict = {k: to_cpu_contiguous(v) for k, v in state_dict.items()}

    # Save as safetensors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_save_file(cpu_state_dict, str(output_path), metadata={"converted_from": str(input_path)})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Конвертация модели из .pt/.pth в .safetensors (сохраняет рядом с входным файлом)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Путь к входному файлу модели (.pt/.pth)",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"[Ошибка] Файл не найден: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Output next to input with .safetensors extension
    out_path = in_path.with_suffix(".safetensors")

    try:
        convert_pt_to_safetensors(in_path, out_path)
    except Exception as e:
        print(f"[Ошибка] Конвертация не удалась: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
