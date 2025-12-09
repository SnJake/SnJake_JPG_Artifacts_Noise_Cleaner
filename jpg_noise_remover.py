import os
import math
from typing import Tuple, Dict, List, Optional, NamedTuple

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

# HF client (optional; node works without it, but autoload disabled)
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    from .legacy_unet import UNetRestorer as LegacyUNetRestorer
    from .nafnet_unet import UNetRestorer as NAFNetRestorer
except Exception:
    # Allow running as a plain script
    from legacy_unet import UNetRestorer as LegacyUNetRestorer
    from nafnet_unet import UNetRestorer as NAFNetRestorer


# -----------------------
#  HF settings
# -----------------------
_HF_REPO_ID = "SnJake/JPG_Noise_Remover"
_HF_DEFAULT_WEIGHT_CANDIDATES = (
    "best_ema_v2_E11.pt",
    "best_ema_v2_E11_BF16.safetensors",
    "best_ema_15E.safetensors",
)


# -----------------------
#  Known weight metadata
# -----------------------

class _ModelSpec(NamedTuple):
    arch: str
    base_ch: int
    preferred_amp: Optional[str] = None


_KNOWN_MODEL_SPECS: Dict[str, _ModelSpec] = {
    "best_ema_v2_E11.pt": _ModelSpec("nafnet_v2", 64, None),
    "best_ema_v2_E11_BF16.safetensors": _ModelSpec("nafnet_v2", 64, "bf16"),
    "best_ema_15E.safetensors": _ModelSpec("legacy_v1", 64, None),
}


# -----------------------
#  Utilities
# -----------------------

_VALID_EXTS = (".pt", ".pth", ".ckpt", ".safetensors")


def _norm_name(name: Optional[str]) -> str:
    return name.lower() if isinstance(name, str) else ""


def _lookup_model_spec(name: Optional[str]) -> Optional[_ModelSpec]:
    norm = _norm_name(name)
    if not norm:
        return None
    for key, spec in _KNOWN_MODEL_SPECS.items():
        if key.lower() == norm:
            return spec
    return None


def _resolve_model_spec(weights_name: str, weights_path: str, user_base_ch: int) -> _ModelSpec:
    spec = _lookup_model_spec(weights_name)
    if spec is None and weights_path:
        spec = _lookup_model_spec(os.path.basename(weights_path))
    if spec is not None:
        return spec

    ref = _norm_name(weights_name or os.path.basename(weights_path or ""))
    if "v2" in ref or "naf" in ref:
        return _ModelSpec("nafnet_v2", user_base_ch, None)
    return _ModelSpec("legacy_v1", user_base_ch, None)


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
    amp_enabled = (amp_dtype is not None) and (device.type == "cuda")

    if tile <= 0:
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
            out = model(img_chw.unsqueeze(0).to(device, memory_format=torch.channels_last))
        return out.squeeze(0).cpu()

    stride = max(1, tile - overlap)
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
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype if amp_enabled else torch.float16, enabled=amp_enabled):
                pred = model(patch)

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

            out[:, :, y0:y0 + use_h, x0:x0 + use_w] += pred * w
            weight[:, :, y0:y0 + use_h, x0:x0 + use_w] += w

    out = out / weight.clamp(min=1e-8)
    return out.squeeze(0).cpu()


# -----------------------
#  Model cache/loader
# -----------------------

_MODEL_CACHE: Dict[Tuple[str, int, str, str], nn.Module] = {}


def _resolve_amp_dtype(mode: str, preferred_amp: Optional[str] = None):
    mode_in = (mode or "auto").lower()
    pref = preferred_amp.lower() if isinstance(preferred_amp, str) else None
    mode = pref if mode_in == "auto" and pref in ("bf16", "fp16", "auto") else mode_in
    if mode == "none":
        return None
    if not torch.cuda.is_available():
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16  # "fp16"


def _build_model(arch: str, base_ch: int) -> nn.Module:
    if arch == "nafnet_v2":
        return NAFNetRestorer(in_ch=3, out_ch=3, base_ch=base_ch)
    if arch == "legacy_v1":
        return LegacyUNetRestorer(in_ch=3, out_ch=3, base_ch=base_ch)
    raise ValueError(f"Unknown architecture '{arch}'")


def _load_checkpoint_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if safe_load_file is None:
            raise ImportError("safetensors not installed; install 'safetensors' to load .safetensors files")
        return safe_load_file(path)
    # torch native
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_model(weights_path: str, arch: str, base_ch: int, device: torch.device) -> nn.Module:
    key = (weights_path, base_ch, str(device), arch)
    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m
    model = _build_model(arch, base_ch)
    ckpt = _load_checkpoint_any(weights_path)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
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

        local_names = _list_artifact_models_local()

        remote_names: List[str] = list(_HF_DEFAULT_WEIGHT_CANDIDATES)
        for known in _KNOWN_MODEL_SPECS.keys():
            if known not in remote_names:
                remote_names.append(known)

        combined_names = set(remote_names)
        if local_names and local_names[0] != "<none found>":
            combined_names.update(local_names)

        names = sorted(list(combined_names))
        if not names:
            names = ["<none found>"]

        default_name = next((n for n in _HF_DEFAULT_WEIGHT_CANDIDATES if n in names), names[0])
        default_path = os.path.join(models_dir, default_name) if default_name not in ("<none found>",) else ""

        return {
            "required": {
                "image": ("IMAGE",),
                "weights_name": (names, {"default": default_name}),
                "weights_path": ("STRING", {"default": default_path, "multiline": False}),
                "base_ch": ("INT", {"default": 64, "min": 16, "max": 256, "step": 8}),
                "tile": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 16}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 4}),
                "edge_aware_window": ("BOOLEAN", {"default": True}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "amp_dtype": (["auto", "bf16", "fp16", "none"], {"default": "auto"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "SnJake/JPG & Noise Remover"

    def _resolve_weights(self, weights_name: str, weights_path: str) -> str:
        root = _resolve_models_dir()

        if weights_name and weights_name not in ("<none found>",):
            model_path = os.path.join(root, weights_name)

            if not os.path.isfile(model_path):
                print(f"[SnJakeArtifactsRemover] Model '{weights_name}' not found locally. Trying to download from Hugging Face...")
                if hf_hub_download is None:
                    raise ImportError("huggingface_hub is not installed. Please install it to download models automatically: pip install huggingface_hub")
                try:
                    hf_hub_download(
                        repo_id=_HF_REPO_ID,
                        filename=weights_name,
                        local_dir=root,
                        local_dir_use_symlinks=False,
                    )
                    print(f"[SnJakeArtifactsRemover] Download complete: {weights_name}")
                except Exception as e:
                    raise FileNotFoundError(f"Failed to download '{weights_name}' from Hugging Face. Error: {e}")

            if os.path.isfile(model_path):
                return model_path

        if weights_path and os.path.isfile(weights_path):
            return weights_path

        raise FileNotFoundError(f"Model weights not found. Name checked='{weights_name}' and path='{weights_path}'")

    def apply(self, image, weights_name, weights_path, base_ch, tile, overlap, edge_aware_window, blend, amp_dtype, device):
        if device == "auto":
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(device)

        wp = self._resolve_weights(weights_name, weights_path)
        spec = _resolve_model_spec(weights_name, wp, base_ch)
        amp = _resolve_amp_dtype(amp_dtype, preferred_amp=spec.preferred_amp)

        if spec.base_ch != base_ch:
            print(f"[SnJakeArtifactsRemover] Using base_ch={spec.base_ch} required by '{weights_name}' instead of input={base_ch}.")

        model = _load_model(wp, spec.arch, spec.base_ch, device_t)

        b, h, w, c = image.shape
        if c != 3:
            raise ValueError("Only 3-channel RGB images are supported.")

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
