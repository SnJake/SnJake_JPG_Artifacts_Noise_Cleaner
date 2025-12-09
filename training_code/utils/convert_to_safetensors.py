import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    print("[Ошибка] Требуется PyTorch: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors.torch import save_file as safetensors_save_file
except ImportError:
    print("[Ошибка] Требуется safetensors: pip install safetensors", file=sys.stderr)
    sys.exit(1)


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Убирает префикс 'module.' из ключей, если модель была сохранена через DataParallel."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_state_dict(ckpt: Any, key_selector: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Извлекает словарь весов из чекпоинта.
    Приоритет: 
    1. Если указан key_selector, ищем этот ключ.
    2. Иначе ищем 'ema' (обычно лучшее качество).
    3. Иначе ищем 'model'.
    4. Иначе ищем 'state_dict'.
    5. Иначе проверяем, является ли сам объект словарем тензоров.
    """
    
    # Если ckpt - это сразу словарь весов (flat dict)
    is_flat_dict = isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values())
    
    if is_flat_dict and not key_selector:
        print(f" -> Чекпоинт определен как плоский словарь весов.")
        return ckpt

    if not isinstance(ckpt, dict):
        raise ValueError("Чекпоинт не является словарем.")

    # Логика автоматического выбора
    keys_to_try = []
    if key_selector:
        keys_to_try.append(key_selector)
    else:
        # Приоритет по умолчанию для скриптов обучения из этого проекта
        keys_to_try = ["ema", "model", "state_dict", "params"]

    for k in keys_to_try:
        if k in ckpt:
            candidate = ckpt[k]
            # Проверяем, что внутри словарь
            if isinstance(candidate, dict):
                # Простейшая проверка, что там есть тензоры
                # (иногда ema - это объект класса, но torch.save сохраняет его state_dict как dict)
                print(f" -> Найден ключ '{k}', используем его.")
                return candidate
            elif isinstance(candidate, torch.nn.Module):
                print(f" -> Ключ '{k}' является nn.Module, берем state_dict().")
                return candidate.state_dict()
    
    # Если ничего не нашли, но пользователь не просил конкретный ключ, 
    # пробуем отфильтровать тензоры из корня (на случай если там смешаны 'epoch' и веса)
    tensor_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if len(tensor_dict) > 0 and not key_selector:
        print(" -> Ключи модели не найдены явно, извлечены все тензоры из корня словаря.")
        return tensor_dict

    available_keys = list(ckpt.keys())
    raise ValueError(f"Не удалось найти веса модели. Доступные ключи: {available_keys}. "
                     f"Попробуйте указать --key.")


def convert_pt_to_safetensors(
    input_path: Path, 
    output_path: Path, 
    key: Optional[str] = None, 
    target_dtype: Optional[torch.dtype] = None
) -> None:
    
    print(f"Загрузка: {input_path}")
    # Загружаем на CPU, чтобы не занимать GPU
    ckpt = torch.load(str(input_path), map_location="cpu", weights_only=False)

    state_dict = get_state_dict(ckpt, key)
    state_dict = _strip_module_prefix(state_dict)

    # Подготовка тензоров
    clean_state_dict = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
            
        # Отвязываем от графа, переносим на CPU
        t = v.detach().cpu()
        
        # Приведение типов (например, float32 -> float16)
        if target_dtype is not None:
            # Не конвертируем целочисленные тензоры (например, буферы позиций или шаги)
            if t.dtype in (torch.float32, torch.float64):
                t = t.to(dtype=target_dtype)
        
        # Safetensors требует contiguous памяти
        if not t.is_contiguous():
            t = t.contiguous()
            
        clean_state_dict[k] = t

    if not clean_state_dict:
        raise ValueError("Словарь весов пуст после фильтрации.")

    print(f"Сохранение: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {"format": "pt"}
    if key:
        metadata["source_key"] = key
        
    safetensors_save_file(clean_state_dict, str(output_path), metadata=metadata)
    print(f"Успешно конвертировано {len(clean_state_dict)} тензоров.")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация модели из .pt/.pth в .safetensors"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Путь к входному файлу модели (.pt/.pth)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Путь к выходному файлу (по умолчанию: рядом с входным с расширением .safetensors)",
    )
    parser.add_argument(
        "--key", "-k",
        type=str,
        default=None,
        help="Ключ в словаре чекпоинта, где лежат веса (например: 'ema' или 'model'). "
             "Если не указан, скрипт попытается найти 'ema', затем 'model'."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Целевой тип данных для float-тензоров. 'fp16' уменьшает размер файла в 2 раза. (default: fp16)"
    )

    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"[Ошибка] Файл не найден: {in_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = in_path.with_suffix(".safetensors")

    # Определение dtype
    dt_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "auto": None
    }
    target_dtype = dt_map.get(args.dtype, None)

    try:
        convert_pt_to_safetensors(in_path, out_path, args.key, target_dtype)
    except Exception as e:
        print(f"[Ошибка] Конвертация не удалась: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()