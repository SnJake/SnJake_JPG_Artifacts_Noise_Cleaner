import random
from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

from utils import list_images, read_image, random_crop_pair, pil_to_torch, degrade


class DegradeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        patch_size: int = 256,
        jpeg_min: int = 5,
        jpeg_max: int = 75,
        noise_std: Tuple[float, float] = (0.0, 10.0),
        augment: bool = True,
        seed: int | None = None,
        files: list[Path] | None = None,
        clean_prob: float = 0.0,
    ):
        super().__init__()
        self.root = Path(root)
        self.files = files if files is not None else list_images(self.root)
        self.patch_size = patch_size
        self.jpeg_min = jpeg_min
        self.jpeg_max = jpeg_max
        self.noise_std = noise_std
        self.augment = augment
        self.rng = random.Random(seed)
        self._bad: set[Path] = set()
        self.clean_prob = float(clean_prob)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tries = 0
        while True:
            tries += 1
            path = self.files[idx]
            try:
                img = read_image(path)
                img = random_crop_pair(img, self.patch_size)

                clean = img
                # Deterministic per call if seed set and idx used, else random
                if self.rng is not None:
                    state = random.getstate()
                    random.seed(self.rng.randint(0, 2**31 - 1))
                is_clean = 0
                if random.random() < self.clean_prob:
                    noisy = clean
                    is_clean = 1
                else:
                    noisy = degrade(clean, self.jpeg_min, self.jpeg_max, self.noise_std)
                if self.rng is not None:
                    random.setstate(state)

                if self.augment:
                    # Random flips/rot90
                    if random.random() < 0.5:
                        clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
                        noisy = noisy.transpose(Image.FLIP_LEFT_RIGHT)
                    k = random.randint(0, 3)
                    if k:
                        clean = clean.rotate(90 * k, expand=False)
                        noisy = noisy.rotate(90 * k, expand=False)

                clean_t = pil_to_torch(clean)
                noisy_t = pil_to_torch(noisy)
                # Return a mask flag marking clean-as-noisy samples for identity loss
                return noisy_t, clean_t, torch.tensor(is_clean, dtype=torch.uint8)
            except (UnidentifiedImageError, OSError, ValueError):
                self._bad.add(path)
                # пробуем другой индекс
                idx = random.randrange(len(self.files))
                if tries >= 20:
                    raise
