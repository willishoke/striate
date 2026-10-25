"""Device selection and image I/O helpers."""

import os

import numpy as np
import torch
from PIL import Image


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_image(tensor: torch.Tensor, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path
