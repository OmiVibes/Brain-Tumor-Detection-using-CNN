"""Preprocessing and augmentation for multimodal BraTS slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


def zscore_normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Normalize one MRI modality using non-zero voxels only."""
    array = volume.astype(np.float32, copy=True)
    nonzero = array != 0
    if not np.any(nonzero):
        return np.zeros_like(array, dtype=np.float32)

    values = array[nonzero]
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-8:
        std = 1.0
    array[nonzero] = (array[nonzero] - mean) / std
    array[~nonzero] = 0.0
    return array


def build_binary_mask(mask_volume: np.ndarray) -> np.ndarray:
    """Derive the whole-tumor target at runtime without mutating source labels."""
    return (mask_volume > 0).astype(np.uint8, copy=False)


def stack_modalities(modality_volumes: list[np.ndarray]) -> np.ndarray:
    """Stack normalized modalities into [C, H, W, D]."""
    if not modality_volumes:
        raise ValueError("At least one modality volume is required.")
    first_shape = modality_volumes[0].shape
    for volume in modality_volumes:
        if volume.shape != first_shape:
            raise ValueError("All modality volumes must share the same shape.")
    return np.stack(modality_volumes, axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SegmentationAugmentationConfig:
    horizontal_flip_probability: float = 0.5
    max_rotation_degrees: float = 5.0
    max_translation_pixels: int = 4


class SegmentationAugmenter:
    """Apply conservative identical transforms to image and mask tensors."""

    def __init__(self, config: SegmentationAugmentationConfig) -> None:
        self.config = config

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(image.shape)}")
        if mask.ndim != 3:
            raise ValueError(f"Expected mask tensor [1,H,W], got shape {tuple(mask.shape)}")

        if torch.rand(1).item() < self.config.horizontal_flip_probability:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        angle = 0.0
        max_rotation = float(self.config.max_rotation_degrees)
        if max_rotation > 0.0:
            angle = float((torch.rand(1).item() * 2.0 - 1.0) * max_rotation)

        max_translate = int(self.config.max_translation_pixels)
        translate_x = 0
        translate_y = 0
        if max_translate > 0:
            translate_x = int(torch.randint(-max_translate, max_translate + 1, (1,)).item())
            translate_y = int(torch.randint(-max_translate, max_translate + 1, (1,)).item())

        if angle != 0.0 or translate_x != 0 or translate_y != 0:
            image = tvf.affine(
                image,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = tvf.affine(
                mask,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
            )

        mask = (mask > 0.5).to(dtype=torch.float32)
        return image.contiguous(), mask.contiguous()
