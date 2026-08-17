"""Preprocessing and augmentation for multimodal BraTS slices."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


@dataclass(frozen=True)
class VolumeNormalizationStats:
    mean_nonzero: float
    std_nonzero: float
    nonzero_count: int


def compute_nonzero_stats(values: np.ndarray) -> VolumeNormalizationStats:
    """Compute stable non-zero z-score stats for a volume or slice."""
    nonzero = values != 0
    nonzero_count = int(np.count_nonzero(nonzero))
    if nonzero_count == 0:
        return VolumeNormalizationStats(mean_nonzero=0.0, std_nonzero=1.0, nonzero_count=0)

    selected = values[nonzero].astype(np.float64, copy=False)
    mean = float(selected.mean())
    std = float(selected.std())
    if std <= 1e-8:
        std = 1.0
    return VolumeNormalizationStats(mean_nonzero=mean, std_nonzero=std, nonzero_count=nonzero_count)


def accumulate_nonzero_stats(
    *,
    count: int,
    total: float,
    total_squares: float,
    values: np.ndarray,
) -> tuple[int, float, float]:
    """Incrementally accumulate non-zero statistics without retaining full volumes."""
    nonzero = values != 0
    if not np.any(nonzero):
        return count, total, total_squares
    selected = values[nonzero].astype(np.float64, copy=False)
    return (
        count + int(selected.size),
        total + float(selected.sum(dtype=np.float64)),
        total_squares + float(np.square(selected, dtype=np.float64).sum(dtype=np.float64)),
    )


def finalize_nonzero_stats(*, count: int, total: float, total_squares: float) -> VolumeNormalizationStats:
    """Finalize incremental non-zero stats into portable mean/std metadata."""
    if count <= 0:
        return VolumeNormalizationStats(mean_nonzero=0.0, std_nonzero=1.0, nonzero_count=0)
    mean = total / float(count)
    variance = max((total_squares / float(count)) - (mean * mean), 0.0)
    std = math.sqrt(variance)
    if std <= 1e-8:
        std = 1.0
    return VolumeNormalizationStats(mean_nonzero=float(mean), std_nonzero=float(std), nonzero_count=int(count))


def normalize_nonzero_slice(slice_array: np.ndarray, stats: VolumeNormalizationStats) -> np.ndarray:
    """Apply volume-level non-zero z-score stats to one 2D slice."""
    array = np.asarray(slice_array, dtype=np.float32)
    normalized = np.zeros_like(array, dtype=np.float32)
    nonzero = array != 0
    if not np.any(nonzero):
        return normalized
    normalized[nonzero] = (array[nonzero] - float(stats.mean_nonzero)) / float(stats.std_nonzero)
    return normalized


def zscore_normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Normalize one MRI modality using non-zero voxels only."""
    stats = compute_nonzero_stats(np.asarray(volume))
    array = np.asarray(volume, dtype=np.float32).copy()
    nonzero = array != 0
    if not np.any(nonzero):
        return np.zeros_like(array, dtype=np.float32)
    array[nonzero] = (array[nonzero] - float(stats.mean_nonzero)) / float(stats.std_nonzero)
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
