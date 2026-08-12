"""Preprocessing and transform builders for BrainScanAI Phase 1C."""

from __future__ import annotations

from typing import Iterable

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_image_size(image_size: Iterable[int]) -> tuple[int, int]:
    size_tuple = tuple(int(value) for value in image_size)
    if len(size_tuple) != 2:
        raise ValueError(f"image_size must contain exactly two integers, got {size_tuple}")
    return size_tuple[0], size_tuple[1]


def build_train_transform(image_size: Iterable[int]) -> transforms.Compose:
    """Build conservative train-time transforms for RGB MRI slices.

    Augmentations are intentionally light in Phase 1C:
    - resize to the configured transfer-learning size
    - small random rotation
    - tensor conversion
    - ImageNet normalization for pretrained-backbone compatibility
    """

    height, width = _normalize_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: Iterable[int]) -> transforms.Compose:
    """Build deterministic validation/test transforms."""

    height, width = _normalize_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
