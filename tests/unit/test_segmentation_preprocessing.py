from __future__ import annotations

import numpy as np
import torch

from brainscan.segmentation.data.preprocessing import (
    SegmentationAugmentationConfig,
    SegmentationAugmenter,
    build_binary_mask,
    zscore_normalize_nonzero,
)


def test_nonzero_zscore_normalization_is_finite_and_preserves_background() -> None:
    volume = np.zeros((4, 4, 2), dtype=np.float32)
    volume[1:3, 1:3, :] = 10.0
    normalized = zscore_normalize_nonzero(volume)
    assert np.isfinite(normalized).all()
    assert float(normalized[0, 0, 0]) == 0.0


def test_binary_mask_generation_uses_mask_gt_zero() -> None:
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    binary = build_binary_mask(mask)
    assert binary.tolist() == [[0.0, 1.0], [1.0, 1.0]]


def test_augmentation_keeps_image_and_mask_aligned() -> None:
    augmenter = SegmentationAugmenter(
        SegmentationAugmentationConfig(
            horizontal_flip_probability=1.0,
            max_rotation_degrees=0.0,
            max_translation_pixels=0,
        )
    )
    image = torch.zeros((4, 8, 8), dtype=torch.float32)
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    image[:, 2:4, 1:3] = 1.0
    mask[:, 2:4, 1:3] = 1.0
    aug_image, aug_mask = augmenter(image, mask)
    image_region = torch.nonzero(aug_image[0] > 0.5, as_tuple=False)
    mask_region = torch.nonzero(aug_mask[0] > 0.5, as_tuple=False)
    assert torch.equal(image_region, mask_region)
