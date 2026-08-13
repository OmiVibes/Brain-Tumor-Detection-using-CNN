from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from brainscan.robustness.quality import compute_quality_metrics, detect_quality_flags


def test_black_image_detected() -> None:
    image = Image.new("RGB", (32, 32), color=(0, 0, 0))
    metrics = compute_quality_metrics(image)
    flags = detect_quality_flags(
        metrics,
        {
            "brightness_mean_low": 5.0,
            "brightness_mean_high": 250.0,
            "contrast_std_low": 2.0,
            "blur_laplacian_variance_low": 1.0,
            "near_black_ratio_high": 0.9,
            "near_white_ratio_high": 0.9,
            "dynamic_range_low": 5.0,
            "low_information_dynamic_range": 5.0,
            "low_information_contrast_std": 2.0,
        },
    )
    assert "LOW_INFORMATION_IMAGE" in flags
    assert "NEAR_BLACK_DOMINANT" in flags


def test_white_image_detected() -> None:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    metrics = compute_quality_metrics(image)
    flags = detect_quality_flags(
        metrics,
        {
            "brightness_mean_low": 5.0,
            "brightness_mean_high": 250.0,
            "contrast_std_low": 2.0,
            "blur_laplacian_variance_low": 1.0,
            "near_black_ratio_high": 0.9,
            "near_white_ratio_high": 0.9,
            "dynamic_range_low": 5.0,
            "low_information_dynamic_range": 5.0,
            "low_information_contrast_std": 2.0,
        },
    )
    assert "LOW_INFORMATION_IMAGE" in flags
    assert "NEAR_WHITE_DOMINANT" in flags


def test_constant_gray_image_detected() -> None:
    image = Image.new("RGB", (32, 32), color=(127, 127, 127))
    metrics = compute_quality_metrics(image)
    assert metrics.contrast_std == 0.0
    assert metrics.dynamic_range == 0.0


def test_blur_metric_behavior() -> None:
    array = np.zeros((64, 64), dtype=np.uint8)
    array[16:48, 16:48] = 255
    sharp = Image.fromarray(array).convert("RGB")
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    sharp_metrics = compute_quality_metrics(sharp)
    blurred_metrics = compute_quality_metrics(blurred)
    assert sharp_metrics.blur_laplacian_variance > blurred_metrics.blur_laplacian_variance


def test_brightness_metric_behavior() -> None:
    dark = Image.new("RGB", (32, 32), color=(10, 10, 10))
    bright = Image.new("RGB", (32, 32), color=(240, 240, 240))
    dark_metrics = compute_quality_metrics(dark)
    bright_metrics = compute_quality_metrics(bright)
    assert dark_metrics.brightness_mean < bright_metrics.brightness_mean
