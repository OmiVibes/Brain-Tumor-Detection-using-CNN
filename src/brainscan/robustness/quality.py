"""Transparent image-quality heuristics for BrainScanAI robustness checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityMetrics:
    brightness_mean: float
    contrast_std: float
    blur_laplacian_variance: float
    near_black_ratio: float
    near_white_ratio: float
    dynamic_range: float


def _laplacian_variance(grayscale: np.ndarray) -> float:
    """Compute Laplacian variance without requiring OpenCV."""
    padded = np.pad(grayscale, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )
    return float(laplacian.var())


def compute_quality_metrics(image: Image.Image) -> QualityMetrics:
    """Compute simple grayscale quality heuristics for one RGB image."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    brightness_mean = float(grayscale.mean())
    contrast_std = float(grayscale.std())
    dynamic_range = float(grayscale.max() - grayscale.min())
    near_black_ratio = float(np.mean(grayscale <= 10.0))
    near_white_ratio = float(np.mean(grayscale >= 245.0))
    blur_laplacian_variance = _laplacian_variance(grayscale)

    return QualityMetrics(
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        blur_laplacian_variance=blur_laplacian_variance,
        near_black_ratio=near_black_ratio,
        near_white_ratio=near_white_ratio,
        dynamic_range=dynamic_range,
    )


def detect_quality_flags(
    metrics: QualityMetrics,
    thresholds: dict[str, float],
) -> list[str]:
    """Return explicit quality flags from validation-derived thresholds."""
    flags: list[str] = []

    if metrics.contrast_std <= thresholds["contrast_std_low"]:
        flags.append("LOW_CONTRAST_IMAGE")
    if metrics.dynamic_range <= thresholds["dynamic_range_low"]:
        flags.append("LOW_DYNAMIC_RANGE")
    if metrics.blur_laplacian_variance <= thresholds["blur_laplacian_variance_low"]:
        flags.append("SEVERE_BLUR")
    if metrics.brightness_mean <= thresholds["brightness_mean_low"]:
        flags.append("VERY_DARK_IMAGE")
    if metrics.brightness_mean >= thresholds["brightness_mean_high"]:
        flags.append("VERY_BRIGHT_IMAGE")
    if metrics.near_black_ratio >= thresholds["near_black_ratio_high"]:
        flags.append("NEAR_BLACK_DOMINANT")
    if metrics.near_white_ratio >= thresholds["near_white_ratio_high"]:
        flags.append("NEAR_WHITE_DOMINANT")
    if metrics.dynamic_range <= thresholds["low_information_dynamic_range"]:
        flags.append("LOW_INFORMATION_IMAGE")
    if metrics.contrast_std <= thresholds["low_information_contrast_std"]:
        flags.append("LOW_INFORMATION_IMAGE")

    deduplicated: list[str] = []
    for flag in flags:
        if flag not in deduplicated:
            deduplicated.append(flag)
    return deduplicated


def derive_quality_thresholds(metrics_rows: list[QualityMetrics]) -> dict[str, float | str]:
    """Derive transparent heuristic thresholds from in-distribution metrics."""
    if not metrics_rows:
        raise ValueError("Cannot derive quality thresholds from an empty metrics set.")

    brightness = np.asarray([row.brightness_mean for row in metrics_rows], dtype=np.float64)
    contrast = np.asarray([row.contrast_std for row in metrics_rows], dtype=np.float64)
    blur = np.asarray([row.blur_laplacian_variance for row in metrics_rows], dtype=np.float64)
    near_black = np.asarray([row.near_black_ratio for row in metrics_rows], dtype=np.float64)
    near_white = np.asarray([row.near_white_ratio for row in metrics_rows], dtype=np.float64)
    dynamic_range = np.asarray([row.dynamic_range for row in metrics_rows], dtype=np.float64)

    return {
        "brightness_mean_low": float(np.quantile(brightness, 0.01)),
        "brightness_mean_high": float(np.quantile(brightness, 0.99)),
        "contrast_std_low": float(np.quantile(contrast, 0.01)),
        "blur_laplacian_variance_low": float(np.quantile(blur, 0.01)),
        "near_black_ratio_high": float(np.quantile(near_black, 0.99)),
        "near_white_ratio_high": float(np.quantile(near_white, 0.99)),
        "dynamic_range_low": float(np.quantile(dynamic_range, 0.01)),
        "low_information_dynamic_range": float(min(np.quantile(dynamic_range, 0.005), 5.0)),
        "low_information_contrast_std": float(min(np.quantile(contrast, 0.005), 2.0)),
        "derivation": (
            "brightness low/high = validation 1st/99th percentiles; "
            "contrast, blur, dynamic range low = validation 1st percentile; "
            "near-black and near-white high = validation 99th percentile; "
            "low-information thresholds = conservative min(validation 0.5th percentile, fixed ceiling)"
        ),
    }
