"""Robustness helpers for BrainScanAI safe inference."""

from .input_validation import ImageValidationResult, SUPPORTED_IMAGE_EXTENSIONS, validate_image_file
from .ood import (
    OODReference,
    ResNet18FeatureExtractor,
    build_diagonal_mahalanobis_reference,
    compute_min_diagonal_mahalanobis_scores,
    extract_features_from_dataloader,
    load_ood_reference,
    save_ood_reference,
)
from .quality import QualityMetrics, compute_quality_metrics, derive_quality_thresholds, detect_quality_flags

__all__ = [
    "ImageValidationResult",
    "OODReference",
    "QualityMetrics",
    "ResNet18FeatureExtractor",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "build_diagonal_mahalanobis_reference",
    "compute_min_diagonal_mahalanobis_scores",
    "compute_quality_metrics",
    "derive_quality_thresholds",
    "detect_quality_flags",
    "extract_features_from_dataloader",
    "load_ood_reference",
    "save_ood_reference",
    "validate_image_file",
]
