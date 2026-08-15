"""Robustness helpers for BrainScanAI safe inference."""

from .abstention import AbstentionDecision, decide_abstention
from .evaluation import (
    build_abstention_policy_artifact,
    build_risk_coverage_curve,
    compute_classwise_status_counts,
    compute_correctness_breakdown,
    compute_entropy_ood_relationship,
    compute_error_capture,
    compute_quality_flag_frequencies,
    compute_selective_metrics,
    compute_status_summary,
    plot_risk_coverage_curve,
    save_csv,
    save_json,
    summarize_numeric_distribution,
    summarize_perturbation_rows,
    summarize_synthetic_ood,
)
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
from brainscan.models import ArchitectureFeatureExtractor

__all__ = [
    "AbstentionDecision",
    "ArchitectureFeatureExtractor",
    "ImageValidationResult",
    "OODReference",
    "QualityMetrics",
    "ResNet18FeatureExtractor",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "build_abstention_policy_artifact",
    "build_diagonal_mahalanobis_reference",
    "build_risk_coverage_curve",
    "compute_classwise_status_counts",
    "compute_correctness_breakdown",
    "compute_entropy_ood_relationship",
    "compute_error_capture",
    "compute_quality_flag_frequencies",
    "compute_min_diagonal_mahalanobis_scores",
    "compute_quality_metrics",
    "compute_selective_metrics",
    "compute_status_summary",
    "decide_abstention",
    "derive_quality_thresholds",
    "detect_quality_flags",
    "extract_features_from_dataloader",
    "load_ood_reference",
    "plot_risk_coverage_curve",
    "save_csv",
    "save_json",
    "save_ood_reference",
    "summarize_numeric_distribution",
    "summarize_perturbation_rows",
    "summarize_synthetic_ood",
    "validate_image_file",
]
