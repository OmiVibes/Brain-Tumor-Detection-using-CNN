"""Calibration and uncertainty exports for BrainScanAI."""

from .calibration import (
    TemperatureScaler,
    build_temperature_artifact_payload,
    collect_logits_from_dataloader,
    fit_temperature_from_dataloader,
    load_temperature_artifact,
    save_temperature_artifact,
)
from .metrics import (
    CalibrationBin,
    assign_uncertainty_levels,
    compute_probability_metrics,
    derive_uncertainty_thresholds,
    error_detection_auroc,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood_from_logits,
    predictive_entropy,
    softmax_probabilities_from_logits,
    summarize_uncertainty_by_correctness,
    top1_top2_margin,
)

__all__ = [
    "CalibrationBin",
    "TemperatureScaler",
    "assign_uncertainty_levels",
    "build_temperature_artifact_payload",
    "collect_logits_from_dataloader",
    "compute_probability_metrics",
    "derive_uncertainty_thresholds",
    "error_detection_auroc",
    "expected_calibration_error",
    "fit_temperature_from_dataloader",
    "load_temperature_artifact",
    "multiclass_brier_score",
    "negative_log_likelihood_from_logits",
    "predictive_entropy",
    "save_temperature_artifact",
    "softmax_probabilities_from_logits",
    "summarize_uncertainty_by_correctness",
    "top1_top2_margin",
]
