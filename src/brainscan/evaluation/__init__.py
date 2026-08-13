"""Evaluation utilities for BrainScanAI."""

from .classification import (
    annotate_predictions_with_near_duplicates,
    benchmark_model_inference,
    build_confusion_pairs,
    build_predictions_rows,
    compute_classification_metrics_bundle,
    compute_confidence_statistics,
    create_error_rows,
    evaluate_classifier,
    export_predictions_csv,
    load_near_duplicate_annotations,
    plot_confusion_matrix_figure,
    plot_pr_curves,
    plot_roc_curves,
    summarize_filtered_subset,
    verify_checkpoint_metadata,
)

__all__ = [
    "annotate_predictions_with_near_duplicates",
    "benchmark_model_inference",
    "build_confusion_pairs",
    "build_predictions_rows",
    "compute_classification_metrics_bundle",
    "compute_confidence_statistics",
    "create_error_rows",
    "evaluate_classifier",
    "export_predictions_csv",
    "load_near_duplicate_annotations",
    "plot_confusion_matrix_figure",
    "plot_pr_curves",
    "plot_roc_curves",
    "summarize_filtered_subset",
    "verify_checkpoint_metadata",
]
