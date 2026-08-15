from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from brainscan.robustness.evaluation import (
    build_abstention_policy_artifact,
    build_risk_coverage_curve,
    compute_classwise_status_counts,
    compute_correctness_breakdown,
    compute_entropy_ood_relationship,
    compute_error_capture,
    compute_quality_flag_frequencies,
    compute_selective_metrics,
    compute_status_summary,
    save_json,
    summarize_numeric_distribution,
    summarize_perturbation_rows,
    summarize_synthetic_ood,
)


def _toy_rows() -> list[dict[str, object]]:
    return [
        {
            "true_class": "glioma",
            "safe_status": "ACCEPT",
            "classifier_correct": True,
            "normalized_entropy": 0.1,
            "ood_score": 10.0,
            "quality_flags": "",
        },
        {
            "true_class": "glioma",
            "safe_status": "REVIEW",
            "classifier_correct": True,
            "normalized_entropy": 0.2,
            "ood_score": 20.0,
            "quality_flags": "SEVERE_BLUR",
        },
        {
            "true_class": "meningioma",
            "safe_status": "ABSTAIN",
            "classifier_correct": True,
            "normalized_entropy": 0.3,
            "ood_score": 30.0,
            "quality_flags": "LOW_INFORMATION_IMAGE",
        },
        {
            "true_class": "meningioma",
            "safe_status": "ACCEPT",
            "classifier_correct": False,
            "normalized_entropy": 0.4,
            "ood_score": 40.0,
            "quality_flags": "",
        },
        {
            "true_class": "pituitary",
            "safe_status": "REVIEW",
            "classifier_correct": False,
            "normalized_entropy": 0.5,
            "ood_score": 50.0,
            "quality_flags": "VERY_BRIGHT_IMAGE",
        },
        {
            "true_class": "no_tumor",
            "safe_status": "ABSTAIN",
            "classifier_correct": False,
            "normalized_entropy": 0.6,
            "ood_score": 60.0,
            "quality_flags": "LOW_INFORMATION_IMAGE",
        },
    ]


def test_status_counting() -> None:
    summary = compute_status_summary(_toy_rows())
    assert summary["counts"] == {"ACCEPT": 2, "REVIEW": 2, "ABSTAIN": 2}
    assert summary["total"] == 6


def test_coverage_and_selective_risk() -> None:
    selective = compute_selective_metrics(_toy_rows(), accepted_statuses={"ACCEPT"})
    assert selective["selected_count"] == 2
    assert selective["coverage"] == 2 / 6
    assert selective["accuracy"] == 0.5
    assert selective["risk"] == 0.5


def test_error_capture_calculation() -> None:
    capture = compute_error_capture(_toy_rows())
    assert capture["wrong_total"] == 3
    assert capture["unsafe_accepted_error_count"] == 1
    assert capture["error_capture_rate"] == 2 / 3
    assert capture["hard_error_capture_rate"] == 1 / 3


def test_false_abstention_and_non_acceptance() -> None:
    breakdown = compute_correctness_breakdown(_toy_rows())
    assert breakdown["false_abstention_count"] == 1
    assert breakdown["false_abstention_rate"] == 1 / 3
    assert breakdown["correct_non_acceptance_count"] == 2
    assert breakdown["correct_non_acceptance_rate"] == 2 / 3


def test_classwise_status_aggregation() -> None:
    classwise = compute_classwise_status_counts(_toy_rows())
    assert classwise["glioma"] == {"ACCEPT": 1, "REVIEW": 1, "ABSTAIN": 0}
    assert classwise["meningioma"] == {"ACCEPT": 1, "REVIEW": 0, "ABSTAIN": 1}


def test_quality_flag_frequencies() -> None:
    frequencies = compute_quality_flag_frequencies(_toy_rows())
    assert frequencies["LOW_INFORMATION_IMAGE"] == 2
    assert frequencies["SEVERE_BLUR"] == 1


def test_numeric_summary() -> None:
    summary = summarize_numeric_distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5


def test_entropy_ood_relationship() -> None:
    relationship = compute_entropy_ood_relationship(_toy_rows())
    assert relationship["count"] == 6
    assert relationship["pearson_correlation"] is not None


def test_risk_coverage_curve() -> None:
    curve = build_risk_coverage_curve(_toy_rows(), score_key="normalized_entropy", thresholds=[0.2, 0.6])
    assert curve[0]["selected_count"] == 2
    assert curve[-1]["selected_count"] == 6


def test_perturbation_summary_logic() -> None:
    rows = [
        {
            "perturbation_name": "mild_noise",
            "prediction_consistent": True,
            "raw_confidence_change": -0.1,
            "entropy_change": 0.2,
            "ood_score_change": 1.0,
            "safe_status": "ACCEPT",
        },
        {
            "perturbation_name": "mild_noise",
            "prediction_consistent": False,
            "raw_confidence_change": -0.2,
            "entropy_change": 0.3,
            "ood_score_change": 2.0,
            "safe_status": "REVIEW",
        },
    ]
    summary = summarize_perturbation_rows(rows)
    assert summary[0]["sample_count"] == 2
    assert summary[0]["prediction_consistency"] == 0.5


def test_synthetic_ood_summary() -> None:
    rows = [
        {"safe_status": "ABSTAIN"},
        {"safe_status": "REVIEW"},
        {"safe_status": "ABSTAIN"},
    ]
    summary = summarize_synthetic_ood(rows)
    assert summary["counts"] == {"ACCEPT": 0, "REVIEW": 1, "ABSTAIN": 2}
    assert summary["ood_non_acceptance_rate"] == 1.0


def test_frozen_policy_artifact_serialization() -> None:
    artifact_dir = Path("artifacts/test_tmp") / f"policy-{uuid.uuid4().hex}"
    try:
        payload, path = build_abstention_policy_artifact(
            checkpoint_path="artifacts/models/resnet18_baseline_best.pt",
            checkpoint_epoch=9,
            dataset_fingerprint="abc123",
            ood_metadata={
                "architecture": "resnet18",
                "feature_layer": "avgpool",
                "feature_dimension": 512,
                "ood_method": "class_conditional_diagonal_mahalanobis",
                "warning_threshold": 1.0,
                "warning_threshold_quantile": 0.9,
                "warning_threshold_derivation": "validation 90th percentile",
                "threshold": 2.0,
                "threshold_quantile": 0.95,
                "threshold_derivation": "validation 95th percentile",
            },
            quality_payload={"thresholds": {"low_information_dynamic_range": 5.0}},
            uncertainty_metrics={
                "uncertainty_thresholds": {"entropy_q75": 0.2},
                "default_probability_mode": "raw",
            },
            git_commit="deadbeef",
            output_path=artifact_dir / "abstention_policy.json",
        )
        assert payload["checkpoint"] == "artifacts/models/resnet18_baseline_best.pt"
        assert payload["architecture"] == "resnet18"
        assert payload["feature_layer"] == "avgpool"
        assert payload["feature_dimension"] == 512
        assert payload["probability_mode"] == "raw"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["dataset_fingerprint"] == "abc123"
        assert saved["threshold_derivation_source"]["uncertainty"] == "validation only"
        assert not Path(str(saved["checkpoint"])).is_absolute()
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
