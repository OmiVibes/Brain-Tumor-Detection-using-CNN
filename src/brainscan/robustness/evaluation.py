"""Reusable robustness and selective-classification analysis helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from brainscan.core.config import make_project_relative_path, resolve_project_path


def save_json(payload: object, output_path: str | Path) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_path


def save_csv(rows: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot save an empty CSV artifact.")

    fieldnames = list(rows[0].keys())
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved_path


def build_abstention_policy_artifact(
    *,
    checkpoint_path: str | Path,
    checkpoint_epoch: int,
    dataset_fingerprint: str,
    ood_metadata: dict[str, object],
    quality_payload: dict[str, object],
    uncertainty_metrics: dict[str, object],
    git_commit: str | None,
    output_path: str | Path,
    policy_version: str = "phase2b_resnet18_v1",
) -> tuple[dict[str, object], Path]:
    """Build and persist a frozen-policy metadata artifact before test evaluation."""
    payload = {
        "policy_version": policy_version,
        "checkpoint": make_project_relative_path(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "dataset_fingerprint": dataset_fingerprint,
        "ood_method": ood_metadata["ood_method"],
        "ood_thresholds": {
            "warning_threshold": ood_metadata["warning_threshold"],
            "warning_threshold_quantile": ood_metadata["warning_threshold_quantile"],
            "warning_threshold_derivation": ood_metadata["warning_threshold_derivation"],
            "fail_threshold": ood_metadata["threshold"],
            "fail_threshold_quantile": ood_metadata["threshold_quantile"],
            "fail_threshold_derivation": ood_metadata["threshold_derivation"],
        },
        "quality_thresholds": quality_payload["thresholds"],
        "uncertainty_thresholds": uncertainty_metrics["uncertainty_thresholds"],
        "decision_rules": {
            "abstain": [
                "invalid_or_corrupt_input",
                "LOW_INFORMATION_IMAGE",
                "OOD status == FAIL",
            ],
            "review": [
                "OOD status == BORDERLINE",
                "quality warning flags present",
                "uncertainty level in {MEDIUM, HIGH}",
            ],
            "accept": [
                "validation passes",
                "no low-information flag",
                "OOD status == PASS",
                "uncertainty level == LOW",
                "no quality warnings",
            ],
        },
        "threshold_derivation_source": {
            "ood": "validation only",
            "quality": "train + validation only",
            "uncertainty": "validation only",
        },
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
    }
    return payload, save_json(payload, output_path)


def compute_status_summary(rows: Sequence[dict[str, object]], *, status_key: str = "safe_status") -> dict[str, object]:
    counts = Counter(str(row[status_key]) for row in rows)
    total = len(rows)
    return {
        "total": total,
        "counts": {
            "ACCEPT": counts.get("ACCEPT", 0),
            "REVIEW": counts.get("REVIEW", 0),
            "ABSTAIN": counts.get("ABSTAIN", 0),
        },
        "rates": {
            "ACCEPT": counts.get("ACCEPT", 0) / total if total else 0.0,
            "REVIEW": counts.get("REVIEW", 0) / total if total else 0.0,
            "ABSTAIN": counts.get("ABSTAIN", 0) / total if total else 0.0,
        },
    }


def compute_correctness_breakdown(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    correct_rows = [row for row in rows if bool(row["classifier_correct"])]
    summary = compute_status_summary(correct_rows)
    total = len(correct_rows)
    correct_non_acceptance_count = summary["counts"]["REVIEW"] + summary["counts"]["ABSTAIN"]
    return {
        "correct_total": total,
        "counts": summary["counts"],
        "rates": summary["rates"],
        "false_abstention_count": summary["counts"]["ABSTAIN"],
        "false_abstention_rate": summary["counts"]["ABSTAIN"] / total if total else 0.0,
        "correct_non_acceptance_count": correct_non_acceptance_count,
        "correct_non_acceptance_rate": correct_non_acceptance_count / total if total else 0.0,
    }


def compute_selective_metrics(
    rows: Sequence[dict[str, object]],
    *,
    accepted_statuses: set[str],
) -> dict[str, float | int]:
    selected = [row for row in rows if str(row["safe_status"]) in accepted_statuses]
    total = len(rows)
    selected_total = len(selected)
    correct_count = sum(bool(row["classifier_correct"]) for row in selected)
    accuracy = correct_count / selected_total if selected_total else 0.0
    risk = 1.0 - accuracy if selected_total else 0.0
    return {
        "selected_count": selected_total,
        "coverage": selected_total / total if total else 0.0,
        "accuracy": accuracy,
        "risk": risk,
    }


def compute_error_capture(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    wrong_rows = [row for row in rows if not bool(row["classifier_correct"])]
    summary = compute_status_summary(wrong_rows)
    total = len(wrong_rows)
    captured = summary["counts"]["REVIEW"] + summary["counts"]["ABSTAIN"]
    unsafe_accepted = summary["counts"]["ACCEPT"]
    return {
        "wrong_total": total,
        "counts": summary["counts"],
        "unsafe_accepted_error_count": unsafe_accepted,
        "unsafe_accepted_error_rate_all_test": unsafe_accepted / len(rows) if rows else 0.0,
        "error_capture_rate": captured / total if total else 0.0,
        "hard_error_capture_rate": summary["counts"]["ABSTAIN"] / total if total else 0.0,
        "unsafe_accepted_errors": [
            row for row in wrong_rows if str(row["safe_status"]) == "ACCEPT"
        ],
        "error_rows": wrong_rows,
    }


def compute_classwise_status_counts(
    rows: Sequence[dict[str, object]],
    *,
    class_key: str = "true_class",
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row[class_key])][str(row["safe_status"])] += 1
    return {
        class_name: {
            "ACCEPT": counts.get("ACCEPT", 0),
            "REVIEW": counts.get("REVIEW", 0),
            "ABSTAIN": counts.get("ABSTAIN", 0),
        }
        for class_name, counts in grouped.items()
    }


def compute_quality_flag_frequencies(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        flags = str(row["quality_flags"]).split("|") if row["quality_flags"] else []
        for flag in flags:
            if flag:
                counts[flag] += 1
    return dict(sorted(counts.items()))


def summarize_numeric_distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "q90": 0.0,
        }
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }


def compute_entropy_ood_relationship(rows: Sequence[dict[str, object]]) -> dict[str, float | int | None]:
    entropy = np.asarray([float(row["normalized_entropy"]) for row in rows], dtype=np.float64)
    ood = np.asarray([float(row["ood_score"]) for row in rows], dtype=np.float64)
    if entropy.size < 2 or ood.size < 2:
        return {"count": int(entropy.size), "pearson_correlation": None}
    correlation = float(np.corrcoef(entropy, ood)[0, 1])
    return {"count": int(entropy.size), "pearson_correlation": correlation}


def build_risk_coverage_curve(
    rows: Sequence[dict[str, object]],
    *,
    score_key: str,
    thresholds: Sequence[float],
) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    total = len(rows)
    for threshold in thresholds:
        selected = [row for row in rows if float(row[score_key]) <= threshold]
        selected_total = len(selected)
        correct_count = sum(bool(row["classifier_correct"]) for row in selected)
        accuracy = correct_count / selected_total if selected_total else 0.0
        points.append(
            {
                "threshold": float(threshold),
                "selected_count": selected_total,
                "coverage": selected_total / total if total else 0.0,
                "accuracy": accuracy,
                "selective_risk": 1.0 - accuracy if selected_total else 0.0,
            }
        )
    return points


def plot_risk_coverage_curve(
    validation_points: Sequence[dict[str, float | int]],
    test_points: Sequence[dict[str, float | int]],
    output_path: str | Path,
) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(
        [float(point["coverage"]) for point in validation_points],
        [float(point["selective_risk"]) for point in validation_points],
        label="validation",
    )
    plt.plot(
        [float(point["coverage"]) for point in test_points],
        [float(point["selective_risk"]) for point in test_points],
        label="test",
    )
    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk")
    plt.title("BrainScanAI Risk-Coverage Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved_path)
    plt.close()
    return resolved_path


def summarize_synthetic_ood(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    summary = compute_status_summary(rows)
    total = summary["total"]
    return {
        "total": total,
        "counts": summary["counts"],
        "ood_abstention_rate": summary["counts"]["ABSTAIN"] / total if total else 0.0,
        "ood_non_acceptance_rate": (
            (summary["counts"]["REVIEW"] + summary["counts"]["ABSTAIN"]) / total if total else 0.0
        ),
    }


def summarize_perturbation_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["perturbation_name"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for perturbation_name, group_rows in grouped.items():
        total = len(group_rows)
        consistency = sum(bool(row["prediction_consistent"]) for row in group_rows) / total if total else 0.0
        summary = compute_status_summary(group_rows)
        summary_rows.append(
            {
                "perturbation_name": perturbation_name,
                "sample_count": total,
                "prediction_consistency": consistency,
                "mean_confidence_change": float(
                    np.mean([float(row["raw_confidence_change"]) for row in group_rows]) if total else 0.0
                ),
                "mean_entropy_change": float(
                    np.mean([float(row["entropy_change"]) for row in group_rows]) if total else 0.0
                ),
                "mean_ood_score_change": float(
                    np.mean([float(row["ood_score_change"]) for row in group_rows]) if total else 0.0
                ),
                "accept_rate": summary["rates"]["ACCEPT"],
                "review_rate": summary["rates"]["REVIEW"],
                "abstain_rate": summary["rates"]["ABSTAIN"],
            }
        )
    return sorted(summary_rows, key=lambda row: str(row["perturbation_name"]))
