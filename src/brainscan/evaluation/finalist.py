"""Finalist checkpoint and architecture-decision helpers for Phase 2C."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
import csv
import json
import math

from brainscan.comparison import resolve_frozen_finalist
from brainscan.core.config import resolve_project_path


def load_prediction_rows(predictions_csv_path: str | Path) -> list[dict[str, object]]:
    resolved_path = resolve_project_path(predictions_csv_path)
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {resolved_path}")
    return rows


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    raise TypeError(f"Unsupported boolean-like value: {value!r}")


def compute_paired_disagreement(
    reference_rows: Sequence[dict[str, object]],
    challenger_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    reference_by_path = {str(row["relative_path"]): row for row in reference_rows}
    challenger_by_path = {str(row["relative_path"]): row for row in challenger_rows}

    if set(reference_by_path) != set(challenger_by_path):
        raise ValueError("Reference and challenger prediction sets must cover the same relative paths.")

    counts = Counter()
    disagreement_rows: list[dict[str, object]] = []
    for relative_path in sorted(reference_by_path):
        reference_row = reference_by_path[relative_path]
        challenger_row = challenger_by_path[relative_path]
        reference_correct = _coerce_bool(reference_row["correct"])
        challenger_correct = _coerce_bool(challenger_row["correct"])

        if reference_correct and challenger_correct:
            counts["both_correct"] += 1
        elif (not reference_correct) and (not challenger_correct):
            counts["both_wrong"] += 1
        elif reference_correct and (not challenger_correct):
            counts["reference_correct_challenger_wrong"] += 1
        else:
            counts["challenger_correct_reference_wrong"] += 1

        if reference_correct != challenger_correct:
            disagreement_rows.append(
                {
                    "relative_path": relative_path,
                    "reference_correct": reference_correct,
                    "challenger_correct": challenger_correct,
                    "reference_prediction": reference_row["predicted_class"],
                    "challenger_prediction": challenger_row["predicted_class"],
                    "true_class": reference_row["true_class"],
                }
            )

    return {
        "counts": {
            "both_correct": counts["both_correct"],
            "both_wrong": counts["both_wrong"],
            "reference_correct_challenger_wrong": counts["reference_correct_challenger_wrong"],
            "challenger_correct_reference_wrong": counts["challenger_correct_reference_wrong"],
        },
        "disagreement_rows": disagreement_rows,
    }


def compute_mcnemar_exact(reference_correct_challenger_wrong: int, challenger_correct_reference_wrong: int) -> dict[str, object]:
    total_discordant = reference_correct_challenger_wrong + challenger_correct_reference_wrong
    if total_discordant == 0:
        return {
            "discordant_pairs": 0,
            "two_sided_p_value": 1.0,
            "method": "exact binomial",
        }

    smaller_tail = min(reference_correct_challenger_wrong, challenger_correct_reference_wrong)
    tail_probability = sum(
        math.comb(total_discordant, k) * (0.5**total_discordant)
        for k in range(smaller_tail + 1)
    )
    return {
        "discordant_pairs": total_discordant,
        "two_sided_p_value": min(1.0, 2.0 * tail_probability),
        "method": "exact binomial",
    }


def count_confusion_pair(
    confusion_pairs: Sequence[dict[str, object]],
    *,
    true_class: str,
    predicted_class: str,
) -> int:
    for row in confusion_pairs:
        if str(row["true_class"]) == true_class and str(row["predicted_class"]) == predicted_class:
            return int(row["count"])
    return 0


def build_final_architecture_decision_payload(
    *,
    frozen_finalist: dict[str, object],
    dataset_fingerprint: str,
    git_commit: str | None,
    validation_evidence: dict[str, object],
    challenger_test_metrics: dict[str, object],
    baseline_test_metrics: dict[str, object],
    parameter_comparison: dict[str, object],
    latency_comparison: dict[str, object],
    throughput_comparison: dict[str, object],
    stability_information: dict[str, object],
    benchmark_metadata_audit: dict[str, object],
    final_backbone: str,
    rationale: str,
    downstream_migration_required: bool,
) -> dict[str, object]:
    return {
        "frozen_finalist": {
            "architecture": frozen_finalist["architecture"],
            "experiment_name": frozen_finalist["experiment_name"],
            "seed": frozen_finalist["seed"],
            "checkpoint_path": frozen_finalist["checkpoint_path"],
            "best_epoch": frozen_finalist["best_epoch"],
            "validation_macro_f1": frozen_finalist["validation_macro_f1"],
        },
        "dataset_fingerprint": dataset_fingerprint,
        "validation_evidence": validation_evidence,
        "challenger_test_metrics": challenger_test_metrics,
        "resnet18_reference_metrics": baseline_test_metrics,
        "parameter_comparison": parameter_comparison,
        "latency_comparison": latency_comparison,
        "throughput_comparison": throughput_comparison,
        "stability_information": stability_information,
        "benchmark_metadata_audit": benchmark_metadata_audit,
        "final_backbone": final_backbone,
        "rationale": rationale,
        "downstream_migration_required": downstream_migration_required,
        "decision_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
    }


def resolve_finalist_summary_payload() -> dict[str, object]:
    finalist = resolve_frozen_finalist()
    return finalist["summary_payload"]  # type: ignore[return-value]
