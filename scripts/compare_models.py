"""Aggregate validation-only architecture comparison results for BrainScanAI."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.comparison import (
    aggregate_architecture_runs,
    build_selection_decision_payload,
    choose_architecture_from_validation,
    load_run_summary,
)
from brainscan.core.config import resolve_project_path
from brainscan.training.train_classifier import get_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate BrainScanAI comparison experiments.")
    parser.add_argument(
        "--comparison-dir",
        default="artifacts/training/comparison",
        help="Directory containing per-experiment comparison summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/model_comparison",
        help="Directory for aggregated comparison artifacts.",
    )
    return parser.parse_args()


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty comparison CSV.")

    fieldnames = list(rows[0].keys())
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def _write_json(payload: object, output_path: Path) -> Path:
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved


def _discover_summary_paths(comparison_dir: Path) -> list[Path]:
    resolved_dir = resolve_project_path(comparison_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Comparison directory not found: {resolved_dir}")

    summary_paths = sorted(resolved_dir.glob("*/summary.json"))
    filtered_paths: list[Path] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if bool(payload.get("smoke_test")):
            continue
        filtered_paths.append(summary_path)

    if not filtered_paths:
        raise ValueError("No non-smoke comparison summaries were found.")
    return filtered_paths


def main() -> None:
    args = parse_args()
    summary_paths = _discover_summary_paths(Path(args.comparison_dir))
    run_summaries = [load_run_summary(path) for path in summary_paths]
    aggregated_rows = aggregate_architecture_runs(run_summaries)
    selection = choose_architecture_from_validation(aggregated_rows)

    output_dir = Path(args.output_dir)
    raw_rows = [row.to_row() for row in run_summaries]
    raw_rows = sorted(raw_rows, key=lambda row: (str(row["architecture"]), int(row["seed"])))
    ranked_rows = selection["ranked_rows"]
    selected_row = selection["selected_row"]

    raw_json_payload = {
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "runs": raw_rows,
    }
    comparison_json_payload = {
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "comparison_rule": {
            "ranking_split": "validation",
            "selection_policy": selection["rationale"],
            "validation_f1_tolerance": selection["validation_f1_tolerance"],
            "validation_accuracy_tolerance": selection["validation_accuracy_tolerance"],
        },
        "architectures": ranked_rows,
    }
    selection_payload = build_selection_decision_payload(
        selected_architecture=str(selected_row["architecture"]),
        selected_experiment_names=list(selected_row["experiment_names"]),
        dataset_fingerprint=str(selected_row["dataset_fingerprint"]),
        git_commit=get_git_commit(),
        validation_basis={
            "ranking_split": "validation",
            "aggregated_rows_path": (output_dir / "model_comparison.json").as_posix(),
            "selected_best_validation_macro_f1": selected_row["best_validation_macro_f1"],
            "selected_mean_validation_macro_f1": selected_row["mean_validation_macro_f1"],
            "selected_mean_validation_accuracy": selected_row["mean_validation_accuracy"],
            "selected_run_count": selected_row["run_count"],
            "selected_seeds": selected_row["seeds"],
        },
        engineering_tradeoffs={
            "rationale": selection["rationale"],
            "mean_per_image_latency_ms": selected_row["mean_per_image_latency_ms"],
            "throughput_images_per_second": selected_row["throughput_images_per_second"],
            "parameter_count": selected_row["parameter_count"],
            "checkpoint_size_mb_mean": selected_row["checkpoint_size_mb_mean"],
            "peak_vram_mb_max": selected_row["peak_vram_mb_max"],
        },
    )
    selection_payload["candidate_rank_order"] = [row["architecture"] for row in ranked_rows]
    selection_payload["ranked_architectures"] = ranked_rows

    raw_csv_path = _write_csv(raw_rows, output_dir / "model_runs.csv")
    raw_json_path = _write_json(raw_json_payload, output_dir / "model_runs.json")
    comparison_csv_path = _write_csv(ranked_rows, output_dir / "model_comparison.csv")
    comparison_json_path = _write_json(comparison_json_payload, output_dir / "model_comparison.json")
    selection_json_path = _write_json(selection_payload, output_dir / "selection_decision.json")

    print("model_comparison_summary=")
    print(f"  run_count={len(run_summaries)}")
    print(f"  architecture_count={len(aggregated_rows)}")
    print(f"  selected_architecture={selected_row['architecture']}")
    print(f"  raw_runs_csv={resolve_project_path(raw_csv_path)}")
    print(f"  raw_runs_json={resolve_project_path(raw_json_path)}")
    print(f"  comparison_csv={resolve_project_path(comparison_csv_path)}")
    print(f"  comparison_json={resolve_project_path(comparison_json_path)}")
    print(f"  selection_json={resolve_project_path(selection_json_path)}")


if __name__ == "__main__":
    main()
