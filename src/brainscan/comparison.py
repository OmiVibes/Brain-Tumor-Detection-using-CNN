"""Experiment naming, aggregation, and selection helpers for controlled model comparison."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

from brainscan.core.config import make_project_relative_path, resolve_project_path


EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9_\\-]+$")
VALIDATION_F1_TOLERANCE = 0.002
VALIDATION_ACCURACY_TOLERANCE = 0.002


@dataclass(frozen=True)
class RunSummary:
    experiment_name: str
    architecture: str
    seed: int
    batch_size: int
    learning_rate: float
    mixed_precision: bool
    dataset_fingerprint: str
    parameter_count: int
    trainable_parameter_count: int
    estimated_model_size_mb: float
    checkpoint_size_mb: float
    best_epoch: int
    best_validation_accuracy: float
    best_validation_precision_macro: float
    best_validation_recall_macro: float
    best_validation_macro_f1: float
    best_validation_loss: float
    epochs_completed: int
    training_duration_seconds: float | None
    training_duration_complete: bool
    training_duration_note: str | None
    mean_per_image_latency_ms: float
    throughput_images_per_second: float
    checkpoint_load_time_ms: float | None
    peak_vram_mb: float | None
    peak_benchmark_vram_mb: float | None
    weights_name: str | None
    input_size: list[int]
    class_mapping: dict[str, int]
    source_summary_path: str

    def to_row(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "architecture": self.architecture,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "mixed_precision": self.mixed_precision,
            "dataset_fingerprint": self.dataset_fingerprint,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "estimated_model_size_mb": self.estimated_model_size_mb,
            "checkpoint_size_mb": self.checkpoint_size_mb,
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_validation_accuracy,
            "best_validation_precision_macro": self.best_validation_precision_macro,
            "best_validation_recall_macro": self.best_validation_recall_macro,
            "best_validation_macro_f1": self.best_validation_macro_f1,
            "best_validation_loss": self.best_validation_loss,
            "epochs_completed": self.epochs_completed,
            "training_duration_seconds": self.training_duration_seconds,
            "training_duration_complete": self.training_duration_complete,
            "training_duration_note": self.training_duration_note,
            "mean_per_image_latency_ms": self.mean_per_image_latency_ms,
            "throughput_images_per_second": self.throughput_images_per_second,
            "checkpoint_load_time_ms": self.checkpoint_load_time_ms,
            "peak_vram_mb": self.peak_vram_mb,
            "peak_benchmark_vram_mb": self.peak_benchmark_vram_mb,
            "weights_name": self.weights_name,
            "input_size": list(self.input_size),
            "class_mapping": dict(self.class_mapping),
            "source_summary_path": self.source_summary_path,
        }


def sanitize_experiment_name(experiment_name: str) -> str:
    normalized = experiment_name.strip().lower()
    if not normalized:
        raise ValueError("Experiment name must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("Experiment name must not contain path separators.")
    if not EXPERIMENT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Experiment name may contain only lowercase letters, digits, underscores, and hyphens."
        )
    return normalized


def build_experiment_name(architecture: str, seed: int) -> str:
    return sanitize_experiment_name(f"{architecture.lower()}_seed{int(seed)}")


def build_comparison_paths(experiment_name: str) -> dict[str, Path]:
    safe_name = sanitize_experiment_name(experiment_name)
    model_dir = Path("artifacts/models/comparison") / safe_name
    training_dir = Path("artifacts/training/comparison") / safe_name
    return {
        "experiment_name": Path(safe_name),
        "model_dir": model_dir,
        "training_dir": training_dir,
        "best_checkpoint": model_dir / "best.pt",
        "last_checkpoint": model_dir / "last.pt",
        "history_json": training_dir / "history.json",
        "history_csv": training_dir / "history.csv",
        "summary_json": training_dir / "summary.json",
        "benchmark_json": training_dir / "benchmark.json",
        "smoke_json": training_dir / "smoke.json",
    }


def create_comparison_config(
    base_config: dict[str, object],
    *,
    architecture: str,
    seed: int,
    experiment_name: str,
    batch_size: int,
    learning_rate: float | None = None,
) -> dict[str, object]:
    config = deepcopy(base_config)
    paths = build_comparison_paths(experiment_name)
    config["training"]["seed"] = int(seed)
    config["training"]["batch_size"] = int(batch_size)
    if learning_rate is not None:
        config["training"]["learning_rate"] = float(learning_rate)
    config["model"]["architecture"] = architecture.lower()
    config["checkpoint"]["model_dir"] = paths["model_dir"].as_posix()
    config["checkpoint"]["training_dir"] = paths["training_dir"].as_posix()
    config["checkpoint"]["best_filename"] = paths["best_checkpoint"].name
    config["checkpoint"]["last_filename"] = paths["last_checkpoint"].name
    config["checkpoint"]["history_json"] = paths["history_json"].name
    config["checkpoint"]["history_csv"] = paths["history_csv"].name
    return config


def save_json_artifact(payload: dict[str, object], output_path: str | Path) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_path


def load_json_artifact(input_path: str | Path) -> dict[str, object]:
    resolved_path = resolve_project_path(input_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {resolved_path}, found {type(payload).__name__}.")
    return payload


def summarize_history_best_metrics(history: Sequence[dict[str, object]]) -> dict[str, float | int]:
    if not history:
        raise ValueError("Cannot summarize empty training history.")

    best_entry = max(history, key=lambda row: float(row["val_macro_f1"]))
    return {
        "best_epoch": int(best_entry["epoch"]),
        "best_validation_accuracy": float(best_entry["val_accuracy"]),
        "best_validation_precision_macro": float(best_entry["val_precision_macro"]),
        "best_validation_recall_macro": float(best_entry["val_recall_macro"]),
        "best_validation_macro_f1": float(best_entry["val_macro_f1"]),
        "best_validation_loss": float(best_entry["val_loss"]),
    }


def audit_training_duration_payload(summary_payload: dict[str, object]) -> dict[str, object]:
    cumulative_duration = summary_payload.get("cumulative_training_duration_seconds")
    resumed_from_epoch = summary_payload.get("resumed_from_epoch")
    reported_duration = summary_payload.get("training_duration_seconds")

    if cumulative_duration is not None:
        return {
            "status": "complete",
            "reported_training_duration_seconds": float(cumulative_duration),
            "corrected_training_duration_seconds": float(cumulative_duration),
            "note": "Cumulative training duration was persisted across resumed segments.",
        }

    if resumed_from_epoch is not None:
        return {
            "status": "incomplete",
            "reported_training_duration_seconds": (
                float(reported_duration) if reported_duration is not None else None
            ),
            "corrected_training_duration_seconds": None,
            "note": (
                "Only the latest resumed training segment duration is available in summary metadata; "
                "the cumulative total cannot be reconstructed reliably from the existing artifacts."
            ),
        }

    if reported_duration is None:
        return {
            "status": "unknown",
            "reported_training_duration_seconds": None,
            "corrected_training_duration_seconds": None,
            "note": "No training duration was recorded in the summary payload.",
        }

    return {
        "status": "complete",
        "reported_training_duration_seconds": float(reported_duration),
        "corrected_training_duration_seconds": float(reported_duration),
        "note": "Single uninterrupted training segment recorded in the summary payload.",
    }


def run_summary_from_payload(summary_payload: dict[str, object]) -> RunSummary:
    history = summary_payload["history"]
    if not isinstance(history, list):
        raise TypeError("Comparison summary payload must contain a list-valued 'history'.")
    best_metrics = summarize_history_best_metrics(history)
    duration_audit = audit_training_duration_payload(summary_payload)
    model_metadata = summary_payload["model_metadata"]
    benchmark = summary_payload["benchmark"]
    device_info = summary_payload["device_info"]
    if not isinstance(model_metadata, dict) or not isinstance(benchmark, dict) or not isinstance(device_info, dict):
        raise TypeError("Comparison summary payload is missing structured metadata blocks.")

    return RunSummary(
        experiment_name=str(summary_payload["experiment_name"]),
        architecture=str(summary_payload["architecture"]),
        seed=int(summary_payload["seed"]),
        batch_size=int(summary_payload["batch_size"]),
        learning_rate=float(summary_payload["learning_rate"]),
        mixed_precision=bool(summary_payload.get("mixed_precision", True)),
        dataset_fingerprint=str(summary_payload["dataset_fingerprint"]),
        parameter_count=int(model_metadata["parameter_count"]),
        trainable_parameter_count=int(model_metadata["trainable_parameter_count"]),
        estimated_model_size_mb=float(model_metadata["estimated_model_size_mb"]),
        checkpoint_size_mb=float(summary_payload["best_checkpoint_size_mb"]),
        best_epoch=int(best_metrics["best_epoch"]),
        best_validation_accuracy=float(best_metrics["best_validation_accuracy"]),
        best_validation_precision_macro=float(best_metrics["best_validation_precision_macro"]),
        best_validation_recall_macro=float(best_metrics["best_validation_recall_macro"]),
        best_validation_macro_f1=float(best_metrics["best_validation_macro_f1"]),
        best_validation_loss=float(best_metrics["best_validation_loss"]),
        epochs_completed=int(summary_payload["epochs_completed"]),
        training_duration_seconds=(
            float(duration_audit["corrected_training_duration_seconds"])
            if duration_audit["corrected_training_duration_seconds"] is not None
            else None
        ),
        training_duration_complete=duration_audit["status"] == "complete",
        training_duration_note=(
            str(duration_audit["note"]) if duration_audit.get("note") is not None else None
        ),
        mean_per_image_latency_ms=float(benchmark["mean_per_image_latency_ms"]),
        throughput_images_per_second=float(benchmark["throughput_images_per_second"]),
        checkpoint_load_time_ms=(
            float(benchmark["checkpoint_load_time_ms"])
            if benchmark.get("checkpoint_load_time_ms") is not None
            else None
        ),
        peak_vram_mb=(float(summary_payload["peak_vram_mb"]) if summary_payload.get("peak_vram_mb") is not None else None),
        peak_benchmark_vram_mb=(
            float(summary_payload["peak_benchmark_vram_mb"])
            if summary_payload.get("peak_benchmark_vram_mb") is not None
            else None
        ),
        weights_name=(str(model_metadata["pretrained_weights"]) if model_metadata.get("pretrained_weights") else None),
        input_size=[int(value) for value in model_metadata["input_size"]],
        class_mapping={str(key): int(value) for key, value in dict(model_metadata["class_mapping"]).items()},
        source_summary_path=str(summary_payload["source_summary_path"]),
    )


def load_run_summary(summary_path: str | Path) -> RunSummary:
    resolved_path = resolve_project_path(summary_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    try:
        payload["source_summary_path"] = make_project_relative_path(resolved_path)
    except ValueError:
        payload["source_summary_path"] = resolved_path.as_posix()
    return run_summary_from_payload(payload)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def load_selection_decision(selection_path: str | Path = "artifacts/model_comparison/selection_decision.json") -> dict[str, object]:
    return load_json_artifact(selection_path)


def resolve_selected_architecture_row(selection_payload: dict[str, object]) -> dict[str, object]:
    selected_architecture = str(selection_payload["selected_architecture"])
    ranked_architectures = selection_payload.get("ranked_architectures")
    if not isinstance(ranked_architectures, list):
        raise TypeError("Selection payload must contain a 'ranked_architectures' list.")
    for row in ranked_architectures:
        if isinstance(row, dict) and str(row.get("architecture")) == selected_architecture:
            return row
    raise ValueError(f"Selected architecture '{selected_architecture}' was not found in ranked_architectures.")


def resolve_frozen_finalist(
    *,
    selection_path: str | Path = "artifacts/model_comparison/selection_decision.json",
    comparison_training_dir: str | Path = "artifacts/training/comparison",
) -> dict[str, object]:
    selection_payload = load_selection_decision(selection_path)
    selected_row = resolve_selected_architecture_row(selection_payload)

    experiment_name = selected_row.get("best_run_experiment_name")
    if experiment_name is None:
        selected_experiment_names = selection_payload.get("selected_experiment_names")
        if not isinstance(selected_experiment_names, list) or len(selected_experiment_names) != 1:
            raise ValueError("Unable to resolve a unique frozen finalist experiment from the selection artifact.")
        experiment_name = selected_experiment_names[0]

    summary_path = Path(comparison_training_dir) / str(experiment_name) / "summary.json"
    summary_payload = load_json_artifact(summary_path)
    return {
        "selection_path": str(selection_path).replace("\\", "/"),
        "architecture": str(selection_payload["selected_architecture"]),
        "experiment_name": str(experiment_name),
        "seed": int(summary_payload["seed"]),
        "checkpoint_path": str(summary_payload["best_checkpoint_path"]),
        "summary_path": str(summary_path).replace("\\", "/"),
        "best_epoch": int(summary_payload["best_epoch"]),
        "validation_macro_f1": float(summary_payload["best_validation_macro_f1"]),
        "dataset_fingerprint": str(summary_payload["dataset_fingerprint"]),
        "batch_size": int(summary_payload["batch_size"]),
        "summary_payload": summary_payload,
        "selected_row": selected_row,
    }


def build_benchmark_metadata_audit(
    summary_payloads: Sequence[dict[str, object]],
    *,
    dataset_fingerprint: str,
    git_commit: str | None,
) -> dict[str, object]:
    run_audits: list[dict[str, object]] = []
    incomplete_runs: list[str] = []

    for payload in summary_payloads:
        duration_audit = audit_training_duration_payload(payload)
        run_record = {
            "experiment_name": str(payload["experiment_name"]),
            "architecture": str(payload["architecture"]),
            "seed": int(payload["seed"]),
            "epochs_completed": int(payload["epochs_completed"]),
            "resumed_from_epoch": payload.get("resumed_from_epoch"),
            **duration_audit,
        }
        run_audits.append(run_record)
        if duration_audit["status"] != "complete":
            incomplete_runs.append(str(payload["experiment_name"]))

    return {
        "dataset_fingerprint": dataset_fingerprint,
        "git_commit": git_commit,
        "audit_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "duration_accounting_policy": {
            "complete": "duration is cumulative across all recorded training segments",
            "incomplete": "only the latest resumed segment duration is known; cumulative total is unavailable",
            "unknown": "no trustworthy duration information exists in the available payload",
        },
        "runs": run_audits,
        "incomplete_runs": incomplete_runs,
    }


def aggregate_architecture_runs(run_summaries: Sequence[RunSummary]) -> list[dict[str, object]]:
    if not run_summaries:
        raise ValueError("At least one run summary is required for aggregation.")

    grouped: dict[str, list[RunSummary]] = defaultdict(list)
    for run_summary in run_summaries:
        grouped[run_summary.architecture].append(run_summary)

    aggregated_rows: list[dict[str, object]] = []
    for architecture, runs in sorted(grouped.items()):
        ordered_runs = sorted(runs, key=lambda row: row.seed)
        macro_f1_values = [row.best_validation_macro_f1 for row in ordered_runs]
        accuracy_values = [row.best_validation_accuracy for row in ordered_runs]
        best_run = max(ordered_runs, key=lambda row: row.best_validation_macro_f1)
        worst_run = min(ordered_runs, key=lambda row: row.best_validation_macro_f1)
        peak_vram_values = [row.peak_vram_mb for row in ordered_runs if row.peak_vram_mb is not None]
        peak_benchmark_vram_values = [
            row.peak_benchmark_vram_mb for row in ordered_runs if row.peak_benchmark_vram_mb is not None
        ]

        aggregated_rows.append(
            {
                "architecture": architecture,
                "run_count": len(ordered_runs),
                "experiment_names": [row.experiment_name for row in ordered_runs],
                "seeds": [row.seed for row in ordered_runs],
                "batch_sizes": [row.batch_size for row in ordered_runs],
                "learning_rates": [row.learning_rate for row in ordered_runs],
                "mixed_precision_modes": sorted({row.mixed_precision for row in ordered_runs}),
                "weights_name": best_run.weights_name,
                "parameter_count": best_run.parameter_count,
                "trainable_parameter_count": best_run.trainable_parameter_count,
                "estimated_model_size_mb": best_run.estimated_model_size_mb,
                "checkpoint_size_mb_mean": _mean([row.checkpoint_size_mb for row in ordered_runs]),
                "checkpoint_size_mb_max": max(row.checkpoint_size_mb for row in ordered_runs),
                "input_size": list(best_run.input_size),
                "class_mapping": dict(best_run.class_mapping),
                "best_validation_accuracy": max(accuracy_values),
                "mean_validation_accuracy": _mean(accuracy_values),
                "std_validation_accuracy": _std(accuracy_values),
                "best_validation_macro_f1": max(macro_f1_values),
                "mean_validation_macro_f1": _mean(macro_f1_values),
                "std_validation_macro_f1": _std(macro_f1_values),
                "worst_validation_macro_f1": min(macro_f1_values),
                "best_validation_loss": min(row.best_validation_loss for row in ordered_runs),
                "best_epoch": best_run.best_epoch,
                "training_duration_seconds_mean": (
                    _mean([row.training_duration_seconds for row in ordered_runs if row.training_duration_seconds is not None])
                    if all(row.training_duration_seconds is not None for row in ordered_runs)
                    else None
                ),
                "training_duration_seconds_max": (
                    max(row.training_duration_seconds for row in ordered_runs if row.training_duration_seconds is not None)
                    if all(row.training_duration_seconds is not None for row in ordered_runs)
                    else None
                ),
                "training_duration_complete": all(row.training_duration_complete for row in ordered_runs),
                "training_duration_note": None
                if all(row.training_duration_complete for row in ordered_runs)
                else "One or more runs have incomplete cumulative training-duration metadata.",
                "mean_per_image_latency_ms": _mean([row.mean_per_image_latency_ms for row in ordered_runs]),
                "throughput_images_per_second": _mean(
                    [row.throughput_images_per_second for row in ordered_runs]
                ),
                "checkpoint_load_time_ms_mean": _mean(
                    [row.checkpoint_load_time_ms for row in ordered_runs if row.checkpoint_load_time_ms is not None]
                )
                if any(row.checkpoint_load_time_ms is not None for row in ordered_runs)
                else None,
                "peak_vram_mb_max": max(peak_vram_values) if peak_vram_values else None,
                "peak_benchmark_vram_mb_max": max(peak_benchmark_vram_values) if peak_benchmark_vram_values else None,
                "best_run_experiment_name": best_run.experiment_name,
                "best_run_seed": best_run.seed,
                "worst_run_experiment_name": worst_run.experiment_name,
                "worst_run_seed": worst_run.seed,
                "dataset_fingerprint": best_run.dataset_fingerprint,
            }
        )

    return aggregated_rows


def rank_architecture_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("At least one aggregated architecture row is required for ranking.")

    best_mean_f1 = max(float(row["mean_validation_macro_f1"]) for row in rows)
    best_mean_accuracy = max(float(row["mean_validation_accuracy"]) for row in rows)

    ranked = sorted(
        rows,
        key=lambda row: (
            0
            if (
                best_mean_f1 - float(row["mean_validation_macro_f1"]) <= VALIDATION_F1_TOLERANCE
                and best_mean_accuracy - float(row["mean_validation_accuracy"]) <= VALIDATION_ACCURACY_TOLERANCE
            )
            else 1,
            float(row["mean_per_image_latency_ms"]),
            float(row["checkpoint_size_mb_mean"]),
            int(row["parameter_count"]),
            -float(row["mean_validation_macro_f1"]),
        ),
    )
    return ranked


def choose_architecture_from_validation(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    ranked = rank_architecture_rows(rows)
    selected = ranked[0]
    best_mean_f1 = max(float(row["mean_validation_macro_f1"]) for row in rows)

    f1_gap = best_mean_f1 - float(selected["mean_validation_macro_f1"])
    near_best_candidates = [
        row
        for row in rows
        if best_mean_f1 - float(row["mean_validation_macro_f1"]) <= VALIDATION_F1_TOLERANCE
    ]
    rationale = (
        "Selected the fastest and smallest architecture among candidates that remained within "
        f"{VALIDATION_F1_TOLERANCE:.3f} macro-F1 of the best mean validation score."
        if len(near_best_candidates) > 1 and abs(f1_gap) <= VALIDATION_F1_TOLERANCE
        else "Selected the architecture with the strongest mean validation macro F1."
    )
    return {
        "selected_row": selected,
        "ranked_rows": ranked,
        "rationale": rationale,
        "validation_f1_tolerance": VALIDATION_F1_TOLERANCE,
        "validation_accuracy_tolerance": VALIDATION_ACCURACY_TOLERANCE,
    }


def build_selection_decision_payload(
    *,
    selected_architecture: str,
    selected_experiment_names: list[str],
    dataset_fingerprint: str,
    git_commit: str | None,
    validation_basis: dict[str, object],
    engineering_tradeoffs: dict[str, object],
) -> dict[str, object]:
    return {
        "selected_architecture": selected_architecture,
        "selected_experiment_names": list(selected_experiment_names),
        "dataset_fingerprint": dataset_fingerprint,
        "validation_basis": validation_basis,
        "engineering_tradeoffs": engineering_tradeoffs,
        "decision_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "git_commit": git_commit,
        "selection_rule": (
            "validation-only architecture ranking using predictive performance plus "
            "latency, parameter count, checkpoint size, VRAM fit, and training stability"
        ),
        "test_evaluated_before_freeze": False,
    }
