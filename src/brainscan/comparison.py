"""Experiment naming and artifact helpers for controlled model comparison."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re

from brainscan.core.config import make_project_relative_path, resolve_project_path


EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9_\\-]+$")


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
