from __future__ import annotations

import inspect
from pathlib import Path
import json

import pytest

from brainscan.comparison import (
    build_comparison_paths,
    build_experiment_name,
    build_selection_decision_payload,
    create_comparison_config,
    sanitize_experiment_name,
    save_json_artifact,
)


def test_experiment_name_resolves_safely() -> None:
    assert build_experiment_name("DenseNet121", 42) == "densenet121_seed42"
    with pytest.raises(ValueError):
        sanitize_experiment_name("../bad")


def test_training_artifacts_do_not_overwrite_baseline() -> None:
    paths = build_comparison_paths("densenet121_seed42")
    assert "comparison" in paths["best_checkpoint"].as_posix()
    assert paths["best_checkpoint"].name == "best.pt"
    assert "resnet18_baseline_best.pt" not in paths["best_checkpoint"].as_posix()


def test_comparison_config_uses_validation_only_training_paths() -> None:
    base_config = {
        "training": {"seed": 42, "batch_size": 16, "learning_rate": 0.0001},
        "model": {"architecture": "resnet18"},
        "checkpoint": {},
    }
    config = create_comparison_config(
        base_config,
        architecture="convnext_tiny",
        seed=1337,
        experiment_name="convnext_tiny_seed1337",
        batch_size=4,
    )
    assert config["model"]["architecture"] == "convnext_tiny"
    assert config["training"]["seed"] == 1337
    assert config["training"]["batch_size"] == 4
    assert "comparison" in config["checkpoint"]["model_dir"]


def test_selection_artifact_generated_before_test_evaluation(tmp_path: Path) -> None:
    payload = build_selection_decision_payload(
        selected_architecture="densenet121",
        selected_experiment_names=["densenet121_seed42"],
        dataset_fingerprint="abc123",
        git_commit="deadbeef",
        validation_basis={"best_val_macro_f1": 0.9},
        engineering_tradeoffs={"latency_ms": 10.0},
    )
    artifact_path = tmp_path / "selection_decision.json"
    save_json_artifact(payload, artifact_path)
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved["test_evaluated_before_freeze"] is False
    assert saved["selected_architecture"] == "densenet121"


def test_model_selection_logic_uses_validation_only_language() -> None:
    payload = build_selection_decision_payload(
        selected_architecture="resnet18",
        selected_experiment_names=["resnet18_baseline_seed42"],
        dataset_fingerprint="abc123",
        git_commit="deadbeef",
        validation_basis={"ranking_split": "validation"},
        engineering_tradeoffs={"reason": "latency"},
    )
    serialized = json.dumps(payload)
    assert "validation" in serialized


def test_architecture_ranking_path_does_not_request_test_loader() -> None:
    import scripts.train_comparison as train_comparison

    source = inspect.getsource(train_comparison)
    assert "create_train_val_dataloaders_from_config" in source
    assert "create_test_dataloader" not in source
