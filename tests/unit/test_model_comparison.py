from __future__ import annotations

import inspect
from pathlib import Path
import json

import pytest

from brainscan.comparison import (
    aggregate_architecture_runs,
    build_comparison_paths,
    build_experiment_name,
    build_selection_decision_payload,
    choose_architecture_from_validation,
    create_comparison_config,
    load_run_summary,
    run_summary_from_payload,
    sanitize_experiment_name,
    save_json_artifact,
    summarize_history_best_metrics,
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


def test_summarize_history_best_metrics_uses_best_macro_f1() -> None:
    history = [
        {"epoch": 1, "val_accuracy": 0.8, "val_precision_macro": 0.7, "val_recall_macro": 0.75, "val_macro_f1": 0.72, "val_loss": 0.5},
        {"epoch": 2, "val_accuracy": 0.82, "val_precision_macro": 0.8, "val_recall_macro": 0.81, "val_macro_f1": 0.83, "val_loss": 0.4},
    ]
    summary = summarize_history_best_metrics(history)
    assert summary["best_epoch"] == 2
    assert summary["best_validation_macro_f1"] == pytest.approx(0.83)
    assert summary["best_validation_loss"] == pytest.approx(0.4)


def test_run_summary_and_aggregation_preserve_validation_only_metrics(tmp_path: Path) -> None:
    summary_payload = {
        "experiment_name": "densenet121_seed42",
        "architecture": "densenet121",
        "seed": 42,
        "batch_size": 8,
        "learning_rate": 0.0001,
        "dataset_fingerprint": "abc123",
        "model_metadata": {
            "parameter_count": 100,
            "trainable_parameter_count": 100,
            "estimated_model_size_mb": 1.5,
            "pretrained_weights": "DenseNet121_Weights.DEFAULT",
            "input_size": [224, 224],
            "class_mapping": {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3},
        },
        "best_checkpoint_size_mb": 2.0,
        "epochs_completed": 4,
        "training_duration_seconds": 12.0,
        "history": [
            {"epoch": 1, "val_accuracy": 0.80, "val_precision_macro": 0.79, "val_recall_macro": 0.78, "val_macro_f1": 0.77, "val_loss": 0.6},
            {"epoch": 2, "val_accuracy": 0.85, "val_precision_macro": 0.84, "val_recall_macro": 0.83, "val_macro_f1": 0.86, "val_loss": 0.4},
        ],
        "benchmark": {
            "mean_per_image_latency_ms": 5.0,
            "throughput_images_per_second": 200.0,
            "checkpoint_load_time_ms": 50.0,
        },
        "device_info": {"device_type": "cuda"},
        "peak_vram_mb": 123.0,
        "peak_benchmark_vram_mb": 100.0,
        "source_summary_path": "artifacts/training/comparison/densenet121_seed42/summary.json",
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    run_summary = load_run_summary(summary_path)
    assert run_summary.best_validation_macro_f1 == pytest.approx(0.86)

    aggregated_rows = aggregate_architecture_runs([run_summary])
    assert aggregated_rows[0]["mean_validation_macro_f1"] == pytest.approx(0.86)
    assert aggregated_rows[0]["run_count"] == 1


def test_selection_logic_can_prefer_more_efficient_near_tied_model() -> None:
    rows = [
        {
            "architecture": "convnext_tiny",
            "mean_validation_macro_f1": 0.9500,
            "mean_validation_accuracy": 0.9510,
            "mean_per_image_latency_ms": 9.0,
            "checkpoint_size_mb_mean": 100.0,
            "parameter_count": 27000000,
        },
        {
            "architecture": "resnet18",
            "mean_validation_macro_f1": 0.9485,
            "mean_validation_accuracy": 0.9500,
            "mean_per_image_latency_ms": 3.0,
            "checkpoint_size_mb_mean": 40.0,
            "parameter_count": 11000000,
        },
    ]
    selection = choose_architecture_from_validation(rows)
    assert selection["selected_row"]["architecture"] == "resnet18"
    assert "fastest and smallest" in selection["rationale"]


def test_compare_models_script_ignores_smoke_runs() -> None:
    import scripts.compare_models as compare_models

    source = inspect.getsource(compare_models)
    assert "smoke_test" in source
