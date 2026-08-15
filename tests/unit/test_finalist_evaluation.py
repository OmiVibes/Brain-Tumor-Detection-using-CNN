from __future__ import annotations

import inspect
from pathlib import Path
import json

from brainscan.comparison import audit_training_duration_payload, resolve_frozen_finalist, save_json_artifact
from brainscan.data.constants import CANONICAL_CLASS_TO_ID
from brainscan.evaluation.classification import verify_checkpoint_metadata
from brainscan.evaluation.finalist import (
    build_final_architecture_decision_payload,
    compute_mcnemar_exact,
    compute_paired_disagreement,
)


def test_audit_training_duration_marks_resumed_segment_incomplete() -> None:
    audit = audit_training_duration_payload(
        {
            "experiment_name": "convnext_tiny_seed42",
            "resumed_from_epoch": 20,
            "training_duration_seconds": 1.818,
        }
    )
    assert audit["status"] == "incomplete"
    assert audit["corrected_training_duration_seconds"] is None


def test_resolve_frozen_finalist_uses_best_run_metadata(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection_decision.json"
    comparison_dir = tmp_path / "comparison"
    summary_dir = comparison_dir / "densenet121_seed42"
    summary_dir.mkdir(parents=True)
    save_json_artifact(
        {
            "experiment_name": "densenet121_seed42",
            "architecture": "densenet121",
            "seed": 42,
            "best_checkpoint_path": "artifacts/models/comparison/densenet121_seed42/best.pt",
            "best_epoch": 13,
            "best_validation_macro_f1": 0.9944,
            "dataset_fingerprint": "abc123",
            "batch_size": 8,
        },
        summary_dir / "summary.json",
    )
    save_json_artifact(
        {
            "selected_architecture": "densenet121",
            "selected_experiment_names": ["densenet121_seed42", "densenet121_seed1337"],
            "ranked_architectures": [
                {
                    "architecture": "densenet121",
                    "best_run_experiment_name": "densenet121_seed42",
                }
            ],
            "test_evaluated_before_freeze": False,
        },
        selection_path,
    )

    finalist = resolve_frozen_finalist(selection_path=selection_path, comparison_training_dir=comparison_dir)
    assert finalist["architecture"] == "densenet121"
    assert finalist["experiment_name"] == "densenet121_seed42"
    assert finalist["seed"] == 42


def test_verify_checkpoint_metadata_accepts_densenet_finalist() -> None:
    checkpoint = {
        "epoch": 13,
        "metadata": {
            "architecture": "densenet121",
            "num_classes": 4,
            "canonical_class_mapping": dict(CANONICAL_CLASS_TO_ID),
            "dataset_fingerprint": "abc123",
            "image_size": [224, 224],
            "epoch": 13,
            "validation_score": 0.9944,
        },
    }
    metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint="abc123",
        expected_architecture="densenet121",
        expected_epoch=13,
        expected_validation_score=0.9944,
    )
    assert metadata["architecture"] == "densenet121"


def test_paired_disagreement_and_exact_mcnemar() -> None:
    baseline_rows = [
        {"relative_path": "a.jpg", "correct": "True", "predicted_class": "glioma", "true_class": "glioma"},
        {"relative_path": "b.jpg", "correct": "True", "predicted_class": "glioma", "true_class": "glioma"},
        {"relative_path": "c.jpg", "correct": "False", "predicted_class": "glioma", "true_class": "meningioma"},
    ]
    challenger_rows = [
        {"relative_path": "a.jpg", "correct": "True", "predicted_class": "glioma", "true_class": "glioma"},
        {"relative_path": "b.jpg", "correct": "False", "predicted_class": "meningioma", "true_class": "glioma"},
        {"relative_path": "c.jpg", "correct": "True", "predicted_class": "meningioma", "true_class": "meningioma"},
    ]
    disagreement = compute_paired_disagreement(baseline_rows, challenger_rows)
    assert disagreement["counts"]["both_correct"] == 1
    assert disagreement["counts"]["reference_correct_challenger_wrong"] == 1
    assert disagreement["counts"]["challenger_correct_reference_wrong"] == 1
    mcnemar = compute_mcnemar_exact(1, 1)
    assert mcnemar["discordant_pairs"] == 2
    assert 0.0 <= mcnemar["two_sided_p_value"] <= 1.0


def test_final_architecture_decision_payload_keeps_relative_paths() -> None:
    payload = build_final_architecture_decision_payload(
        frozen_finalist={
            "architecture": "densenet121",
            "experiment_name": "densenet121_seed42",
            "seed": 42,
            "checkpoint_path": "artifacts/models/comparison/densenet121_seed42/best.pt",
            "best_epoch": 13,
            "validation_macro_f1": 0.9944,
        },
        dataset_fingerprint="abc123",
        git_commit="deadbeef",
        validation_evidence={"selected_mean_validation_macro_f1": 0.9930},
        challenger_test_metrics={"overall_metrics": {"f1_macro": 0.99}},
        baseline_test_metrics={"overall_metrics": {"f1_macro": 0.98}},
        parameter_comparison={"resnet18_parameters": 10, "densenet121_parameters": 9},
        latency_comparison={"resnet18_comparison_per_image_latency_ms": 1.8},
        throughput_comparison={"resnet18_comparison_throughput_images_per_second": 555.0},
        stability_information={"densenet121_run_count": 2},
        benchmark_metadata_audit={"incomplete_runs": ["convnext_tiny_seed42"]},
        final_backbone="densenet121",
        rationale="DenseNet wins.",
        downstream_migration_required=True,
    )
    serialized = json.dumps(payload)
    assert "artifacts/models/comparison/densenet121_seed42/best.pt" in serialized
    assert ":\\\\" not in serialized


def test_finalist_script_uses_frozen_selection_and_does_not_rerank() -> None:
    import scripts.evaluate_finalist as evaluate_finalist

    source = inspect.getsource(evaluate_finalist)
    assert "resolve_frozen_finalist" in source
    assert "choose_architecture_from_validation" not in source
