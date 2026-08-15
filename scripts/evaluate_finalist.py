"""Formal held-out evaluation for the frozen Phase 2C challenger finalist."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch

from brainscan.comparison import (
    build_benchmark_metadata_audit,
    load_json_artifact,
    resolve_frozen_finalist,
    save_json_artifact,
)
from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.data import BrainMRIDataset, CANONICAL_CLASS_NAMES, CANONICAL_CLASS_TO_ID
from brainscan.data.preprocessing import build_eval_transform
from brainscan.evaluation import (
    annotate_predictions_with_near_duplicates,
    benchmark_model_inference,
    build_confusion_pairs,
    build_final_architecture_decision_payload,
    build_predictions_rows,
    compute_classification_metrics_bundle,
    compute_confidence_statistics,
    compute_mcnemar_exact,
    compute_paired_disagreement,
    count_confusion_pair,
    create_error_rows,
    evaluate_classifier,
    load_near_duplicate_annotations,
    load_prediction_rows,
    plot_confusion_matrix_figure,
    plot_pr_curves,
    plot_roc_curves,
    summarize_filtered_subset,
    verify_checkpoint_metadata,
    write_evaluation_artifacts,
)
from brainscan.models import build_classifier
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device
from brainscan.training.train_classifier import get_git_commit


COMPARISON_TRAINING_DIR = Path("artifacts/training/comparison")
SELECTION_DECISION_PATH = Path("artifacts/model_comparison/selection_decision.json")
BENCHMARK_AUDIT_PATH = Path("artifacts/model_comparison/benchmark_metadata_audit.json")
FINAL_DECISION_PATH = Path("artifacts/model_comparison/final_architecture_decision.json")
BASELINE_METRICS_PATH = Path("artifacts/evaluation/resnet18_baseline/metrics.json")
BASELINE_PREDICTIONS_PATH = Path("artifacts/evaluation/resnet18_baseline/predictions.csv")
BASELINE_CONFUSION_PAIRS_PATH = Path("artifacts/evaluation/resnet18_baseline/confusion_pairs.csv")
NEAR_DUPLICATE_REPORT_PATH = Path("artifacts/data_audit/near_duplicates.csv")
FINALIST_EVALUATION_DIR = Path("artifacts/evaluation/densenet121_finalist")


def _load_summary_payloads() -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for summary_path in sorted(resolve_project_path(COMPARISON_TRAINING_DIR).glob("*/summary.json")):
        payload = load_json_artifact(summary_path)
        if bool(payload.get("smoke_test")):
            continue
        payloads.append(payload)
    if not payloads:
        raise ValueError("No non-smoke comparison summary payloads were found.")
    return payloads


def _save_json(payload: object, path: Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    return load_prediction_rows(path)


def _select_baseline_architecture_row() -> dict[str, object]:
    comparison_payload = load_json_artifact("artifacts/model_comparison/model_comparison.json")
    architectures = comparison_payload.get("architectures")
    if not isinstance(architectures, list):
        raise TypeError("Comparison payload is missing an 'architectures' list.")
    for row in architectures:
        if isinstance(row, dict) and str(row.get("architecture")) == "resnet18":
            return row
    raise ValueError("ResNet18 architecture row not found in model comparison payload.")


def _build_test_loader(config: dict[str, object], *, batch_size: int) -> tuple[torch.utils.data.DataLoader, set[str]]:
    split_manifest_dir = Path(config["dataset"]["split_manifest_dir"])
    test_dataset = BrainMRIDataset(
        manifest_path=split_manifest_dir / "test.csv",
        transform=build_eval_transform(config["training"]["image_size"]),
        return_metadata=True,
    )
    expected_test_paths = {record.relative_path for record in test_dataset.records}
    if any(record.split != "test" for record in test_dataset.records):
        raise ValueError("The manifest-backed evaluation dataset contains non-test split rows.")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    return test_loader, expected_test_paths


def main() -> None:
    finalist = resolve_frozen_finalist(
        selection_path=SELECTION_DECISION_PATH,
        comparison_training_dir=COMPARISON_TRAINING_DIR,
    )
    selection_payload = load_json_artifact(SELECTION_DECISION_PATH)
    if selection_payload["test_evaluated_before_freeze"] is not False:
        raise ValueError("Selection artifact indicates test evaluation occurred before freeze.")

    summary_payload = finalist["summary_payload"]
    assert isinstance(summary_payload, dict)
    if str(finalist["architecture"]) != "densenet121":
        raise ValueError(f"Frozen finalist architecture mismatch: expected densenet121, found {finalist['architecture']}")

    print("frozen_finalist=")
    print(f"  architecture={finalist['architecture']}")
    print(f"  experiment_name={finalist['experiment_name']}")
    print(f"  seed={finalist['seed']}")
    print(f"  checkpoint_path={finalist['checkpoint_path']}")
    print(f"  best_epoch={finalist['best_epoch']}")
    print(f"  validation_macro_f1={finalist['validation_macro_f1']}")

    summary_payloads = _load_summary_payloads()
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]
    benchmark_audit = build_benchmark_metadata_audit(
        summary_payloads,
        dataset_fingerprint=dataset_fingerprint,
        git_commit=get_git_commit(),
    )
    save_json_artifact(benchmark_audit, BENCHMARK_AUDIT_PATH)

    checkpoint_path = Path(str(finalist["checkpoint_path"]))
    resolved_checkpoint_path = resolve_project_path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen finalist checkpoint not found: {resolved_checkpoint_path}")

    config = load_train_config("configs/train.yaml")
    if list(config["training"]["image_size"]) != list(summary_payload["model_metadata"]["input_size"]):
        raise ValueError("Finalist summary input size does not match the configured evaluation preprocessing.")

    checkpoint_load_start = time.perf_counter()
    model = build_classifier(
        architecture=str(finalist["architecture"]),
        num_classes=4,
        pretrained=False,
    )
    checkpoint = load_checkpoint(checkpoint_path, model, map_location="cpu")
    checkpoint_load_end = time.perf_counter()

    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_architecture=str(finalist["architecture"]),
        expected_epoch=int(finalist["best_epoch"]),
        expected_class_mapping=dict(CANONICAL_CLASS_TO_ID),
        expected_validation_score=float(finalist["validation_macro_f1"]),
    )
    if int(checkpoint_metadata["seed"]) != int(finalist["seed"]):
        raise ValueError(
            f"Checkpoint seed mismatch: expected {finalist['seed']}, found {checkpoint_metadata['seed']}"
        )
    if str(checkpoint_metadata.get("experiment_name")) != str(finalist["experiment_name"]):
        raise ValueError(
            "Checkpoint experiment_name does not match the frozen finalist experiment metadata."
        )

    device, device_info = resolve_device(prefer_cuda=True)
    model.to(device)
    model.eval()

    test_loader, expected_test_paths = _build_test_loader(config, batch_size=int(finalist["batch_size"]))
    evaluation_start = time.perf_counter()
    predictions = evaluate_classifier(model, test_loader, device, class_names=CANONICAL_CLASS_NAMES)
    evaluation_end = time.perf_counter()

    if len(predictions["relative_paths"]) != 702:
        raise ValueError(f"Expected exactly 702 test predictions, found {len(predictions['relative_paths'])}.")
    if set(predictions["relative_paths"]) != expected_test_paths:
        raise ValueError("Evaluation predictions do not exactly match the test manifest rows.")

    prediction_rows = build_predictions_rows(
        relative_paths=predictions["relative_paths"],
        true_ids=predictions["true_ids"],
        predicted_ids=predictions["predicted_ids"],
        probabilities=predictions["probabilities"],
        class_names=CANONICAL_CLASS_NAMES,
    )
    annotations = load_near_duplicate_annotations(
        NEAR_DUPLICATE_REPORT_PATH,
        test_relative_paths=predictions["relative_paths"],
    )
    prediction_rows = annotate_predictions_with_near_duplicates(prediction_rows, annotations)
    error_rows = create_error_rows(prediction_rows)
    confusion_pairs = build_confusion_pairs(prediction_rows)
    metrics_bundle = compute_classification_metrics_bundle(
        true_ids=predictions["true_ids"],
        predicted_ids=predictions["predicted_ids"],
        probabilities=predictions["probabilities"],
        class_names=CANONICAL_CLASS_NAMES,
    )
    confidence_stats = compute_confidence_statistics(prediction_rows)
    filtered_subset_summary = summarize_filtered_subset(prediction_rows, class_names=CANONICAL_CLASS_NAMES)
    if filtered_subset_summary["metrics"] is not None:
        subset_metrics = filtered_subset_summary["metrics"]
        filtered_subset_summary["metrics"] = {
            "overall_metrics": subset_metrics["overall_metrics"],
            "per_class_metrics": subset_metrics["per_class_metrics"],
            "confusion_matrix": subset_metrics["confusion_matrix"],
        }

    sample_batch = next(iter(test_loader))[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    performance_metrics = benchmark_model_inference(model, sample_batch, device)
    performance_metrics["checkpoint_load_time_ms"] = (checkpoint_load_end - checkpoint_load_start) * 1000.0
    performance_metrics["full_test_wall_time_seconds"] = evaluation_end - evaluation_start
    performance_metrics["end_to_end_images_per_second"] = len(prediction_rows) / max(
        evaluation_end - evaluation_start,
        1e-9,
    )
    performance_metrics["peak_benchmark_vram_mb"] = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else None
    )

    metrics_payload = {
        "checkpoint": {
            "path": str(checkpoint_path).replace("\\", "/"),
            "epoch": int(checkpoint["epoch"]),
            "architecture": checkpoint["metadata"]["architecture"],
            "validation_score": float(checkpoint["metadata"]["validation_score"]),
            "experiment_name": checkpoint["metadata"]["experiment_name"],
            "seed": int(checkpoint["metadata"]["seed"]),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "test_sample_count": len(prediction_rows),
        "class_mapping": dict(CANONICAL_CLASS_TO_ID),
        "overall_metrics": metrics_bundle["overall_metrics"],
        "per_class_metrics": metrics_bundle["per_class_metrics"],
        "evaluation_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_info": device_info,
        "performance": performance_metrics,
        "limitations": {
            "subject_level_leakage_cannot_be_ruled_out": True,
            "near_duplicate_candidates_exist": True,
            "clinical_validation": False,
        },
        "frozen_finalist": {
            "selection_path": str(SELECTION_DECISION_PATH).replace("\\", "/"),
            "architecture": finalist["architecture"],
            "experiment_name": finalist["experiment_name"],
            "seed": finalist["seed"],
            "best_epoch": finalist["best_epoch"],
            "validation_macro_f1": finalist["validation_macro_f1"],
        },
    }
    metrics_payload["confidence_analysis"] = confidence_stats
    metrics_payload["near_duplicate_sensitivity"] = {
        "flagged_test_count": sum(
            bool(row["near_duplicate_to_train"]) or bool(row["near_duplicate_to_val"]) for row in prediction_rows
        ),
        "filtered_subset": filtered_subset_summary,
    }

    artifact_paths = write_evaluation_artifacts(
        output_dir=FINALIST_EVALUATION_DIR,
        prediction_rows=prediction_rows,
        error_rows=error_rows,
        confusion_pairs=confusion_pairs,
        metrics_payload=metrics_payload,
        metrics_bundle=metrics_bundle,
    )
    _save_json(performance_metrics, FINALIST_EVALUATION_DIR / "performance.json")
    plot_confusion_matrix_figure(
        metrics_bundle["confusion_matrix"],
        FINALIST_EVALUATION_DIR / "confusion_matrix.png",
        class_names=CANONICAL_CLASS_NAMES,
    )
    plot_roc_curves(
        metrics_bundle["roc_curves"],
        {class_name: values["roc_auc_ovr"] for class_name, values in metrics_bundle["per_class_metrics"].items()},
        FINALIST_EVALUATION_DIR / "roc_curves.png",
    )
    plot_pr_curves(
        metrics_bundle["pr_curves"],
        {
            class_name: values["pr_auc_average_precision"]
            for class_name, values in metrics_bundle["per_class_metrics"].items()
        },
        FINALIST_EVALUATION_DIR / "pr_curves.png",
    )

    baseline_metrics_payload = load_json_artifact(BASELINE_METRICS_PATH)
    baseline_prediction_rows = _load_csv_rows(BASELINE_PREDICTIONS_PATH)
    challenger_prediction_rows = _load_csv_rows(FINALIST_EVALUATION_DIR / "predictions.csv")
    paired_disagreement = compute_paired_disagreement(baseline_prediction_rows, challenger_prediction_rows)
    mcnemar_result = compute_mcnemar_exact(
        int(paired_disagreement["counts"]["reference_correct_challenger_wrong"]),
        int(paired_disagreement["counts"]["challenger_correct_reference_wrong"]),
    )
    paired_artifact = {
        "paired_counts": paired_disagreement["counts"],
        "mcnemar_exact": mcnemar_result,
    }
    _save_json(paired_artifact, FINALIST_EVALUATION_DIR / "paired_comparison.json")

    baseline_confusion_pairs = _load_csv_rows(BASELINE_CONFUSION_PAIRS_PATH)
    glioma_to_meningioma_baseline = count_confusion_pair(
        baseline_confusion_pairs,
        true_class="glioma",
        predicted_class="meningioma",
    )
    meningioma_to_glioma_baseline = count_confusion_pair(
        baseline_confusion_pairs,
        true_class="meningioma",
        predicted_class="glioma",
    )
    glioma_to_meningioma_challenger = count_confusion_pair(
        confusion_pairs,
        true_class="glioma",
        predicted_class="meningioma",
    )
    meningioma_to_glioma_challenger = count_confusion_pair(
        confusion_pairs,
        true_class="meningioma",
        predicted_class="glioma",
    )

    baseline_architecture_row = _select_baseline_architecture_row()
    selected_row = finalist["selected_row"]
    assert isinstance(selected_row, dict)

    challenger_overall = metrics_payload["overall_metrics"]
    baseline_overall = baseline_metrics_payload["overall_metrics"]
    challenger_errors = len(error_rows)
    baseline_errors = 702 - int(round(float(baseline_overall["accuracy"]) * 702))
    macro_f1_difference = float(challenger_overall["f1_macro"]) - float(baseline_overall["f1_macro"])
    error_difference = baseline_errors - challenger_errors

    final_backbone = "densenet121"
    if macro_f1_difference <= 0.002 and error_difference <= 2:
        final_backbone = "resnet18"

    rationale = (
        "DenseNet121 remains the final BrainScanAI backbone because it improved held-out predictive performance, "
        "reduced total errors, stayed stable across two validation seeds, and still uses fewer parameters than ResNet18. "
        "Its slower inference is acceptable for MRI analysis workloads that are not real-time critical."
        if final_backbone == "densenet121"
        else "ResNet18 remains the final BrainScanAI backbone because DenseNet121 did not show a practically meaningful "
        "held-out improvement large enough to justify its slower inference and downstream migration cost."
    )

    final_decision_payload = build_final_architecture_decision_payload(
        frozen_finalist={
            key: finalist[key]
            for key in ("architecture", "experiment_name", "seed", "checkpoint_path", "best_epoch", "validation_macro_f1")
        },
        dataset_fingerprint=dataset_fingerprint,
        git_commit=get_git_commit(),
        validation_evidence={
            "selection_path": str(SELECTION_DECISION_PATH).replace("\\", "/"),
            "selected_architecture": finalist["architecture"],
            "selected_mean_validation_macro_f1": selected_row["mean_validation_macro_f1"],
            "selected_validation_macro_f1_std": selected_row["std_validation_macro_f1"],
            "selected_run_count": selected_row["run_count"],
            "selected_seeds": selected_row["seeds"],
        },
        challenger_test_metrics={
            "sample_count": len(prediction_rows),
            "overall_metrics": challenger_overall,
            "total_errors": challenger_errors,
            "major_confusion_pairs": confusion_pairs[:5],
            "glioma_to_meningioma": glioma_to_meningioma_challenger,
            "meningioma_to_glioma": meningioma_to_glioma_challenger,
        },
        baseline_test_metrics={
            "sample_count": int(baseline_metrics_payload["test_sample_count"]),
            "overall_metrics": baseline_overall,
            "total_errors": baseline_errors,
            "glioma_to_meningioma": glioma_to_meningioma_baseline,
            "meningioma_to_glioma": meningioma_to_glioma_baseline,
        },
        parameter_comparison={
            "resnet18_parameters": int(baseline_architecture_row["parameter_count"]),
            "densenet121_parameters": int(selected_row["parameter_count"]),
            "resnet18_estimated_model_size_mb": float(baseline_architecture_row["estimated_model_size_mb"]),
            "densenet121_estimated_model_size_mb": float(selected_row["estimated_model_size_mb"]),
            "resnet18_checkpoint_size_mb": float(baseline_architecture_row["checkpoint_size_mb_mean"]),
            "densenet121_checkpoint_size_mb": float(selected_row["checkpoint_size_mb_mean"]),
        },
        latency_comparison={
            "resnet18_comparison_per_image_latency_ms": float(baseline_architecture_row["mean_per_image_latency_ms"]),
            "densenet121_comparison_per_image_latency_ms": float(selected_row["mean_per_image_latency_ms"]),
            "densenet121_formal_test_per_image_latency_ms": float(performance_metrics["mean_per_image_latency_ms"]),
            "resnet18_phase1e_per_image_latency_ms": float(baseline_metrics_payload["performance"]["mean_per_image_latency_ms"]),
            "methodology_note": (
                "Phase 2C comparison latencies are benchmark-only apples-to-apples numbers; "
                "the older Phase 1E baseline latency used the formal evaluation pipeline and is not directly identical."
            ),
        },
        throughput_comparison={
            "resnet18_comparison_throughput_images_per_second": float(
                baseline_architecture_row["throughput_images_per_second"]
            ),
            "densenet121_comparison_throughput_images_per_second": float(
                selected_row["throughput_images_per_second"]
            ),
            "densenet121_formal_test_throughput_images_per_second": float(
                performance_metrics["throughput_images_per_second"]
            ),
        },
        stability_information={
            "densenet121_run_count": int(selected_row["run_count"]),
            "densenet121_mean_validation_macro_f1": float(selected_row["mean_validation_macro_f1"]),
            "densenet121_validation_macro_f1_std": selected_row["std_validation_macro_f1"],
            "densenet121_best_run_experiment_name": selected_row["best_run_experiment_name"],
            "densenet121_worst_run_experiment_name": selected_row["worst_run_experiment_name"],
        },
        benchmark_metadata_audit=benchmark_audit,
        final_backbone=final_backbone,
        rationale=rationale,
        downstream_migration_required=final_backbone == "densenet121",
    )
    final_decision_payload["absolute_macro_f1_difference"] = macro_f1_difference
    final_decision_payload["error_count_difference"] = error_difference
    final_decision_payload["paired_disagreement"] = paired_disagreement["counts"]
    final_decision_payload["mcnemar_exact"] = mcnemar_result
    final_decision_payload["vram_comparison"] = {
        "resnet18_peak_training_vram_mb": baseline_architecture_row["peak_vram_mb_max"],
        "densenet121_peak_training_vram_mb": selected_row["peak_vram_mb_max"],
        "resnet18_peak_benchmark_vram_mb": baseline_architecture_row["peak_benchmark_vram_mb_max"],
        "densenet121_peak_benchmark_vram_mb": selected_row["peak_benchmark_vram_mb_max"],
        "densenet121_formal_test_peak_benchmark_vram_mb": performance_metrics["peak_benchmark_vram_mb"],
    }
    final_decision_payload["engineering_context"] = {
        "existing_resnet18_stack": [
            "gradcam",
            "calibration_analysis",
            "uncertainty_thresholds",
            "ood_reference",
            "safe_inference_pipeline",
            "robustness_evaluation",
        ],
        "migration_note": (
            "DenseNet121 predictive evidence was evaluated independently. Existing ResNet18 explainability, "
            "uncertainty, OOD, and robustness artifacts do not automatically transfer to DenseNet121."
        ),
    }
    final_decision_payload["next_required_migrations"] = (
        [
            "gradcam target-layer resolver update",
            "temperature-scaling and uncertainty verification for DenseNet121",
            "OOD reference rebuild in DenseNet121 feature space",
            "safe inference integration",
            "robustness revalidation",
        ]
        if final_backbone == "densenet121"
        else []
    )
    final_decision_payload["glioma_meningioma_confusion_delta"] = {
        "glioma_to_meningioma_baseline": glioma_to_meningioma_baseline,
        "glioma_to_meningioma_densenet121": glioma_to_meningioma_challenger,
        "meningioma_to_glioma_baseline": meningioma_to_glioma_baseline,
        "meningioma_to_glioma_densenet121": meningioma_to_glioma_challenger,
    }
    save_json_artifact(final_decision_payload, FINAL_DECISION_PATH)

    print("finalist_evaluation_summary=")
    print(f"  selected_architecture={finalist['architecture']}")
    print(f"  selected_experiment_name={finalist['experiment_name']}")
    print(f"  selected_seed={finalist['seed']}")
    print(f"  checkpoint={resolve_project_path(checkpoint_path)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  test_sample_count={len(prediction_rows)}")
    print(f"  test_accuracy={metrics_payload['overall_metrics']['accuracy']:.6f}")
    print(f"  test_balanced_accuracy={metrics_payload['overall_metrics']['balanced_accuracy']:.6f}")
    print(f"  test_macro_f1={metrics_payload['overall_metrics']['f1_macro']:.6f}")
    print(f"  total_incorrect={len(error_rows)}")
    print(f"  benchmark_metadata_audit={resolve_project_path(BENCHMARK_AUDIT_PATH)}")
    print(f"  metrics_json={artifact_paths['metrics_json']}")
    print(f"  final_architecture_decision={resolve_project_path(FINAL_DECISION_PATH)}")


if __name__ == "__main__":
    main()
