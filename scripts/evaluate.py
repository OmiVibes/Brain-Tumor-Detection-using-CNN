"""Formal held-out evaluation for the Phase 1D ResNet18 baseline."""

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

from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.data import BrainMRIDataset, CANONICAL_CLASS_NAMES
from brainscan.data.preprocessing import build_eval_transform
from brainscan.models import build_classifier
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device
from brainscan.evaluation.classification import (
    annotate_predictions_with_near_duplicates,
    benchmark_model_inference,
    build_confusion_pairs,
    build_evaluation_summary_payload,
    build_predictions_rows,
    compute_classification_metrics_bundle,
    compute_confidence_statistics,
    create_error_rows,
    evaluate_classifier,
    load_near_duplicate_annotations,
    plot_confusion_matrix_figure,
    plot_pr_curves,
    plot_roc_curves,
    summarize_filtered_subset,
    verify_checkpoint_metadata,
    write_evaluation_artifacts,
)


CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
TRAINING_HISTORY_PATH = Path("artifacts/training/resnet18_baseline_history.json")
EVALUATION_OUTPUT_DIR = Path("artifacts/evaluation/resnet18_baseline")
NEAR_DUPLICATE_REPORT_PATH = Path("artifacts/data_audit/near_duplicates.csv")


def _save_json(payload: object, path: Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved


def main() -> None:
    config = load_train_config("configs/train.yaml")
    split_manifest_dir = Path(config["dataset"]["split_manifest_dir"])
    test_dataset = BrainMRIDataset(
        manifest_path=split_manifest_dir / "test.csv",
        transform=build_eval_transform(config["training"]["image_size"]),
        return_metadata=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    checkpoint_load_start = time.perf_counter()
    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(CHECKPOINT_PATH, model, map_location="cpu")
    checkpoint_load_end = time.perf_counter()

    history = json.loads(resolve_project_path(TRAINING_HISTORY_PATH).read_text(encoding="utf-8"))
    best_history_entry = next(entry for entry in history if int(entry["epoch"]) == 9)
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]
    verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_validation_score=float(best_history_entry["val_macro_f1"]),
    )

    device, device_info = resolve_device(prefer_cuda=True)
    model.to(device)
    model.eval()

    evaluation_start = time.perf_counter()
    predictions = evaluate_classifier(model, test_loader, device, class_names=CANONICAL_CLASS_NAMES)
    evaluation_end = time.perf_counter()

    if len(predictions["relative_paths"]) != 702:
        raise ValueError(f"Expected exactly 702 test predictions, found {len(predictions['relative_paths'])}.")
    if any(not path.startswith("dataset/test/") for path in predictions["relative_paths"]):
        raise ValueError("Evaluation produced rows outside dataset/test/.")

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

    sample_batch = next(iter(test_loader))[0]
    performance_metrics = benchmark_model_inference(model, sample_batch, device)
    performance_metrics["checkpoint_load_time_ms"] = (checkpoint_load_end - checkpoint_load_start) * 1000.0
    performance_metrics["full_test_wall_time_seconds"] = evaluation_end - evaluation_start
    performance_metrics["end_to_end_images_per_second"] = len(prediction_rows) / max(
        evaluation_end - evaluation_start,
        1e-9,
    )

    metrics_payload = build_evaluation_summary_payload(
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint=checkpoint,
        metrics_bundle=metrics_bundle,
        prediction_rows=prediction_rows,
        performance_metrics=performance_metrics,
        dataset_fingerprint=dataset_fingerprint,
        device_info=device_info,
    )
    metrics_payload["confidence_analysis"] = confidence_stats
    metrics_payload["near_duplicate_sensitivity"] = {
        "flagged_test_count": sum(
            bool(row["near_duplicate_to_train"]) or bool(row["near_duplicate_to_val"]) for row in prediction_rows
        ),
        "filtered_subset": filtered_subset_summary,
    }

    artifact_paths = write_evaluation_artifacts(
        output_dir=EVALUATION_OUTPUT_DIR,
        prediction_rows=prediction_rows,
        error_rows=error_rows,
        confusion_pairs=confusion_pairs,
        metrics_payload=metrics_payload,
        metrics_bundle=metrics_bundle,
    )
    _save_json(performance_metrics, EVALUATION_OUTPUT_DIR / "performance.json")

    plot_confusion_matrix_figure(
        metrics_bundle["confusion_matrix"],
        EVALUATION_OUTPUT_DIR / "confusion_matrix.png",
        class_names=CANONICAL_CLASS_NAMES,
    )
    plot_roc_curves(
        metrics_bundle["roc_curves"],
        {class_name: values["roc_auc_ovr"] for class_name, values in metrics_bundle["per_class_metrics"].items()},
        EVALUATION_OUTPUT_DIR / "roc_curves.png",
    )
    plot_pr_curves(
        metrics_bundle["pr_curves"],
        {
            class_name: values["pr_auc_average_precision"]
            for class_name, values in metrics_bundle["per_class_metrics"].items()
        },
        EVALUATION_OUTPUT_DIR / "pr_curves.png",
    )

    print("held_out_evaluation_summary=")
    print(f"  checkpoint={resolve_project_path(CHECKPOINT_PATH)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  test_sample_count={len(prediction_rows)}")
    print(f"  test_accuracy={metrics_payload['overall_metrics']['accuracy']:.6f}")
    print(f"  test_balanced_accuracy={metrics_payload['overall_metrics']['balanced_accuracy']:.6f}")
    print(f"  test_macro_f1={metrics_payload['overall_metrics']['f1_macro']:.6f}")
    print(f"  total_incorrect={len(error_rows)}")
    print(
        "  near_duplicate_flagged_test_count="
        f"{metrics_payload['near_duplicate_sensitivity']['flagged_test_count']}"
    )
    print(f"  metrics_json={artifact_paths['metrics_json']}")
    print(f"  predictions_csv={artifact_paths['predictions_csv']}")
    print(f"  errors_csv={artifact_paths['errors_csv']}")


if __name__ == "__main__":
    main()
