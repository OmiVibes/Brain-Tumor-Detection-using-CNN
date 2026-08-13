"""Validation-only temperature scaling and uncertainty analysis for BrainScanAI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import torch

from brainscan.core.config import load_train_config, make_project_relative_path, resolve_project_path
from brainscan.data import BrainMRIDataset, CANONICAL_CLASS_NAMES
from brainscan.data.preprocessing import build_eval_transform
from brainscan.evaluation.classification import (
    build_predictions_rows,
    compute_classification_metrics_bundle,
    create_error_rows,
    verify_checkpoint_metadata,
)
from brainscan.models import build_classifier
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device
from brainscan.training.train_classifier import get_git_commit
from brainscan.uncertainty import (
    TemperatureScaler,
    assign_uncertainty_levels,
    build_temperature_artifact_payload,
    collect_logits_from_dataloader,
    compute_probability_metrics,
    derive_uncertainty_thresholds,
    error_detection_auroc,
    fit_temperature_from_dataloader,
    save_temperature_artifact,
    softmax_probabilities_from_logits,
    summarize_uncertainty_by_correctness,
)


CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
TRAINING_HISTORY_PATH = Path("artifacts/training/resnet18_baseline_history.json")
PHASE1E_ERRORS_CSV_PATH = Path("artifacts/evaluation/resnet18_baseline/errors.csv")
OUTPUT_DIR = Path("artifacts/calibration/resnet18_baseline")
TEMPERATURE_ARTIFACT_PATH = OUTPUT_DIR / "temperature.json"
SUMMARY_ARTIFACT_PATH = OUTPUT_DIR / "metrics.json"
RELIABILITY_BEFORE_JSON_PATH = OUTPUT_DIR / "reliability_before.json"
RELIABILITY_AFTER_JSON_PATH = OUTPUT_DIR / "reliability_after.json"
RELIABILITY_BEFORE_CSV_PATH = OUTPUT_DIR / "reliability_before.csv"
RELIABILITY_AFTER_CSV_PATH = OUTPUT_DIR / "reliability_after.csv"
RELIABILITY_BEFORE_PNG_PATH = OUTPUT_DIR / "reliability_before.png"
RELIABILITY_AFTER_PNG_PATH = OUTPUT_DIR / "reliability_after.png"


def _save_json(payload: object, path: Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved


def _save_reliability_csv(bin_rows: list[dict[str, object]], path: Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not bin_rows:
        raise ValueError("Reliability export requires at least one bin row.")
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)
    return resolved


def _plot_reliability_diagram(bin_rows: list[dict[str, object]], path: Path, title: str) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    occupied_rows = [row for row in bin_rows if int(row["sample_count"]) > 0]
    centers = [(float(row["lower_bound"]) + float(row["upper_bound"])) / 2.0 for row in occupied_rows]
    accuracies = [float(row["empirical_accuracy"]) for row in occupied_rows]
    widths = [float(row["upper_bound"]) - float(row["lower_bound"]) for row in occupied_rows]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.bar(
        centers,
        accuracies,
        width=widths,
        alpha=0.35,
        align="center",
        edgecolor="black",
        label="Observed accuracy by occupied bin",
    )
    plt.plot(centers, accuracies, marker="o", color="tab:blue")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved)
    plt.close()
    return resolved


def _build_split_loader(config: dict[str, object], split_name: str) -> torch.utils.data.DataLoader:
    split_manifest_dir = Path(config["dataset"]["split_manifest_dir"])
    dataset = BrainMRIDataset(
        manifest_path=split_manifest_dir / f"{split_name}.csv",
        transform=build_eval_transform(config["training"]["image_size"]),
        return_metadata=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )


def _build_metrics_summary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
) -> dict[str, object]:
    summary = compute_probability_metrics(logits, labels, n_bins=n_bins)
    classification_bundle = compute_classification_metrics_bundle(
        true_ids=labels.cpu().tolist(),
        predicted_ids=summary["predicted_ids"].cpu().tolist(),  # type: ignore[union-attr]
        probabilities=summary["probabilities"].cpu().tolist(),  # type: ignore[union-attr]
        class_names=CANONICAL_CLASS_NAMES,
    )
    summary["classification_bundle"] = classification_bundle
    return summary


def _extract_confidence_rows(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    relative_paths: list[str],
) -> list[dict[str, object]]:
    prediction_rows = build_predictions_rows(
        relative_paths=relative_paths,
        true_ids=labels.cpu().tolist(),
        predicted_ids=probabilities.argmax(dim=1).cpu().tolist(),
        probabilities=probabilities.cpu().tolist(),
        class_names=CANONICAL_CLASS_NAMES,
    )
    return prediction_rows


def _revisit_high_confidence_errors(
    raw_prediction_rows: list[dict[str, object]],
    calibrated_probabilities: torch.Tensor,
) -> dict[str, list[dict[str, object]]]:
    raw_errors = create_error_rows(raw_prediction_rows)
    raw_by_path = {str(row["relative_path"]): row for row in raw_prediction_rows}
    calibrated_by_path = {
        str(row["relative_path"]): {
            "calibrated_confidence": float(row["confidence"]),
            "entropy": float(row["entropy"]),
            "margin": float(row["margin"]),
        }
        for row in [
            {
                "relative_path": raw_row["relative_path"],
                "confidence": calibrated_probabilities[index].max().item(),
                "entropy": compute_probability_metrics(
                    torch.log(calibrated_probabilities[index].unsqueeze(0)),
                    torch.tensor([int(raw_row["true_class_id"])], dtype=torch.int64),
                    n_bins=15,
                )["entropy"][0].item(),
                "margin": compute_probability_metrics(
                    torch.log(calibrated_probabilities[index].unsqueeze(0)),
                    torch.tensor([int(raw_row["true_class_id"])], dtype=torch.int64),
                    n_bins=15,
                )["top1_top2_margin"][0].item(),
            }
            for index, raw_row in enumerate(raw_prediction_rows)
        ]
    }

    def _rows(threshold: float) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for error_row in raw_errors:
            if float(error_row["model_confidence"]) < threshold:
                continue
            path = str(error_row["relative_path"])
            base_row = raw_by_path[path]
            calibrated = calibrated_by_path[path]
            rows.append(
                {
                    "relative_path": path,
                    "true_class": error_row["true_class"],
                    "predicted_class": error_row["predicted_class"],
                    "raw_confidence": float(error_row["model_confidence"]),
                    "calibrated_confidence": float(calibrated["calibrated_confidence"]),
                    "entropy": float(calibrated["entropy"]),
                    "top1_top2_margin": float(calibrated["margin"]),
                    "true_class_id": int(base_row["true_class_id"]),
                }
            )
        return rows

    return {
        "incorrect_confidence_ge_090": _rows(0.90),
        "incorrect_confidence_ge_095": _rows(0.95),
    }


def main() -> None:
    config = load_train_config("configs/train.yaml")
    history = json.loads(resolve_project_path(TRAINING_HISTORY_PATH).read_text(encoding="utf-8"))
    best_history_entry = next(entry for entry in history if int(entry["epoch"]) == 9)

    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(CHECKPOINT_PATH, model, map_location="cpu")
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]
    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_validation_score=float(best_history_entry["val_macro_f1"]),
    )
    checkpoint_metadata = {**checkpoint_metadata, "epoch": int(checkpoint["epoch"])}

    device, device_info = resolve_device(prefer_cuda=True)
    model.to(device)
    model.eval()

    val_loader = _build_split_loader(config, "val")
    test_loader = _build_split_loader(config, "test")

    scaler, val_logits_bundle, fit_result = fit_temperature_from_dataloader(
        model,
        val_loader,
        device,
        expected_split="val",
    )
    val_logits = val_logits_bundle["logits"]  # type: ignore[assignment]
    val_labels = val_logits_bundle["labels"]  # type: ignore[assignment]
    raw_val_summary = _build_metrics_summary(val_logits, val_labels)
    calibrated_val_logits = scaler(val_logits)
    calibrated_val_summary = _build_metrics_summary(calibrated_val_logits, val_labels)

    test_logits_bundle = collect_logits_from_dataloader(model, test_loader, device, expected_split="test")
    test_logits = test_logits_bundle["logits"]  # type: ignore[assignment]
    test_labels = test_logits_bundle["labels"]  # type: ignore[assignment]
    test_relative_paths = test_logits_bundle["relative_paths"]  # type: ignore[assignment]
    raw_test_summary = _build_metrics_summary(test_logits, test_labels)
    calibrated_test_logits = scaler(test_logits)
    calibrated_test_summary = _build_metrics_summary(calibrated_test_logits, test_labels)

    raw_test_predicted = raw_test_summary["predicted_ids"]  # type: ignore[assignment]
    calibrated_test_predicted = calibrated_test_summary["predicted_ids"]  # type: ignore[assignment]
    if not torch.equal(raw_test_predicted, calibrated_test_predicted):
        raise ValueError("Scalar temperature scaling changed test-set predicted class IDs materially.")

    thresholds = derive_uncertainty_thresholds(
        raw_values := calibrated_val_summary["normalized_entropy"].cpu().tolist(),  # type: ignore[union-attr]
        margin_values := calibrated_val_summary["top1_top2_margin"].cpu().tolist(),  # type: ignore[union-attr]
    )
    uncertainty_levels = assign_uncertainty_levels(
        calibrated_test_summary["normalized_entropy"].cpu().tolist(),  # type: ignore[union-attr]
        calibrated_test_summary["top1_top2_margin"].cpu().tolist(),  # type: ignore[union-attr]
        thresholds,
    )
    level_counts = {
        level: uncertainty_levels.count(level)
        for level in ("LOW", "MEDIUM", "HIGH")
    }

    raw_test_probabilities = softmax_probabilities_from_logits(test_logits)
    calibrated_test_probabilities = softmax_probabilities_from_logits(calibrated_test_logits)
    raw_prediction_rows = _extract_confidence_rows(raw_test_probabilities, test_labels, test_relative_paths)
    high_confidence_error_changes = _revisit_high_confidence_errors(raw_prediction_rows, calibrated_test_probabilities)

    before_rows = raw_val_summary["calibration_bins"]  # type: ignore[assignment]
    after_rows = calibrated_val_summary["calibration_bins"]  # type: ignore[assignment]
    _save_json(before_rows, RELIABILITY_BEFORE_JSON_PATH)
    _save_json(after_rows, RELIABILITY_AFTER_JSON_PATH)
    _save_reliability_csv(before_rows, RELIABILITY_BEFORE_CSV_PATH)
    _save_reliability_csv(after_rows, RELIABILITY_AFTER_CSV_PATH)
    _plot_reliability_diagram(before_rows, RELIABILITY_BEFORE_PNG_PATH, "ResNet18 Reliability Before Calibration")
    _plot_reliability_diagram(after_rows, RELIABILITY_AFTER_PNG_PATH, "ResNet18 Reliability After Calibration")

    timestamp = datetime.now(UTC).isoformat()
    artifact_payload = build_temperature_artifact_payload(
        temperature=float(scaler.temperature.item()),
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_epoch=int(checkpoint["epoch"]),
        dataset_fingerprint=dataset_fingerprint,
        validation_metrics_before=raw_val_summary,
        validation_metrics_after=calibrated_val_summary,
        git_commit=get_git_commit(),
        generated_timestamp_utc=timestamp,
    )
    save_temperature_artifact(artifact_payload, TEMPERATURE_ARTIFACT_PATH)

    summary_payload = {
        "checkpoint": make_project_relative_path(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": checkpoint_metadata["architecture"],
        "dataset_fingerprint": dataset_fingerprint,
        "device_info": device_info,
        "calibration_method": "temperature_scaling",
        "validation_sample_count": len(val_logits_bundle["relative_paths"]),
        "test_sample_count": len(test_relative_paths),
        "temperature": float(scaler.temperature.item()),
        "fit_result": fit_result,
        "validation_before": {
            "accuracy": raw_val_summary["accuracy"],
            "macro_f1": raw_val_summary["macro_f1"],
            "ece": raw_val_summary["ece"],
            "mce": raw_val_summary["mce"],
            "nll": raw_val_summary["nll"],
            "brier_score": raw_val_summary["brier_score"],
        },
        "validation_after": {
            "accuracy": calibrated_val_summary["accuracy"],
            "macro_f1": calibrated_val_summary["macro_f1"],
            "ece": calibrated_val_summary["ece"],
            "mce": calibrated_val_summary["mce"],
            "nll": calibrated_val_summary["nll"],
            "brier_score": calibrated_val_summary["brier_score"],
        },
        "test_before": {
            "accuracy": raw_test_summary["accuracy"],
            "macro_f1": raw_test_summary["macro_f1"],
            "ece": raw_test_summary["ece"],
            "mce": raw_test_summary["mce"],
            "nll": raw_test_summary["nll"],
            "brier_score": raw_test_summary["brier_score"],
        },
        "test_after": {
            "accuracy": calibrated_test_summary["accuracy"],
            "macro_f1": calibrated_test_summary["macro_f1"],
            "ece": calibrated_test_summary["ece"],
            "mce": calibrated_test_summary["mce"],
            "nll": calibrated_test_summary["nll"],
            "brier_score": calibrated_test_summary["brier_score"],
        },
        "argmax_preserved": True,
        "uncertainty_thresholds": thresholds,
        "test_uncertainty_level_counts": level_counts,
        "uncertainty_by_correctness_raw": summarize_uncertainty_by_correctness(raw_test_probabilities, test_labels),
        "uncertainty_by_correctness_calibrated": summarize_uncertainty_by_correctness(
            calibrated_test_probabilities,
            test_labels,
        ),
        "error_detection_auroc": {
            "entropy": error_detection_auroc(calibrated_test_probabilities, test_labels, score_kind="entropy"),
            "one_minus_calibrated_confidence": error_detection_auroc(
                calibrated_test_probabilities,
                test_labels,
                score_kind="one_minus_confidence",
            ),
        },
        "high_confidence_error_changes": high_confidence_error_changes,
        "reliability_artifacts": {
            "before_json": make_project_relative_path(RELIABILITY_BEFORE_JSON_PATH),
            "after_json": make_project_relative_path(RELIABILITY_AFTER_JSON_PATH),
            "before_csv": make_project_relative_path(RELIABILITY_BEFORE_CSV_PATH),
            "after_csv": make_project_relative_path(RELIABILITY_AFTER_CSV_PATH),
            "before_png": make_project_relative_path(RELIABILITY_BEFORE_PNG_PATH),
            "after_png": make_project_relative_path(RELIABILITY_AFTER_PNG_PATH),
        },
        "temperature_artifact": make_project_relative_path(TEMPERATURE_ARTIFACT_PATH),
        "generated_timestamp_utc": timestamp,
    }
    _save_json(summary_payload, SUMMARY_ARTIFACT_PATH)

    print("calibration_summary=")
    print(f"  checkpoint={resolve_project_path(CHECKPOINT_PATH)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  validation_sample_count={len(val_logits_bundle['relative_paths'])}")
    print(f"  learned_temperature={float(scaler.temperature.item()):.6f}")
    print(f"  val_ece_before={raw_val_summary['ece']:.6f}")
    print(f"  val_ece_after={calibrated_val_summary['ece']:.6f}")
    print(f"  test_ece_before={raw_test_summary['ece']:.6f}")
    print(f"  test_ece_after={calibrated_test_summary['ece']:.6f}")
    print(f"  temperature_json={resolve_project_path(TEMPERATURE_ARTIFACT_PATH)}")
    print(f"  summary_json={resolve_project_path(SUMMARY_ARTIFACT_PATH)}")


if __name__ == "__main__":
    main()
