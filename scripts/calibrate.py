"""Validation-only temperature scaling and uncertainty analysis for BrainScanAI."""

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

import matplotlib.pyplot as plt
import torch

from brainscan.core import load_best_validation_reference
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


DEFAULT_CONFIG_PATH = Path("configs/train.yaml")
DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
DEFAULT_REFERENCE_METRICS_PATH = Path("artifacts/training/resnet18_baseline_history.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/calibration/resnet18_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BrainScanAI calibration and uncertainty analysis.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Training-style config path.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Frozen checkpoint path.")
    parser.add_argument(
        "--reference-metrics-path",
        default=str(DEFAULT_REFERENCE_METRICS_PATH),
        help="History JSON or summary JSON used to verify best epoch and validation macro F1.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for calibration artifacts.")
    parser.add_argument("--prefer-cuda", action="store_true", help="Prefer CUDA when available.")
    return parser.parse_args()


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


def _build_metrics_summary(logits: torch.Tensor, labels: torch.Tensor, *, n_bins: int = 15) -> dict[str, object]:
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
    return build_predictions_rows(
        relative_paths=relative_paths,
        true_ids=labels.cpu().tolist(),
        predicted_ids=probabilities.argmax(dim=1).cpu().tolist(),
        probabilities=probabilities.cpu().tolist(),
        class_names=CANONICAL_CLASS_NAMES,
    )


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


def _choose_default_probability_mode(
    raw_val_summary: dict[str, object],
    calibrated_val_summary: dict[str, object],
) -> tuple[str, str]:
    calibrated_better_or_equal = (
        float(calibrated_val_summary["ece"]) <= float(raw_val_summary["ece"])
        and float(calibrated_val_summary["nll"]) <= float(raw_val_summary["nll"])
        and float(calibrated_val_summary["brier_score"]) <= float(raw_val_summary["brier_score"])
    )
    if calibrated_better_or_equal:
        return (
            "calibrated",
            "Calibrated probabilities were kept as default because validation ECE, NLL, and Brier score all improved or stayed equal.",
        )
    return (
        "raw",
        "Raw probabilities remain the default because validation calibration quality did not improve consistently after temperature scaling.",
    )


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    reference_metrics = load_best_validation_reference(args.reference_metrics_path)

    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(args.checkpoint_path, model, map_location="cpu")
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]
    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_architecture=str(config["model"]["architecture"]),
        expected_epoch=int(reference_metrics["best_epoch"]),
        expected_validation_score=float(reference_metrics["best_validation_macro_f1"]),
    )
    checkpoint_metadata = {**checkpoint_metadata, "epoch": int(checkpoint["epoch"])}

    output_dir = Path(args.output_dir)
    temperature_artifact_path = output_dir / "temperature.json"
    summary_artifact_path = output_dir / "metrics.json"
    reliability_before_json_path = output_dir / "reliability_before.json"
    reliability_after_json_path = output_dir / "reliability_after.json"
    reliability_before_csv_path = output_dir / "reliability_before.csv"
    reliability_after_csv_path = output_dir / "reliability_after.csv"
    reliability_before_png_path = output_dir / "reliability_before.png"
    reliability_after_png_path = output_dir / "reliability_after.png"

    device, device_info = resolve_device(prefer_cuda=bool(args.prefer_cuda))
    model.to(device)
    model.eval()

    architecture = str(checkpoint_metadata["architecture"])
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

    default_probability_mode, default_probability_rationale = _choose_default_probability_mode(
        raw_val_summary,
        calibrated_val_summary,
    )
    selected_val_summary = calibrated_val_summary if default_probability_mode == "calibrated" else raw_val_summary
    selected_test_summary = calibrated_test_summary if default_probability_mode == "calibrated" else raw_test_summary

    thresholds = derive_uncertainty_thresholds(
        selected_val_summary["normalized_entropy"].cpu().tolist(),  # type: ignore[union-attr]
        selected_val_summary["top1_top2_margin"].cpu().tolist(),  # type: ignore[union-attr]
    )
    uncertainty_levels = assign_uncertainty_levels(
        selected_test_summary["normalized_entropy"].cpu().tolist(),  # type: ignore[union-attr]
        selected_test_summary["top1_top2_margin"].cpu().tolist(),  # type: ignore[union-attr]
        thresholds,
    )
    level_counts = {level: uncertainty_levels.count(level) for level in ("LOW", "MEDIUM", "HIGH")}

    raw_test_probabilities = softmax_probabilities_from_logits(test_logits)
    calibrated_test_probabilities = softmax_probabilities_from_logits(calibrated_test_logits)
    raw_prediction_rows = _extract_confidence_rows(raw_test_probabilities, test_labels, test_relative_paths)
    high_confidence_error_changes = _revisit_high_confidence_errors(raw_prediction_rows, calibrated_test_probabilities)

    before_rows = raw_val_summary["calibration_bins"]  # type: ignore[assignment]
    after_rows = calibrated_val_summary["calibration_bins"]  # type: ignore[assignment]
    _save_json(before_rows, reliability_before_json_path)
    _save_json(after_rows, reliability_after_json_path)
    _save_reliability_csv(before_rows, reliability_before_csv_path)
    _save_reliability_csv(after_rows, reliability_after_csv_path)
    title_prefix = architecture.replace("_", " ").title()
    _plot_reliability_diagram(before_rows, reliability_before_png_path, f"{title_prefix} Reliability Before Calibration")
    _plot_reliability_diagram(after_rows, reliability_after_png_path, f"{title_prefix} Reliability After Calibration")

    timestamp = datetime.now(UTC).isoformat()
    artifact_payload = build_temperature_artifact_payload(
        temperature=float(scaler.temperature.item()),
        checkpoint_path=args.checkpoint_path,
        checkpoint_epoch=int(checkpoint["epoch"]),
        dataset_fingerprint=dataset_fingerprint,
        validation_metrics_before=raw_val_summary,
        validation_metrics_after=calibrated_val_summary,
        git_commit=get_git_commit(),
        generated_timestamp_utc=timestamp,
    )
    artifact_payload["architecture"] = architecture
    artifact_payload["default_probability_mode"] = default_probability_mode
    artifact_payload["default_probability_rationale"] = default_probability_rationale
    save_temperature_artifact(artifact_payload, temperature_artifact_path)

    summary_payload = {
        "checkpoint": make_project_relative_path(args.checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": architecture,
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
        "default_probability_mode": default_probability_mode,
        "default_probability_rationale": default_probability_rationale,
        "uncertainty_threshold_source": f"{default_probability_mode} validation probabilities only",
        "uncertainty_thresholds": thresholds,
        "test_uncertainty_level_counts": level_counts,
        "uncertainty_by_correctness_raw": summarize_uncertainty_by_correctness(raw_test_probabilities, test_labels),
        "uncertainty_by_correctness_calibrated": summarize_uncertainty_by_correctness(
            calibrated_test_probabilities,
            test_labels,
        ),
        "error_detection_auroc": {
            "raw_entropy": error_detection_auroc(raw_test_probabilities, test_labels, score_kind="entropy"),
            "raw_one_minus_confidence": error_detection_auroc(
                raw_test_probabilities,
                test_labels,
                score_kind="one_minus_confidence",
            ),
            "calibrated_entropy": error_detection_auroc(
                calibrated_test_probabilities,
                test_labels,
                score_kind="entropy",
            ),
            "calibrated_one_minus_confidence": error_detection_auroc(
                calibrated_test_probabilities,
                test_labels,
                score_kind="one_minus_confidence",
            ),
        },
        "selected_error_detection_auroc": {
            "entropy": error_detection_auroc(
                calibrated_test_probabilities if default_probability_mode == "calibrated" else raw_test_probabilities,
                test_labels,
                score_kind="entropy",
            ),
            "one_minus_confidence": error_detection_auroc(
                calibrated_test_probabilities if default_probability_mode == "calibrated" else raw_test_probabilities,
                test_labels,
                score_kind="one_minus_confidence",
            ),
        },
        "high_confidence_error_changes": high_confidence_error_changes,
        "reliability_artifacts": {
            "before_json": make_project_relative_path(reliability_before_json_path),
            "after_json": make_project_relative_path(reliability_after_json_path),
            "before_csv": make_project_relative_path(reliability_before_csv_path),
            "after_csv": make_project_relative_path(reliability_after_csv_path),
            "before_png": make_project_relative_path(reliability_before_png_path),
            "after_png": make_project_relative_path(reliability_after_png_path),
        },
        "temperature_artifact": make_project_relative_path(temperature_artifact_path),
        "generated_timestamp_utc": timestamp,
    }
    _save_json(summary_payload, summary_artifact_path)

    print("calibration_summary=")
    print(f"  checkpoint={resolve_project_path(args.checkpoint_path)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  architecture={architecture}")
    print(f"  validation_sample_count={len(val_logits_bundle['relative_paths'])}")
    print(f"  learned_temperature={float(scaler.temperature.item()):.6f}")
    print(f"  default_probability_mode={default_probability_mode}")
    print(f"  val_ece_before={raw_val_summary['ece']:.6f}")
    print(f"  val_ece_after={calibrated_val_summary['ece']:.6f}")
    print(f"  test_ece_before={raw_test_summary['ece']:.6f}")
    print(f"  test_ece_after={calibrated_test_summary['ece']:.6f}")
    print(f"  temperature_json={resolve_project_path(temperature_artifact_path)}")
    print(f"  summary_json={resolve_project_path(summary_artifact_path)}")


if __name__ == "__main__":
    main()
