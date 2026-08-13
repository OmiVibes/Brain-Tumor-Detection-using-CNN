"""Reusable held-out classification evaluation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import csv
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch import Tensor, nn
from torch.utils.data import DataLoader

from brainscan.core.config import resolve_project_path
from brainscan.data.constants import CANONICAL_CLASS_NAMES, CANONICAL_CLASS_TO_ID, ID_TO_CANONICAL_CLASS
from brainscan.training.train_classifier import get_git_commit


PROBABILITY_COLUMN_BY_CLASS = {
    "glioma": "probability_glioma",
    "meningioma": "probability_meningioma",
    "pituitary": "probability_pituitary",
    "no_tumor": "probability_no_tumor",
}
CONFIDENCE_MARGIN_THRESHOLD = 0.10


@dataclass(frozen=True)
class NearDuplicateAnnotation:
    near_duplicate_to_train: bool
    near_duplicate_to_val: bool
    minimum_phash_distance: int | None


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_checkpoint_metadata(
    checkpoint: dict[str, object],
    *,
    expected_dataset_fingerprint: str,
    expected_architecture: str = "resnet18",
    expected_num_classes: int = 4,
    expected_epoch: int = 9,
    expected_image_size: Sequence[int] = (224, 224),
    expected_class_mapping: dict[str, int] | None = None,
    expected_validation_score: float | None = None,
    validation_tolerance: float = 1e-9,
) -> dict[str, object]:
    """Validate that the frozen checkpoint matches the expected Phase 1D baseline."""
    metadata = checkpoint["metadata"]
    assert isinstance(metadata, dict)
    expected_mapping = expected_class_mapping or dict(CANONICAL_CLASS_TO_ID)

    if metadata["architecture"] != expected_architecture:
        raise ValueError(
            f"Checkpoint architecture mismatch: expected '{expected_architecture}', found '{metadata['architecture']}'"
        )
    if int(metadata["num_classes"]) != expected_num_classes:
        raise ValueError(
            f"Checkpoint num_classes mismatch: expected {expected_num_classes}, found {metadata['num_classes']}"
        )
    if metadata["canonical_class_mapping"] != expected_mapping:
        raise ValueError("Checkpoint class mapping does not match canonical constants.")
    if metadata["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise ValueError("Checkpoint dataset fingerprint does not match the audited fingerprint.")
    if list(metadata["image_size"]) != list(expected_image_size):
        raise ValueError(
            f"Checkpoint image size mismatch: expected {list(expected_image_size)}, found {metadata['image_size']}"
        )
    if int(checkpoint["epoch"]) != expected_epoch:
        raise ValueError(f"Checkpoint epoch mismatch: expected {expected_epoch}, found {checkpoint['epoch']}")
    if int(metadata["epoch"]) != expected_epoch:
        raise ValueError(
            f"Checkpoint metadata epoch mismatch: expected {expected_epoch}, found {metadata['epoch']}"
        )
    if expected_validation_score is not None:
        actual_score = float(metadata["validation_score"])
        if abs(actual_score - expected_validation_score) > validation_tolerance:
            raise ValueError(
                "Checkpoint validation score does not match the training history best validation macro F1."
            )
    return metadata


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    class_names: Sequence[str] = CANONICAL_CLASS_NAMES,
) -> dict[str, object]:
    """Run reusable inference on a dataloader and collect logits, probabilities, and metadata."""
    model.eval()
    labels_expected = list(range(len(class_names)))

    paths: list[str] = []
    true_ids: list[int] = []
    predicted_ids: list[int] = []
    probabilities: list[list[float]] = []
    logits_rows: list[list[float]] = []

    for batch in dataloader:
        if len(batch) != 3:
            raise ValueError("Evaluation dataloader must yield image, label, metadata tuples.")

        inputs, labels, metadata = batch
        if model.training:
            raise ValueError("Model entered training mode during evaluation.")

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.inference_mode():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)

        if logits.ndim != 2 or logits.shape[1] != len(class_names):
            raise ValueError(f"Unexpected logits shape from evaluation model: {tuple(logits.shape)}")
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            raise ValueError("Evaluation probabilities contain NaN or Inf values.")

        predicted = probs.argmax(dim=1)
        sums = probs.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
            raise ValueError("Softmax probabilities do not sum to 1 within tolerance.")
        if predicted.min().item() < labels_expected[0] or predicted.max().item() > labels_expected[-1]:
            raise ValueError("Predicted class IDs are outside the canonical class range.")

        probabilities.extend(probs.detach().cpu().tolist())
        logits_rows.extend(logits.detach().cpu().tolist())
        predicted_ids.extend(predicted.detach().cpu().tolist())
        true_ids.extend(labels.detach().cpu().tolist())
        paths.extend(list(metadata["relative_path"]))

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate relative_path values detected in evaluation predictions.")

    return {
        "relative_paths": paths,
        "true_ids": true_ids,
        "predicted_ids": predicted_ids,
        "probabilities": probabilities,
        "logits": logits_rows,
        "class_names": list(class_names),
    }


def _compute_specificity_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    total = int(confusion.sum())
    specificity: dict[str, float] = {}
    for class_id, class_name in ID_TO_CANONICAL_CLASS.items():
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        denominator = true_negative + false_positive
        specificity[class_name] = float(true_negative / denominator) if denominator else 0.0
    return specificity


def compute_classification_metrics_bundle(
    *,
    true_ids: Sequence[int],
    predicted_ids: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str] = CANONICAL_CLASS_NAMES,
) -> dict[str, object]:
    """Compute overall, per-class, confusion, ROC-AUC, and PR-AUC metrics."""
    y_true = np.asarray(true_ids, dtype=np.int64)
    y_pred = np.asarray(predicted_ids, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    labels = list(range(len(class_names)))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = _compute_specificity_from_confusion(confusion)
    y_true_binarized = label_binarize(y_true, classes=labels)

    multiclass_roc_auc_macro: float | None = None
    roc_auc_by_class: dict[str, float | None] = {}
    pr_auc_by_class: dict[str, float | None] = {}
    roc_curves: dict[str, dict[str, list[float]]] = {}
    pr_curves: dict[str, dict[str, list[float]]] = {}

    try:
        multiclass_roc_auc_macro = float(
            roc_auc_score(y_true_binarized, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        multiclass_roc_auc_macro = None

    for class_id, class_name in enumerate(class_names):
        class_truth = y_true_binarized[:, class_id]
        class_prob = y_prob[:, class_id]
        if len(np.unique(class_truth)) < 2:
            roc_auc_by_class[class_name] = None
            pr_auc_by_class[class_name] = None
            roc_curves[class_name] = {"fpr": [], "tpr": [], "thresholds": []}
            pr_curves[class_name] = {"precision": [], "recall": [], "thresholds": []}
            continue

        fpr, tpr, roc_thresholds = roc_curve(class_truth, class_prob)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(class_truth, class_prob)
        roc_curves[class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
        pr_curves[class_name] = {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
        roc_auc_by_class[class_name] = float(roc_auc_score(class_truth, class_prob))
        pr_auc_by_class[class_name] = float(average_precision_score(class_truth, class_prob))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics: dict[str, dict[str, float | int | None]] = {}
    for class_id, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(per_class_precision[class_id]),
            "recall": float(per_class_recall[class_id]),
            "f1": float(per_class_f1[class_id]),
            "support": int(per_class_support[class_id]),
            "specificity": float(specificity[class_name]),
            "roc_auc_ovr": roc_auc_by_class[class_name],
            "pr_auc_average_precision": pr_auc_by_class[class_name],
        }

    overall_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro_ovr": multiclass_roc_auc_macro,
    }

    return {
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "classification_report": report_dict,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
    }


def build_predictions_rows(
    *,
    relative_paths: Sequence[str],
    true_ids: Sequence[int],
    predicted_ids: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str] = CANONICAL_CLASS_NAMES,
) -> list[dict[str, object]]:
    """Build one structured prediction row per test sample."""
    rows: list[dict[str, object]] = []
    for path, true_id, predicted_id, probability_row in zip(relative_paths, true_ids, predicted_ids, probabilities):
        probability_array = np.asarray(probability_row, dtype=np.float64)
        sorted_probabilities = np.sort(probability_array)
        top_probability = float(probability_array[predicted_id])
        top2_margin = float(sorted_probabilities[-1] - sorted_probabilities[-2]) if probability_array.size > 1 else 1.0

        row = {
            "relative_path": path,
            "true_class": class_names[true_id],
            "true_class_id": int(true_id),
            "predicted_class": class_names[predicted_id],
            "predicted_class_id": int(predicted_id),
            "correct": bool(true_id == predicted_id),
            "confidence": top_probability,
            "top2_margin": top2_margin,
        }
        for class_index, class_name in enumerate(class_names):
            row[PROBABILITY_COLUMN_BY_CLASS[class_name]] = float(probability_array[class_index])
        rows.append(row)
    return rows


def create_error_rows(prediction_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Return only incorrect predictions sorted by highest model confidence first."""
    error_rows: list[dict[str, object]] = []
    for row in prediction_rows:
        if bool(row["correct"]):
            continue
        error_rows.append(
            {
                "relative_path": row["relative_path"],
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "model_confidence": row["confidence"],
                "top2_margin": row["top2_margin"],
                **{
                    column_name: row[column_name]
                    for column_name in PROBABILITY_COLUMN_BY_CLASS.values()
                },
            }
        )
    return sorted(error_rows, key=lambda row: (-float(row["model_confidence"]), float(row["top2_margin"])))


def build_confusion_pairs(prediction_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Count confusion pairs for incorrect predictions."""
    counts = Counter(
        (str(row["true_class"]), str(row["predicted_class"]))
        for row in prediction_rows
        if not bool(row["correct"])
    )
    pairs = [
        {"true_class": true_class, "predicted_class": predicted_class, "count": count}
        for (true_class, predicted_class), count in counts.items()
    ]
    return sorted(pairs, key=lambda row: (-int(row["count"]), row["true_class"], row["predicted_class"]))


def compute_confidence_statistics(prediction_rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    """Summarize raw softmax confidence behavior without claiming calibration."""
    correct_confidences = [float(row["confidence"]) for row in prediction_rows if bool(row["correct"])]
    incorrect_confidences = [float(row["confidence"]) for row in prediction_rows if not bool(row["correct"])]
    low_margin_count = sum(float(row["top2_margin"]) < CONFIDENCE_MARGIN_THRESHOLD for row in prediction_rows)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _median(values: list[float]) -> float:
        return float(np.median(values)) if values else 0.0

    return {
        "mean_confidence_correct": _mean(correct_confidences),
        "mean_confidence_incorrect": _mean(incorrect_confidences),
        "median_confidence_correct": _median(correct_confidences),
        "median_confidence_incorrect": _median(incorrect_confidences),
        "incorrect_confidence_ge_090": sum(value >= 0.90 for value in incorrect_confidences),
        "incorrect_confidence_ge_095": sum(value >= 0.95 for value in incorrect_confidences),
        "predictions_top1_top2_margin_lt_010": low_margin_count,
    }


def load_near_duplicate_annotations(
    report_path: str | Path,
    *,
    test_relative_paths: Iterable[str],
) -> dict[str, NearDuplicateAnnotation]:
    """Load cross-split near-duplicate annotations for test images from the Phase 1B report."""
    resolved_path = resolve_project_path(report_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Near-duplicate report not found: {resolved_path}")

    test_path_set = set(test_relative_paths)
    annotations: dict[str, dict[str, object]] = {
        path: {
            "near_duplicate_to_train": False,
            "near_duplicate_to_val": False,
            "minimum_phash_distance": None,
        }
        for path in test_path_set
    }

    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_a = row["image_a"]
            image_b = row["image_b"]
            split_a = row["split_a"]
            split_b = row["split_b"]
            hash_distance = int(row["hash_distance"])

            if image_a in test_path_set and split_a == "test":
                target_path = image_a
                counterpart_split = split_b
            elif image_b in test_path_set and split_b == "test":
                target_path = image_b
                counterpart_split = split_a
            else:
                continue

            if counterpart_split not in {"train", "val"}:
                continue

            entry = annotations[target_path]
            if counterpart_split == "train":
                entry["near_duplicate_to_train"] = True
            if counterpart_split == "val":
                entry["near_duplicate_to_val"] = True

            minimum = entry["minimum_phash_distance"]
            if minimum is None or hash_distance < minimum:
                entry["minimum_phash_distance"] = hash_distance

    return {
        path: NearDuplicateAnnotation(
            near_duplicate_to_train=bool(values["near_duplicate_to_train"]),
            near_duplicate_to_val=bool(values["near_duplicate_to_val"]),
            minimum_phash_distance=values["minimum_phash_distance"],  # type: ignore[arg-type]
        )
        for path, values in annotations.items()
    }


def annotate_predictions_with_near_duplicates(
    prediction_rows: Sequence[dict[str, object]],
    annotations: dict[str, NearDuplicateAnnotation],
) -> list[dict[str, object]]:
    """Attach near-duplicate analysis flags to prediction rows."""
    enriched_rows: list[dict[str, object]] = []
    for row in prediction_rows:
        annotation = annotations.get(
            str(row["relative_path"]),
            NearDuplicateAnnotation(False, False, None),
        )
        enriched = dict(row)
        enriched["near_duplicate_to_train"] = annotation.near_duplicate_to_train
        enriched["near_duplicate_to_val"] = annotation.near_duplicate_to_val
        enriched["minimum_phash_distance"] = annotation.minimum_phash_distance
        enriched_rows.append(enriched)
    return enriched_rows


def summarize_filtered_subset(
    prediction_rows: Sequence[dict[str, object]],
    *,
    class_names: Sequence[str] = CANONICAL_CLASS_NAMES,
) -> dict[str, object]:
    """Compute a near-duplicate-filtered sensitivity subset summary."""
    subset_rows = [
        row
        for row in prediction_rows
        if not bool(row.get("near_duplicate_to_train")) and not bool(row.get("near_duplicate_to_val"))
    ]
    summary: dict[str, object] = {"subset_size": len(subset_rows), "metrics": None}
    unique_true_classes = {int(row["true_class_id"]) for row in subset_rows}
    if len(subset_rows) < 2 or len(unique_true_classes) < 2:
        return summary

    summary["metrics"] = compute_classification_metrics_bundle(
        true_ids=[int(row["true_class_id"]) for row in subset_rows],
        predicted_ids=[int(row["predicted_class_id"]) for row in subset_rows],
        probabilities=[
            [float(row[PROBABILITY_COLUMN_BY_CLASS[class_name]]) for class_name in class_names]
            for row in subset_rows
        ],
        class_names=class_names,
    )
    return summary


def benchmark_model_inference(
    model: nn.Module,
    sample_inputs: Tensor,
    device: torch.device,
    *,
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
) -> dict[str, float | int | str]:
    """Benchmark pure model inference latency excluding DataLoader I/O."""
    model.eval()
    sample_inputs = sample_inputs.to(device, non_blocking=True)

    _synchronize_if_cuda(device)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(sample_inputs)
        _synchronize_if_cuda(device)

        timings_ms: list[float] = []
        for _ in range(timed_iterations):
            start = time.perf_counter()
            _ = model(sample_inputs)
            _synchronize_if_cuda(device)
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000.0)

    mean_batch_latency_ms = float(np.mean(timings_ms))
    mean_per_image_latency_ms = mean_batch_latency_ms / max(sample_inputs.shape[0], 1)
    throughput_images_per_second = 1000.0 * sample_inputs.shape[0] / mean_batch_latency_ms
    return {
        "device_type": device.type,
        "batch_size": int(sample_inputs.shape[0]),
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "mean_batch_latency_ms": mean_batch_latency_ms,
        "mean_per_image_latency_ms": mean_per_image_latency_ms,
        "throughput_images_per_second": float(throughput_images_per_second),
    }


def export_predictions_csv(rows: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    """Write structured prediction rows to CSV."""
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot export an empty prediction table.")

    fieldnames = list(rows[0].keys())
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved_path


def _save_json(payload: object, output_path: str | Path) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_path


def _plot_and_save(
    figure_path: str | Path,
    *,
    title: str,
    plotter,
) -> Path:
    resolved_path = resolve_project_path(figure_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    plotter()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(resolved_path)
    plt.close()
    return resolved_path


def plot_confusion_matrix_figure(
    confusion_values: Sequence[Sequence[int]],
    output_path: str | Path,
    *,
    class_names: Sequence[str] = CANONICAL_CLASS_NAMES,
) -> Path:
    confusion_array = np.asarray(confusion_values, dtype=np.int64)

    def _plot() -> None:
        plt.imshow(confusion_array, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        threshold = confusion_array.max() / 2.0 if confusion_array.size else 0
        for row_index in range(confusion_array.shape[0]):
            for column_index in range(confusion_array.shape[1]):
                plt.text(
                    column_index,
                    row_index,
                    str(confusion_array[row_index, column_index]),
                    horizontalalignment="center",
                    color="white" if confusion_array[row_index, column_index] > threshold else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    return _plot_and_save(output_path, title="BrainScanAI Test Confusion Matrix", plotter=_plot)


def plot_roc_curves(
    roc_curves: dict[str, dict[str, list[float]]],
    roc_auc_by_class: dict[str, float | None],
    output_path: str | Path,
) -> Path:
    def _plot() -> None:
        for class_name, values in roc_curves.items():
            if not values["fpr"]:
                continue
            auc_value = roc_auc_by_class[class_name]
            label = f"{class_name} (AUC={auc_value:.4f})" if auc_value is not None else class_name
            plt.plot(values["fpr"], values["tpr"], label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

    return _plot_and_save(output_path, title="BrainScanAI One-vs-Rest ROC Curves", plotter=_plot)


def plot_pr_curves(
    pr_curves: dict[str, dict[str, list[float]]],
    pr_auc_by_class: dict[str, float | None],
    output_path: str | Path,
) -> Path:
    def _plot() -> None:
        for class_name, values in pr_curves.items():
            if not values["precision"]:
                continue
            auc_value = pr_auc_by_class[class_name]
            label = f"{class_name} (AP={auc_value:.4f})" if auc_value is not None else class_name
            plt.plot(values["recall"], values["precision"], label=label)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

    return _plot_and_save(output_path, title="BrainScanAI One-vs-Rest Precision-Recall Curves", plotter=_plot)


def build_evaluation_summary_payload(
    *,
    checkpoint_path: str | Path,
    checkpoint: dict[str, object],
    metrics_bundle: dict[str, object],
    prediction_rows: Sequence[dict[str, object]],
    performance_metrics: dict[str, object],
    dataset_fingerprint: str,
    device_info: dict[str, object],
) -> dict[str, object]:
    """Build the main structured metrics payload."""
    return {
        "checkpoint": {
            "path": str(checkpoint_path).replace("\\", "/"),
            "epoch": int(checkpoint["epoch"]),
            "architecture": checkpoint["metadata"]["architecture"],
            "validation_score": float(checkpoint["metadata"]["validation_score"]),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "test_sample_count": len(prediction_rows),
        "class_mapping": dict(CANONICAL_CLASS_TO_ID),
        "overall_metrics": metrics_bundle["overall_metrics"],
        "per_class_metrics": metrics_bundle["per_class_metrics"],
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
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
    }


def write_evaluation_artifacts(
    *,
    output_dir: str | Path,
    prediction_rows: Sequence[dict[str, object]],
    error_rows: Sequence[dict[str, object]],
    confusion_pairs: Sequence[dict[str, object]],
    metrics_payload: dict[str, object],
    metrics_bundle: dict[str, object],
) -> dict[str, Path]:
    """Write structured evaluation artifacts to disk."""
    resolved_dir = resolve_project_path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = export_predictions_csv(prediction_rows, resolved_dir / "predictions.csv")
    errors_path = export_predictions_csv(error_rows, resolved_dir / "errors.csv") if error_rows else resolved_dir / "errors.csv"
    if not error_rows:
        with errors_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "relative_path",
                    "true_class",
                    "predicted_class",
                    "model_confidence",
                    *PROBABILITY_COLUMN_BY_CLASS.values(),
                ]
            )
    confusion_pairs_path = export_predictions_csv(confusion_pairs, resolved_dir / "confusion_pairs.csv") if confusion_pairs else resolved_dir / "confusion_pairs.csv"
    if not confusion_pairs:
        with confusion_pairs_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_class", "predicted_class", "count"])

    metrics_path = _save_json(metrics_payload, resolved_dir / "metrics.json")
    report_path = _save_json(metrics_bundle["classification_report"], resolved_dir / "classification_report.json")
    confusion_json_path = _save_json(metrics_bundle["confusion_matrix"], resolved_dir / "confusion_matrix.json")
    classification_text_path = resolved_dir / "classification_report.txt"
    with classification_text_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics_bundle["classification_report"], indent=2))

    return {
        "predictions_csv": predictions_path,
        "errors_csv": errors_path,
        "confusion_pairs_csv": confusion_pairs_path,
        "metrics_json": metrics_path,
        "classification_report_json": report_path,
        "classification_report_txt": classification_text_path,
        "confusion_matrix_json": confusion_json_path,
    }
