"""Evaluation helpers for 2D whole-tumor segmentation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import statistics

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from brainscan.core.config import resolve_project_path
from brainscan.segmentation.training.losses import thresholded_binary_metrics


@dataclass(frozen=True)
class SubjectSegmentationMetrics:
    subject_id: str
    dice: float
    iou: float
    precision: float
    recall: float
    specificity: float
    tumor_voxel_count: int
    predicted_voxel_count: int
    false_positive_voxel_count: int
    false_negative_voxel_count: int


def _to_metadata_dict(batch_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    subject_ids = list(batch_metadata["subject_id"])
    slice_indices = list(batch_metadata["slice_index"])
    contains_tumor = list(batch_metadata["contains_tumor"])
    tumor_pixel_count = list(batch_metadata["tumor_pixel_count"])
    return [
        {
            "subject_id": str(subject_ids[index]),
            "slice_index": int(slice_indices[index]),
            "contains_tumor": bool(contains_tumor[index]),
            "tumor_pixel_count": int(tumor_pixel_count[index]),
        }
        for index in range(len(subject_ids))
    ]


def reconstruct_subject_volume(slice_predictions: list[tuple[int, np.ndarray]], depth: int) -> np.ndarray:
    if not slice_predictions:
        raise ValueError("Cannot reconstruct a subject volume from zero slices.")
    height, width = slice_predictions[0][1].shape
    volume = np.zeros((height, width, depth), dtype=np.uint8)
    for slice_index, mask in slice_predictions:
        volume[:, :, int(slice_index)] = mask.astype(np.uint8, copy=False)
    return volume


def compute_subject_metrics(
    prediction_volume: np.ndarray,
    target_volume: np.ndarray,
    *,
    subject_id: str,
) -> SubjectSegmentationMetrics:
    pred = prediction_volume.astype(np.uint8, copy=False)
    target = target_volume.astype(np.uint8, copy=False)
    tp = int(np.logical_and(pred == 1, target == 1).sum())
    tn = int(np.logical_and(pred == 0, target == 0).sum())
    fp = int(np.logical_and(pred == 1, target == 0).sum())
    fn = int(np.logical_and(pred == 0, target == 1).sum())
    epsilon = 1e-6
    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    return SubjectSegmentationMetrics(
        subject_id=subject_id,
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        tumor_voxel_count=int(target.sum()),
        predicted_voxel_count=int(pred.sum()),
        false_positive_voxel_count=fp,
        false_negative_voxel_count=fn,
    )


@torch.no_grad()
def evaluate_segmentation_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module | None,
    threshold: float,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    slice_metric_totals = defaultdict(float)
    slice_count = 0
    prediction_slices: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    target_slices: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets, metadata = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets) if criterion is not None else None
        probabilities = torch.sigmoid(logits)
        metrics = thresholded_binary_metrics(probabilities, targets, threshold=threshold)
        batch_size = int(inputs.shape[0])
        for key, value in metrics.items():
            slice_metric_totals[key] += float(value) * batch_size
        slice_count += batch_size
        if loss is not None:
            total_loss += float(loss.item()) * batch_size
            batch_count += batch_size

        metadata_rows = _to_metadata_dict(metadata)
        predictions = (probabilities >= threshold).to(dtype=torch.uint8).cpu().numpy()
        target_array = targets.to(dtype=torch.uint8).cpu().numpy()
        for row, pred_mask, target_mask in zip(metadata_rows, predictions, target_array, strict=False):
            subject_id = row["subject_id"]
            slice_index = int(row["slice_index"])
            prediction_slices[subject_id].append((slice_index, pred_mask[0]))
            target_slices[subject_id].append((slice_index, target_mask[0]))

    subject_metrics: list[SubjectSegmentationMetrics] = []
    for subject_id in sorted(prediction_slices):
        pred_depth = max(int(slice_index) for slice_index, _ in prediction_slices[subject_id]) + 1
        target_depth = max(int(slice_index) for slice_index, _ in target_slices[subject_id]) + 1
        pred_volume = reconstruct_subject_volume(prediction_slices[subject_id], depth=pred_depth)
        target_volume = reconstruct_subject_volume(target_slices[subject_id], depth=target_depth)
        subject_metrics.append(compute_subject_metrics(pred_volume, target_volume, subject_id=subject_id))

    if not subject_metrics:
        raise ValueError("Evaluation loader produced no subject metrics.")

    subject_dice_values = [metric.dice for metric in subject_metrics]
    subject_iou_values = [metric.iou for metric in subject_metrics]
    subject_precision_values = [metric.precision for metric in subject_metrics]
    subject_recall_values = [metric.recall for metric in subject_metrics]
    subject_specificity_values = [metric.specificity for metric in subject_metrics]

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    return {
        "loss": float(total_loss / max(batch_count, 1)) if criterion is not None else None,
        "slice_dice": float(slice_metric_totals["dice"] / max(slice_count, 1)),
        "slice_iou": float(slice_metric_totals["iou"] / max(slice_count, 1)),
        "slice_precision": float(slice_metric_totals["precision"] / max(slice_count, 1)),
        "slice_recall": float(slice_metric_totals["recall"] / max(slice_count, 1)),
        "slice_specificity": float(slice_metric_totals["specificity"] / max(slice_count, 1)),
        "subject_dice": _mean(subject_dice_values),
        "subject_iou": _mean(subject_iou_values),
        "subject_precision": _mean(subject_precision_values),
        "subject_recall": _mean(subject_recall_values),
        "subject_specificity": _mean(subject_specificity_values),
        "subject_metrics": subject_metrics,
    }


def write_subject_metrics_csv(output_path: str | Path, rows: list[SubjectSegmentationMetrics]) -> Path:
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "tumor_voxel_count",
        "predicted_voxel_count",
        "false_positive_voxel_count",
        "false_negative_voxel_count",
    ]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return resolved


def summarize_subject_metrics(rows: list[SubjectSegmentationMetrics]) -> dict[str, Any]:
    dice_values = sorted(row.dice for row in rows)
    iou_values = [row.iou for row in rows]
    precision_values = [row.precision for row in rows]
    recall_values = [row.recall for row in rows]
    specificity_values = [row.specificity for row in rows]

    def _quantile(values: list[float], q: float) -> float:
        return float(np.quantile(np.asarray(values, dtype=np.float64), q))

    return {
        "subject_count": len(rows),
        "mean_dice": float(statistics.mean(dice_values)),
        "median_dice": float(statistics.median(dice_values)),
        "dice_std": float(statistics.pstdev(dice_values)) if len(dice_values) > 1 else 0.0,
        "mean_iou": float(statistics.mean(iou_values)),
        "mean_precision": float(statistics.mean(precision_values)),
        "mean_recall": float(statistics.mean(recall_values)),
        "mean_specificity": float(statistics.mean(specificity_values)),
        "dice_distribution": {
            "min": float(min(dice_values)),
            "q25": _quantile(dice_values, 0.25),
            "median": float(statistics.median(dice_values)),
            "q75": _quantile(dice_values, 0.75),
            "max": float(max(dice_values)),
        },
    }


def write_json(output_path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def build_review_grid(
    output_path: str | Path,
    *,
    good_case: tuple[np.ndarray, np.ndarray, np.ndarray],
    median_case: tuple[np.ndarray, np.ndarray, np.ndarray],
    poor_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    tiles = [good_case, median_case, poor_case]
    tile_width = 160
    tile_height = 160
    canvas = Image.new("RGB", (tile_width * 4, tile_height * len(tiles)), color="white")
    for row_index, (image_slice, target_mask, predicted_mask) in enumerate(tiles):
        base = np.uint8(np.clip(image_slice, 0.0, 1.0) * 255.0)
        target = np.uint8(target_mask > 0) * 255
        predicted = np.uint8(predicted_mask > 0) * 255
        overlay = np.stack([base, base, base], axis=2)
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], predicted)
        overlay[:, :, 1] = np.maximum(overlay[:, :, 1], target)
        tile_images = [
            Image.fromarray(np.stack([base, base, base], axis=2)),
            Image.fromarray(np.stack([target, target, target], axis=2)),
            Image.fromarray(np.stack([predicted, predicted, predicted], axis=2)),
            Image.fromarray(overlay),
        ]
        for column_index, tile in enumerate(tile_images):
            canvas.paste(tile.resize((tile_width, tile_height)), (column_index * tile_width, row_index * tile_height))
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved)
    return resolved
