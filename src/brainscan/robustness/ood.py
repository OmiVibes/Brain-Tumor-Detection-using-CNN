"""Penultimate-feature OOD reference and scoring for BrainScanAI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from brainscan.core.config import make_project_relative_path, resolve_project_path
from brainscan.data.constants import CANONICAL_CLASS_NAMES


@dataclass(frozen=True)
class OODReference:
    feature_layer: str
    feature_dimension: int
    class_order: tuple[str, ...]
    class_centroids: np.ndarray
    diagonal_variance: np.ndarray
    threshold: float
    threshold_quantile: float
    threshold_source: str
    training_sample_count: int
    validation_sample_count: int


class ResNet18FeatureExtractor(nn.Module):
    """Extract penultimate pooled features from the same frozen ResNet18 classifier."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not hasattr(model, "conv1") or not hasattr(model, "fc"):
            raise ValueError("Expected a torchvision-style ResNet18 classifier.")
        self.model = model
        self.feature_layer_name = "avgpool"
        self.feature_dimension = int(model.fc.in_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        return features


def extract_features_from_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    expected_split: str,
) -> dict[str, object]:
    """Collect frozen penultimate-layer features from one manifest-backed split."""
    dataset = getattr(dataloader, "dataset", None)
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("Feature extraction requires a manifest-backed dataset with records.")

    observed_splits = {record.split for record in records}
    if observed_splits != {expected_split}:
        raise ValueError(
            f"Expected only '{expected_split}' records for feature extraction, found {sorted(observed_splits)}."
        )

    extractor = ResNet18FeatureExtractor(model).to(device)
    extractor.eval()
    model.eval()

    features_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    relative_paths: list[str] = []

    for batch in dataloader:
        if len(batch) != 3:
            raise ValueError("Feature extraction dataloader must yield image, label, metadata tuples.")
        inputs, labels, metadata = batch
        inputs = inputs.to(device, non_blocking=True)
        with torch.inference_mode():
            features = extractor(inputs)
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Extracted features contain NaN or Inf values.")
        features_rows.append(features.detach().cpu())
        labels_rows.append(labels.detach().cpu())
        relative_paths.extend(list(metadata["relative_path"]))

    return {
        "features": torch.cat(features_rows, dim=0),
        "labels": torch.cat(labels_rows, dim=0).to(dtype=torch.int64),
        "relative_paths": relative_paths,
        "feature_layer": extractor.feature_layer_name,
        "feature_dimension": extractor.feature_dimension,
        "split": expected_split,
    }


def build_diagonal_mahalanobis_reference(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    *,
    threshold_quantile: float = 0.95,
) -> tuple[OODReference, np.ndarray]:
    """Construct a class-conditional diagonal Mahalanobis reference and validation threshold."""
    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("Expected 2D feature matrices for OOD reference construction.")
    if train_labels.ndim != 1:
        raise ValueError("Expected 1D train labels for OOD reference construction.")

    feature_dimension = int(train_features.shape[1])
    centroids: list[np.ndarray] = []
    diagonal_variances: list[np.ndarray] = []

    train_feature_np = train_features.cpu().numpy().astype(np.float64)
    train_label_np = train_labels.cpu().numpy().astype(np.int64)
    validation_feature_np = validation_features.cpu().numpy().astype(np.float64)

    for class_id, _class_name in enumerate(CANONICAL_CLASS_NAMES):
        class_mask = train_label_np == class_id
        class_features = train_feature_np[class_mask]
        if class_features.size == 0:
            raise ValueError(f"No training features found for class id {class_id}.")
        centroid = class_features.mean(axis=0)
        variance = class_features.var(axis=0) + 1e-6
        centroids.append(centroid)
        diagonal_variances.append(variance)

    centroid_array = np.stack(centroids, axis=0)
    variance_array = np.stack(diagonal_variances, axis=0)

    validation_scores = compute_min_diagonal_mahalanobis_scores(
        validation_feature_np,
        centroid_array,
        variance_array,
    )
    threshold = float(np.quantile(validation_scores, threshold_quantile))

    reference = OODReference(
        feature_layer="avgpool",
        feature_dimension=feature_dimension,
        class_order=tuple(CANONICAL_CLASS_NAMES),
        class_centroids=centroid_array,
        diagonal_variance=variance_array,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        threshold_source=f"validation {int(threshold_quantile * 100)}th percentile of min-class distance",
        training_sample_count=int(train_features.shape[0]),
        validation_sample_count=int(validation_features.shape[0]),
    )
    return reference, validation_scores


def compute_min_diagonal_mahalanobis_scores(
    features: np.ndarray,
    class_centroids: np.ndarray,
    diagonal_variance: np.ndarray,
) -> np.ndarray:
    """Score each feature vector by its minimum class-conditional diagonal Mahalanobis distance."""
    if features.ndim != 2:
        raise ValueError("OOD scoring expects a 2D feature array.")

    all_scores: list[float] = []
    for feature in features:
        class_scores = []
        for centroid, variance in zip(class_centroids, diagonal_variance):
            delta = feature - centroid
            score = float(np.sqrt(np.sum((delta**2) / variance)))
            class_scores.append(score)
        all_scores.append(float(min(class_scores)))
    return np.asarray(all_scores, dtype=np.float64)


def save_ood_reference(
    reference: OODReference,
    *,
    npz_path: str | Path,
    json_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_epoch: int,
    dataset_fingerprint: str,
    architecture: str,
    class_mapping: dict[str, int],
    generated_timestamp_utc: str | None = None,
) -> tuple[Path, Path]:
    """Persist the numerical OOD reference and a portable metadata JSON."""
    resolved_npz = resolve_project_path(npz_path)
    resolved_json = resolve_project_path(json_path)
    resolved_npz.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        resolved_npz,
        class_centroids=reference.class_centroids,
        diagonal_variance=reference.diagonal_variance,
    )

    metadata = {
        "feature_layer": reference.feature_layer,
        "feature_dimension": reference.feature_dimension,
        "ood_method": "class_conditional_diagonal_mahalanobis",
        "threshold": reference.threshold,
        "threshold_quantile": reference.threshold_quantile,
        "threshold_derivation": reference.threshold_source,
        "dataset_fingerprint": dataset_fingerprint,
        "training_sample_count": reference.training_sample_count,
        "validation_sample_count": reference.validation_sample_count,
        "architecture": architecture,
        "checkpoint": make_project_relative_path(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "class_mapping": class_mapping,
        "class_order": list(reference.class_order),
        "reference_npz": make_project_relative_path(resolved_npz),
        "generated_timestamp_utc": generated_timestamp_utc or datetime.now(UTC).isoformat(),
    }
    with resolved_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return resolved_npz, resolved_json


def load_ood_reference(npz_path: str | Path, json_path: str | Path) -> tuple[OODReference, dict[str, object]]:
    """Load a persisted OOD reference artifact."""
    resolved_npz = resolve_project_path(npz_path)
    resolved_json = resolve_project_path(json_path)
    arrays = np.load(resolved_npz)
    metadata = json.loads(resolved_json.read_text(encoding="utf-8"))

    reference = OODReference(
        feature_layer=str(metadata["feature_layer"]),
        feature_dimension=int(metadata["feature_dimension"]),
        class_order=tuple(CANONICAL_CLASS_NAMES),
        class_centroids=arrays["class_centroids"],
        diagonal_variance=arrays["diagonal_variance"],
        threshold=float(metadata["threshold"]),
        threshold_quantile=float(metadata["threshold_quantile"]),
        threshold_source=str(metadata["threshold_derivation"]),
        training_sample_count=int(metadata["training_sample_count"]),
        validation_sample_count=int(metadata["validation_sample_count"]),
    )
    return reference, metadata
