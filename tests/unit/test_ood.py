from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from brainscan.models import build_classifier
from brainscan.robustness.ood import (
    OODReference,
    ResNet18FeatureExtractor,
    build_diagonal_mahalanobis_reference,
    compute_min_diagonal_mahalanobis_scores,
    extract_features_from_dataloader,
    save_ood_reference,
)


class DummyFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, split: str) -> None:
        self.records = [type("Record", (), {"split": split})() for _ in range(4)]
        self._images = torch.rand(4, 3, 224, 224)
        self._labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        return self._images[index], int(self._labels[index]), {"relative_path": f"dataset/{index}.jpg"}


def test_feature_extractor_output_shape() -> None:
    model = build_classifier("resnet18", 4, pretrained=False)
    extractor = ResNet18FeatureExtractor(model)
    features = extractor(torch.rand(2, 3, 224, 224))
    assert tuple(features.shape) == (2, 512)


def test_feature_extractor_uses_same_resnet18() -> None:
    model = build_classifier("resnet18", 4, pretrained=False)
    extractor = ResNet18FeatureExtractor(model)
    assert extractor.model is model


def test_classifier_weights_unchanged_during_feature_extraction() -> None:
    model = build_classifier("resnet18", 4, pretrained=False)
    before = deepcopy(model.state_dict())
    loader = DataLoader(DummyFeatureDataset("train"), batch_size=2, shuffle=False)
    extract_features_from_dataloader(model, loader, torch.device("cpu"), expected_split="train")
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_ood_reference_built_only_from_train() -> None:
    model = build_classifier("resnet18", 4, pretrained=False)
    loader = DataLoader(DummyFeatureDataset("test"), batch_size=2, shuffle=False)
    try:
        extract_features_from_dataloader(model, loader, torch.device("cpu"), expected_split="train")
    except ValueError as exc:
        assert "Expected only 'train' records" in str(exc)
    else:
        raise AssertionError("Expected split mismatch error.")


def test_threshold_derived_without_test_data() -> None:
    train_features = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [15.0, 15.0],
            [15.1, 15.1],
        ],
        dtype=torch.float32,
    )
    train_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    validation_features = train_features.clone()
    reference, _ = build_diagonal_mahalanobis_reference(train_features, train_labels, validation_features)
    assert reference.threshold > 0.0
    assert "validation" in reference.threshold_source


def test_ood_score_finite() -> None:
    centroids = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    variance = np.ones_like(centroids)
    features = np.array([[0.1, 0.2], [3.0, 3.0]], dtype=np.float64)
    scores = compute_min_diagonal_mahalanobis_scores(features, centroids, variance)
    assert np.isfinite(scores).all()


def test_noise_gets_larger_ood_score_than_typical_training_sample() -> None:
    centroids = np.array([[0.0, 0.0]], dtype=np.float64)
    variance = np.array([[1.0, 1.0]], dtype=np.float64)
    typical = np.array([[0.1, 0.1]], dtype=np.float64)
    noise = np.array([[10.0, 10.0]], dtype=np.float64)
    typical_score = compute_min_diagonal_mahalanobis_scores(typical, centroids, variance)[0]
    noise_score = compute_min_diagonal_mahalanobis_scores(noise, centroids, variance)[0]
    assert noise_score > typical_score


def test_ood_metadata_uses_portable_relative_paths(tmp_path: Path) -> None:
    reference = OODReference(
        feature_layer="avgpool",
        feature_dimension=2,
        class_order=("glioma", "meningioma", "pituitary", "no_tumor"),
        class_centroids=np.zeros((4, 2), dtype=np.float64),
        diagonal_variance=np.ones((4, 2), dtype=np.float64),
        warning_threshold=1.11,
        threshold=1.23,
        warning_threshold_quantile=0.90,
        threshold_quantile=0.95,
        warning_threshold_source="validation 90th percentile",
        threshold_source="validation 95th percentile",
        training_sample_count=10,
        validation_sample_count=4,
    )
    artifact_dir = Path("artifacts/test_tmp") / f"ood-metadata-{uuid.uuid4().hex}"
    npz_path = artifact_dir / "reference.npz"
    json_path = artifact_dir / "reference.json"
    checkpoint_path = Path("artifacts/models/resnet18_baseline_best.pt")
    try:
        _, metadata_path = save_ood_reference(
            reference,
            npz_path=npz_path,
            json_path=json_path,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=9,
            dataset_fingerprint="abc123",
            architecture="resnet18",
            class_mapping={"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3},
            generated_timestamp_utc="2026-08-13T00:00:00+00:00",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["checkpoint"] == "artifacts/models/resnet18_baseline_best.pt"
        assert metadata["reference_npz"].endswith("reference.npz")
        assert metadata["warning_threshold"] == 1.11
        assert not Path(str(metadata["checkpoint"])).is_absolute()
        assert not Path(str(metadata["reference_npz"])).is_absolute()
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
