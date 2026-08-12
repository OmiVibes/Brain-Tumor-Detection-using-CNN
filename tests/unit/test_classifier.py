from __future__ import annotations

import torch
import pytest

from brainscan.data.constants import CANONICAL_CLASS_TO_ID
from brainscan.models.classifier import build_classifier, count_trainable_parameters, get_pretrained_weights_name
from brainscan.training.train_classifier import build_checkpoint_metadata


def test_resnet18_builds_successfully() -> None:
    model = build_classifier("resnet18", num_classes=4, pretrained=False)
    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (2, 4)


def test_unsupported_architecture_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_classifier("efficientnet", num_classes=4, pretrained=False)


def test_final_classifier_layer_has_four_outputs() -> None:
    model = build_classifier("resnet18", num_classes=4, pretrained=False)
    assert model.fc.out_features == 4
    assert count_trainable_parameters(model) > 0


def test_checkpoint_metadata_contains_canonical_mapping() -> None:
    config = {
        "model": {"architecture": "resnet18", "num_classes": 4, "pretrained": True},
        "training": {"image_size": [224, 224], "seed": 42},
    }
    fingerprint = {"dataset_fingerprint": "abc123"}
    metadata = build_checkpoint_metadata(
        config=config,
        weights_name=get_pretrained_weights_name("resnet18", pretrained=True),
        dataset_fingerprint=fingerprint,
    )
    assert metadata["canonical_class_mapping"] == dict(CANONICAL_CLASS_TO_ID)
    assert metadata["pretrained_weights"] is not None
