from __future__ import annotations

import torch
import pytest

from brainscan.data.constants import CANONICAL_CLASS_TO_ID
from brainscan.models import (
    SUPPORTED_ARCHITECTURES,
    build_classifier,
    count_parameters,
    count_trainable_parameters,
    get_model_metadata,
    get_pretrained_weights_name,
)


@pytest.mark.parametrize(
    ("architecture", "head_name"),
    [
        ("resnet18", "fc"),
        ("densenet121", "classifier"),
        ("efficientnet_v2_s", "classifier[1]"),
        ("convnext_tiny", "classifier[2]"),
        ("resnet50", "fc"),
    ],
)
def test_supported_architectures_build_with_four_outputs(architecture: str, head_name: str) -> None:
    model = build_classifier(architecture, num_classes=4, pretrained=False)
    outputs = model(torch.randn(2, 3, 224, 224))
    metadata = get_model_metadata(
        architecture=architecture,
        num_classes=4,
        pretrained=False,
        image_size=[224, 224],
        class_mapping=dict(CANONICAL_CLASS_TO_ID),
        model=model,
    )
    assert outputs.shape == (2, 4)
    assert metadata["classifier_head"] == head_name
    assert metadata["classifier_out_features"] == 4
    assert metadata["class_mapping"] == dict(CANONICAL_CLASS_TO_ID)


def test_supported_architecture_list_contains_phase_2c_candidates() -> None:
    assert {"resnet18", "densenet121", "efficientnet_v2_s", "convnext_tiny"}.issubset(
        set(SUPPORTED_ARCHITECTURES)
    )


def test_unsupported_architecture_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_classifier("mobilenet_v3", num_classes=4, pretrained=False)


@pytest.mark.parametrize(
    "architecture",
    ["resnet18", "densenet121", "efficientnet_v2_s", "convnext_tiny", "resnet50"],
)
def test_pretrained_weights_api_returns_name(architecture: str) -> None:
    assert get_pretrained_weights_name(architecture, pretrained=True) is not None
    assert get_pretrained_weights_name(architecture, pretrained=False) is None


def test_parameter_counts_and_metadata_are_positive() -> None:
    model = build_classifier("resnet18", num_classes=4, pretrained=False)
    metadata = get_model_metadata(
        architecture="resnet18",
        num_classes=4,
        pretrained=False,
        image_size=[224, 224],
        class_mapping=dict(CANONICAL_CLASS_TO_ID),
        model=model,
    )
    assert count_parameters(model) > 0
    assert count_trainable_parameters(model) > 0
    assert metadata["estimated_model_size_mb"] > 0
