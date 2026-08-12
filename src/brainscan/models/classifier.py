"""Classifier builders for BrainScanAI."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


SUPPORTED_ARCHITECTURES = ("resnet18",)


def get_pretrained_weights_name(architecture: str, pretrained: bool) -> str | None:
    architecture = architecture.lower()
    if architecture != "resnet18":
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported architectures: {SUPPORTED_ARCHITECTURES}"
        )

    if not pretrained:
        return None

    return str(ResNet18_Weights.DEFAULT)


def build_classifier(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a classifier with a replaceable output head."""
    architecture = architecture.lower()
    if architecture != "resnet18":
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported architectures: {SUPPORTED_ARCHITECTURES}"
        )

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that will receive gradient updates."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
