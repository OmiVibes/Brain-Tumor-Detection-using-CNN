"""Architecture-aware model introspection helpers for BrainScanAI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


SUPPORTED_INTROSPECTION_ARCHITECTURES = ("resnet18", "densenet121")


@dataclass(frozen=True)
class FeatureLayerSpec:
    layer_name: str
    feature_dimension: int


def _normalize_architecture(architecture: str) -> str:
    normalized = architecture.lower()
    if normalized not in SUPPORTED_INTROSPECTION_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}' for BrainScanAI introspection. "
            f"Supported architectures: {SUPPORTED_INTROSPECTION_ARCHITECTURES}"
        )
    return normalized


def resolve_gradcam_target_layer(model: nn.Module, architecture: str) -> tuple[nn.Module, str]:
    """Resolve the verified default Grad-CAM target layer for a supported backbone."""
    normalized = _normalize_architecture(architecture)

    if normalized == "resnet18":
        if not hasattr(model, "layer4"):
            raise ValueError("ResNet18 model is missing layer4; cannot resolve Grad-CAM target layer.")
        target_layer = model.layer4[-1].conv2
        target_layer_name = "layer4[-1].conv2"
    else:
        if not hasattr(model, "features"):
            raise ValueError("DenseNet121 model is missing features; cannot resolve Grad-CAM target layer.")
        target_layer = model.features.denseblock4.denselayer16.conv2
        target_layer_name = "features.denseblock4.denselayer16.conv2"

    if not isinstance(target_layer, nn.Conv2d):
        raise ValueError("Resolved Grad-CAM target layer is not a convolutional layer.")
    return target_layer, target_layer_name


def resolve_feature_layer_spec(model: nn.Module, architecture: str) -> FeatureLayerSpec:
    """Resolve the penultimate feature representation metadata for a supported backbone."""
    normalized = _normalize_architecture(architecture)

    if normalized == "resnet18":
        if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
            raise ValueError("ResNet18 model is missing a Linear 'fc' head.")
        return FeatureLayerSpec(layer_name="avgpool", feature_dimension=int(model.fc.in_features))

    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Linear):
        raise ValueError("DenseNet121 model is missing a Linear 'classifier' head.")
    return FeatureLayerSpec(
        layer_name="features.norm5 -> relu -> adaptive_avg_pool2d",
        feature_dimension=int(model.classifier.in_features),
    )


def extract_penultimate_features(model: nn.Module, architecture: str, inputs: torch.Tensor) -> torch.Tensor:
    """Extract the exact penultimate feature vector that feeds the classifier head."""
    normalized = _normalize_architecture(architecture)

    if normalized == "resnet18":
        x = model.conv1(inputs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    x = model.features(inputs)
    x = F.relu(x, inplace=False)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


class ArchitectureFeatureExtractor(nn.Module):
    """Extract penultimate pooled features from a supported frozen classifier."""

    def __init__(self, model: nn.Module, architecture: str) -> None:
        super().__init__()
        self.model = model
        self.architecture = _normalize_architecture(architecture)
        spec = resolve_feature_layer_spec(model, self.architecture)
        self.feature_layer_name = spec.layer_name
        self.feature_dimension = spec.feature_dimension

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return extract_penultimate_features(self.model, self.architecture, inputs)

