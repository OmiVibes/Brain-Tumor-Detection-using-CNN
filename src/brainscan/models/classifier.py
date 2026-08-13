"""Classifier builders for BrainScanAI."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_v2_s,
    resnet18,
    resnet50,
)


SUPPORTED_ARCHITECTURES = (
    "resnet18",
    "densenet121",
    "efficientnet_v2_s",
    "convnext_tiny",
    "resnet50",
)


@dataclass(frozen=True)
class ArchitectureSpec:
    builder: object
    weights_enum: object
    classifier_attribute: str
    classifier_index: int | None = None


ARCHITECTURE_SPECS: dict[str, ArchitectureSpec] = {
    "resnet18": ArchitectureSpec(
        builder=resnet18,
        weights_enum=ResNet18_Weights,
        classifier_attribute="fc",
    ),
    "densenet121": ArchitectureSpec(
        builder=densenet121,
        weights_enum=DenseNet121_Weights,
        classifier_attribute="classifier",
    ),
    "efficientnet_v2_s": ArchitectureSpec(
        builder=efficientnet_v2_s,
        weights_enum=EfficientNet_V2_S_Weights,
        classifier_attribute="classifier",
        classifier_index=1,
    ),
    "convnext_tiny": ArchitectureSpec(
        builder=convnext_tiny,
        weights_enum=ConvNeXt_Tiny_Weights,
        classifier_attribute="classifier",
        classifier_index=2,
    ),
    "resnet50": ArchitectureSpec(
        builder=resnet50,
        weights_enum=ResNet50_Weights,
        classifier_attribute="fc",
    ),
}


def _resolve_architecture_spec(architecture: str) -> ArchitectureSpec:
    normalized = architecture.lower()
    if normalized not in ARCHITECTURE_SPECS:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported architectures: {SUPPORTED_ARCHITECTURES}"
        )
    return ARCHITECTURE_SPECS[normalized]


def _replace_classifier_head(model: nn.Module, *, spec: ArchitectureSpec, num_classes: int) -> None:
    if spec.classifier_index is None:
        classifier = getattr(model, spec.classifier_attribute)
        if not isinstance(classifier, nn.Linear):
            raise TypeError(
                f"Expected Linear head at '{spec.classifier_attribute}' for model '{type(model).__name__}'."
            )
        setattr(model, spec.classifier_attribute, nn.Linear(classifier.in_features, num_classes))
        return

    classifier = getattr(model, spec.classifier_attribute)
    if not isinstance(classifier, nn.Sequential):
        raise TypeError(
            f"Expected Sequential head at '{spec.classifier_attribute}' for model '{type(model).__name__}'."
        )
    original_head = classifier[spec.classifier_index]
    if not isinstance(original_head, nn.Linear):
        raise TypeError(
            f"Expected Linear layer at '{spec.classifier_attribute}[{spec.classifier_index}]' "
            f"for model '{type(model).__name__}'."
        )
    classifier[spec.classifier_index] = nn.Linear(original_head.in_features, num_classes)


def _get_classifier_head(model: nn.Module, *, spec: ArchitectureSpec) -> nn.Linear:
    if spec.classifier_index is None:
        head = getattr(model, spec.classifier_attribute)
    else:
        classifier = getattr(model, spec.classifier_attribute)
        if not isinstance(classifier, nn.Sequential):
            raise TypeError(
                f"Expected Sequential head at '{spec.classifier_attribute}' for model '{type(model).__name__}'."
            )
        head = classifier[spec.classifier_index]
    if not isinstance(head, nn.Linear):
        raise TypeError("Classifier head is not a Linear layer after replacement.")
    return head


def get_pretrained_weights_name(architecture: str, pretrained: bool) -> str | None:
    spec = _resolve_architecture_spec(architecture)
    if not pretrained:
        return None

    return str(spec.weights_enum.DEFAULT)


def build_classifier(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a classifier with a replaceable output head."""
    spec = _resolve_architecture_spec(architecture)
    weights = spec.weights_enum.DEFAULT if pretrained else None
    model = spec.builder(weights=weights)
    _replace_classifier_head(model, spec=spec, num_classes=num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that will receive gradient updates."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_parameters(model: nn.Module) -> int:
    """Count all parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate raw parameter size in megabytes assuming float32 checkpoints."""
    return round((count_parameters(model) * 4) / (1024**2), 2)


def get_classifier_head_description(architecture: str) -> str:
    spec = _resolve_architecture_spec(architecture)
    if spec.classifier_index is None:
        return spec.classifier_attribute
    return f"{spec.classifier_attribute}[{spec.classifier_index}]"


def get_model_metadata(
    *,
    architecture: str,
    num_classes: int,
    pretrained: bool,
    image_size: list[int] | tuple[int, int],
    class_mapping: dict[str, int],
    model: nn.Module | None = None,
) -> dict[str, object]:
    """Return portable architecture metadata for comparison artifacts and checkpoints."""
    model_instance = model or build_classifier(architecture, num_classes=num_classes, pretrained=pretrained)
    spec = _resolve_architecture_spec(architecture)
    head = _get_classifier_head(model_instance, spec=spec)
    return {
        "architecture": architecture.lower(),
        "pretrained_weights": get_pretrained_weights_name(architecture, pretrained),
        "parameter_count": count_parameters(model_instance),
        "trainable_parameter_count": count_trainable_parameters(model_instance),
        "estimated_model_size_mb": estimate_model_size_mb(model_instance),
        "input_size": [int(image_size[0]), int(image_size[1])],
        "class_mapping": dict(class_mapping),
        "classifier_head": get_classifier_head_description(architecture),
        "classifier_out_features": int(head.out_features),
    }
