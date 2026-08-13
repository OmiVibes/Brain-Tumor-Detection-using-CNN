from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path("src").resolve()))

from brainscan.explainability.gradcam import (  # noqa: E402
    GradCAM,
    build_contact_sheet,
    compute_heatmap_diagnostics,
    create_gradcam_overlay,
    create_heatmap_image,
    de_normalize_image_tensor,
    generate_gradcam_explanation,
    load_display_image,
    resolve_default_gradcam_target_layer,
)
from brainscan.models import build_classifier  # noqa: E402


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = self.features(inputs)
        pooled = self.pool(activations).flatten(1)
        return self.classifier(pooled)


def _checkpoint_metadata() -> dict[str, object]:
    return {
        "epoch": 9,
        "architecture": "resnet18",
        "dataset_fingerprint": "abc123",
        "image_size": [224, 224],
    }


def test_target_layer_resolves_correctly_for_resnet18() -> None:
    model = build_classifier("resnet18", 4, pretrained=False)
    target_layer, target_layer_name = resolve_default_gradcam_target_layer(model, "resnet18")
    assert target_layer_name == "layer4[-1].conv2"
    assert isinstance(target_layer, nn.Conv2d)


def test_gradcam_output_shape_and_normalization() -> None:
    model = TinyConvNet()
    target_layer = model.features[2]
    engine = GradCAM(model, target_layer, "features[2]")
    result = engine.generate(torch.randn(1, 3, 16, 16), target_class=2)
    engine.remove_hooks()
    assert result.heatmap.shape == (16, 16)
    assert np.isfinite(result.heatmap).all()
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0


def test_invalid_target_class_rejected() -> None:
    model = TinyConvNet()
    engine = GradCAM(model, model.features[2], "features[2]")
    with pytest.raises(ValueError, match="Invalid target class id"):
        engine.generate(torch.randn(1, 3, 16, 16), target_class=10)
    engine.remove_hooks()


def test_predicted_class_default_and_gradients_captured() -> None:
    model = TinyConvNet()
    engine = GradCAM(model, model.features[2], "features[2]")
    result = engine.generate(torch.randn(1, 3, 16, 16), target_class=None)
    assert engine.gradients is not None
    assert engine.activations is not None
    assert 0 <= result.predicted_class_id < 4
    engine.remove_hooks()


def test_model_parameters_unchanged_and_eval_preserved(tmp_path) -> None:
    model = TinyConvNet()
    before = deepcopy(model.state_dict())
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), color=(120, 120, 120)).save(image_path)
    input_tensor = torch.rand(1, 3, 24, 24)
    bundle = generate_gradcam_explanation(
        model=model,
        image_tensor=input_tensor,
        relative_path=str(image_path),
        checkpoint_metadata=_checkpoint_metadata(),
        architecture="resnet18",
        target_layer=model.features[2],
        target_layer_name="features[2]",
    )
    after = model.state_dict()
    assert model.training is False
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert bundle["metadata"]["checkpoint_epoch"] == 9


def test_hooks_removed_after_use() -> None:
    model = TinyConvNet()
    engine = GradCAM(model, model.features[2], "features[2]")
    engine.remove_hooks()
    assert engine._forward_handle.id not in model.features[2]._forward_hooks
    assert engine._backward_handle.id not in model.features[2]._backward_hooks


def test_overlay_dimensions_and_display_image_loading(tmp_path) -> None:
    image_path = tmp_path / "display.png"
    Image.new("RGB", (32, 24), color=(80, 90, 100)).save(image_path)
    image = load_display_image(str(image_path))
    overlay = create_gradcam_overlay(image, np.ones((12, 12), dtype=np.float32) * 0.5)
    assert image.size == (32, 24)
    assert overlay.size == image.size


def test_de_normalization_and_heatmap_image_generation() -> None:
    tensor = torch.zeros(3, 8, 8)
    de_normalized = de_normalize_image_tensor(tensor)
    assert de_normalized.shape == tensor.shape
    heatmap_image = create_heatmap_image(np.ones((8, 8), dtype=np.float32) * 0.5)
    assert heatmap_image.size == (8, 8)


def test_target_class_maps_differ_on_controlled_example(tmp_path) -> None:
    model = TinyConvNet()
    image_path = tmp_path / "controlled.png"
    Image.new("RGB", (24, 24), color=(150, 140, 130)).save(image_path)
    tensor = torch.rand(1, 3, 24, 24)
    bundle_a = generate_gradcam_explanation(
        model=model,
        image_tensor=tensor,
        relative_path=str(image_path),
        checkpoint_metadata=_checkpoint_metadata(),
        target_class=0,
        architecture="resnet18",
        target_layer=model.features[2],
        target_layer_name="features[2]",
    )
    bundle_b = generate_gradcam_explanation(
        model=model,
        image_tensor=tensor,
        relative_path=str(image_path),
        checkpoint_metadata=_checkpoint_metadata(),
        target_class=1,
        architecture="resnet18",
        target_layer=model.features[2],
        target_layer_name="features[2]",
    )
    difference = np.abs(bundle_a["resized_heatmap"] - bundle_b["resized_heatmap"]).mean()
    assert difference > 0.0


def test_no_surrogate_model_instantiated(monkeypatch, tmp_path) -> None:
    called = {"count": 0}

    def _unexpected(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("Unexpected surrogate model construction")

    monkeypatch.setattr("brainscan.models.classifier.build_classifier", _unexpected, raising=False)
    model = TinyConvNet()
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), color=(110, 100, 90)).save(image_path)
    generate_gradcam_explanation(
        model=model,
        image_tensor=torch.rand(1, 3, 24, 24),
        relative_path=str(image_path),
        checkpoint_metadata=_checkpoint_metadata(),
        architecture="resnet18",
        target_layer=model.features[2],
        target_layer_name="features[2]",
    )
    assert called["count"] == 0


def test_contact_sheet_and_diagnostics(tmp_path) -> None:
    image = Image.new("RGB", (20, 20), color=(120, 130, 140))
    row = {
        "original_image": image,
        "heatmap_image": image,
        "overlay_image": image,
    }
    path = build_contact_sheet([row], tmp_path / "grid.png", thumbnail_size=(16, 16))
    assert path.exists()
    diagnostics = compute_heatmap_diagnostics(np.ones((10, 10), dtype=np.float32))
    assert 0.0 <= diagnostics["border_attention_ratio"] <= 1.0
