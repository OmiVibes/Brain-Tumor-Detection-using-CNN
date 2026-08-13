"""Scientifically correct Grad-CAM for the BrainScanAI ResNet18 baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor, nn

from brainscan.core.config import resolve_project_path
from brainscan.data.constants import CANONICAL_CLASS_NAMES, ID_TO_CANONICAL_CLASS
from brainscan.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class GradCAMResult:
    """Container for one Grad-CAM explanation."""

    heatmap: np.ndarray
    predicted_class_id: int
    predicted_class: str
    model_confidence: float
    target_class_id: int
    target_layer: str
    probabilities: list[float]
    logits: list[float]


class GradCAM:
    """Reusable Grad-CAM engine bound to a passed model instance."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.target_layer = target_layer
        self.target_layer_name = target_layer_name
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output: Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: Tensor, target_class: int | None = None) -> GradCAMResult:
        """Generate a normalized 0..1 Grad-CAM heatmap for one input tensor."""
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"Grad-CAM expects an input tensor shaped [1, C, H, W], got {tuple(input_tensor.shape)}"
            )

        self.model.eval()
        original_parameters = [parameter.detach().clone() for parameter in self.model.parameters()]

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)
        if logits.ndim != 2:
            raise ValueError(f"Grad-CAM expected 2D logits, got {tuple(logits.shape)}")

        probabilities = torch.softmax(logits, dim=1)
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
            raise ValueError("Grad-CAM probabilities contain NaN or Inf values.")

        predicted_class_id = int(probabilities.argmax(dim=1).item())
        target_class_id = predicted_class_id if target_class is None else int(target_class)

        if target_class_id < 0 or target_class_id >= logits.shape[1]:
            raise ValueError(f"Invalid target class id {target_class_id} for logits shape {tuple(logits.shape)}")

        target_score = logits[:, target_class_id].sum()
        target_score.backward(retain_graph=False)

        if self.activations is None:
            raise RuntimeError("Grad-CAM target activations were not captured.")
        if self.gradients is None:
            raise RuntimeError("Grad-CAM target gradients were not captured.")

        gradients = self.gradients
        activations = self.activations
        if gradients.ndim != 4 or activations.ndim != 4:
            raise RuntimeError(
                f"Unexpected activation/gradient shapes for Grad-CAM: "
                f"{tuple(activations.shape)} and {tuple(gradients.shape)}"
            )

        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (channel_weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_2d = cam[0, 0]
        if torch.isnan(cam_2d).any() or torch.isinf(cam_2d).any():
            raise RuntimeError("Grad-CAM heatmap contains NaN or Inf values.")

        heatmap = cam_2d.detach().cpu().numpy().astype(np.float32)
        if float(np.max(heatmap)) <= 0.0:
            normalized_heatmap = np.zeros_like(heatmap, dtype=np.float32)
        else:
            heatmap_min = float(np.min(heatmap))
            heatmap_max = float(np.max(heatmap))
            denominator = heatmap_max - heatmap_min
            if denominator <= 0.0:
                normalized_heatmap = np.zeros_like(heatmap, dtype=np.float32)
            else:
                normalized_heatmap = (heatmap - heatmap_min) / denominator

        for current, original in zip(self.model.parameters(), original_parameters):
            if not torch.equal(current.detach(), original):
                raise RuntimeError("Model parameters changed during Grad-CAM generation.")

        self.model.eval()

        return GradCAMResult(
            heatmap=normalized_heatmap,
            predicted_class_id=predicted_class_id,
            predicted_class=ID_TO_CANONICAL_CLASS[predicted_class_id],
            model_confidence=float(probabilities[0, predicted_class_id].item()),
            target_class_id=target_class_id,
            target_layer=self.target_layer_name,
            probabilities=probabilities[0].detach().cpu().tolist(),
            logits=logits[0].detach().cpu().tolist(),
        )


def resolve_default_gradcam_target_layer(model: nn.Module, architecture: str = "resnet18") -> tuple[nn.Module, str]:
    """Resolve the verified default Grad-CAM target layer for the supported model."""
    if architecture.lower() != "resnet18":
        raise ValueError(f"Unsupported Grad-CAM architecture '{architecture}'.")

    if not hasattr(model, "layer4"):
        raise ValueError("ResNet18 model is missing layer4; cannot resolve Grad-CAM target layer.")

    target_layer = model.layer4[-1].conv2
    if not isinstance(target_layer, nn.Conv2d):
        raise ValueError("Resolved Grad-CAM target layer is not a convolutional layer.")
    return target_layer, "layer4[-1].conv2"


def de_normalize_image_tensor(image_tensor: Tensor) -> Tensor:
    """Invert ImageNet normalization for display."""
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor for de-normalization, got {tuple(image_tensor.shape)}")

    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    de_normalized = image_tensor * std + mean
    return de_normalized.clamp(0.0, 1.0)


def load_display_image(relative_path: str) -> Image.Image:
    """Load the original RGB MRI slice for non-deceptive visualization."""
    resolved_path = resolve_project_path(relative_path)
    with Image.open(resolved_path) as image:
        return image.convert("RGB")


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a normalized heatmap to a display size."""
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0), mode="L")
    resized = heatmap_image.resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def create_heatmap_image(heatmap: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    """Convert a normalized heatmap into a visible RGB image."""
    display_heatmap = resize_heatmap(heatmap, size) if size is not None else heatmap
    colormap = colormaps["jet"]
    rgba = colormap(np.clip(display_heatmap, 0.0, 1.0))
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
) -> Image.Image:
    """Blend the Grad-CAM heatmap with the original MRI image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Overlay alpha must be within [0, 1], got {alpha}")
    heatmap_image = create_heatmap_image(heatmap, original_image.size)
    return Image.blend(original_image.convert("RGB"), heatmap_image, alpha=alpha)


def compute_heatmap_diagnostics(heatmap: np.ndarray, border_fraction: float = 0.10) -> dict[str, float]:
    """Compute bounded explainability diagnostics, not localization claims."""
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")
    if np.isnan(heatmap).any() or np.isinf(heatmap).any():
        raise ValueError("Heatmap diagnostics received NaN or Inf values.")

    heatmap = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    total_activation = float(heatmap.sum())
    nonzero_percentage = float(np.count_nonzero(heatmap > 0.0) / heatmap.size)

    if total_activation <= 0.0:
        center_x = 0.5
        center_y = 0.5
        border_attention_ratio = 0.0
    else:
        y_indices, x_indices = np.indices(heatmap.shape)
        center_x = float((heatmap * x_indices).sum() / total_activation / max(heatmap.shape[1] - 1, 1))
        center_y = float((heatmap * y_indices).sum() / total_activation / max(heatmap.shape[0] - 1, 1))
        border_y = max(1, int(round(heatmap.shape[0] * border_fraction)))
        border_x = max(1, int(round(heatmap.shape[1] * border_fraction)))
        mask = np.zeros_like(heatmap, dtype=bool)
        mask[:border_y, :] = True
        mask[-border_y:, :] = True
        mask[:, :border_x] = True
        mask[:, -border_x:] = True
        border_attention_ratio = float(heatmap[mask].sum() / total_activation)

    return {
        "heatmap_min": float(heatmap.min()),
        "heatmap_max": float(heatmap.max()),
        "heatmap_mean": float(heatmap.mean()),
        "nonzero_activation_percentage": nonzero_percentage,
        "center_of_mass_x": center_x,
        "center_of_mass_y": center_y,
        "border_attention_ratio": border_attention_ratio,
    }


def save_explanation_metadata(metadata: dict[str, object], path: str | Path) -> Path:
    """Persist one explanation metadata JSON."""
    resolved_path = resolve_project_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return resolved_path


def generate_gradcam_explanation(
    *,
    model: nn.Module,
    image_tensor: Tensor,
    relative_path: str,
    checkpoint_metadata: dict[str, object],
    target_class: int | None = None,
    architecture: str = "resnet18",
    target_layer: nn.Module | None = None,
    target_layer_name: str | None = None,
) -> dict[str, object]:
    """Generate a full Grad-CAM explanation bundle from the frozen prediction model."""
    resolved_layer, resolved_layer_name = (
        (target_layer, target_layer_name)
        if target_layer is not None and target_layer_name is not None
        else resolve_default_gradcam_target_layer(model, architecture)
    )

    gradcam = GradCAM(model=model, target_layer=resolved_layer, target_layer_name=resolved_layer_name)
    try:
        result = gradcam.generate(image_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    original_image = load_display_image(relative_path)
    resized_heatmap = resize_heatmap(result.heatmap, original_image.size)
    heatmap_image = create_heatmap_image(result.heatmap, original_image.size)
    overlay_image = create_gradcam_overlay(original_image, result.heatmap)
    diagnostics = compute_heatmap_diagnostics(resized_heatmap)

    metadata = {
        "source_relative_path": relative_path.replace("\\", "/"),
        "predicted_class": result.predicted_class,
        "predicted_class_id": result.predicted_class_id,
        "model_confidence": result.model_confidence,
        "target_class": ID_TO_CANONICAL_CLASS[result.target_class_id],
        "target_class_id": result.target_class_id,
        "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
        "checkpoint_architecture": checkpoint_metadata["architecture"],
        "target_layer": result.target_layer,
        "dataset_fingerprint": checkpoint_metadata["dataset_fingerprint"],
        "explanation_method": "Grad-CAM",
        "image_size": list(checkpoint_metadata["image_size"]),
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "probabilities": {
            class_name: float(result.probabilities[class_id])
            for class_id, class_name in ID_TO_CANONICAL_CLASS.items()
        },
        "diagnostics": diagnostics,
    }

    return {
        "result": result,
        "original_image": original_image,
        "heatmap_image": heatmap_image,
        "overlay_image": overlay_image,
        "resized_heatmap": resized_heatmap,
        "metadata": metadata,
    }


def build_contact_sheet(
    rows: Sequence[dict[str, object]],
    output_path: str | Path,
    *,
    thumbnail_size: tuple[int, int] = (180, 180),
) -> Path:
    """Build a small Original | Heatmap | Overlay review grid."""
    if not rows:
        raise ValueError("Contact sheet requires at least one row.")

    width, height = thumbnail_size
    columns = ("Original", "Heatmap", "Overlay")
    canvas_width = width * 3
    header_height = 28
    caption_height = 56
    row_height = header_height + height + caption_height
    canvas_height = row_height * len(rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    for row_index, row in enumerate(rows):
        y_offset = row_index * row_height
        original = ImageOps.fit(row["original_image"].convert("RGB"), thumbnail_size)
        heatmap = ImageOps.fit(row["heatmap_image"].convert("RGB"), thumbnail_size)
        overlay = ImageOps.fit(row["overlay_image"].convert("RGB"), thumbnail_size)
        for column_index, image in enumerate((original, heatmap, overlay)):
            canvas.paste(image, (column_index * width, y_offset + header_height))

    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_path)
    return resolved_path
