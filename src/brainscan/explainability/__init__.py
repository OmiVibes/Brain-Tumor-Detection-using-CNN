"""Explainability utilities for BrainScanAI."""

from .gradcam import (
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

__all__ = [
    "GradCAM",
    "build_contact_sheet",
    "compute_heatmap_diagnostics",
    "create_gradcam_overlay",
    "create_heatmap_image",
    "de_normalize_image_tensor",
    "generate_gradcam_explanation",
    "load_display_image",
    "resolve_default_gradcam_target_layer",
]
