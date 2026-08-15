"""Model builders for BrainScanAI."""

from .classifier import (
    SUPPORTED_ARCHITECTURES,
    build_classifier,
    count_parameters,
    count_trainable_parameters,
    estimate_model_size_mb,
    get_classifier_head_description,
    get_model_metadata,
    get_pretrained_weights_name,
)
from .introspection import (
    ArchitectureFeatureExtractor,
    SUPPORTED_INTROSPECTION_ARCHITECTURES,
    extract_penultimate_features,
    resolve_feature_layer_spec,
    resolve_gradcam_target_layer,
)

__all__ = [
    "ArchitectureFeatureExtractor",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_INTROSPECTION_ARCHITECTURES",
    "build_classifier",
    "count_parameters",
    "count_trainable_parameters",
    "estimate_model_size_mb",
    "extract_penultimate_features",
    "get_classifier_head_description",
    "get_model_metadata",
    "get_pretrained_weights_name",
    "resolve_feature_layer_spec",
    "resolve_gradcam_target_layer",
]
