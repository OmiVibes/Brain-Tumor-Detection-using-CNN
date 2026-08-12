"""Core utilities for BrainScanAI."""

from .config import (
    get_project_root,
    load_config,
    load_inference_config,
    load_train_config,
    resolve_project_path,
)

__all__ = [
    "get_project_root",
    "load_config",
    "load_inference_config",
    "load_train_config",
    "resolve_project_path",
]
