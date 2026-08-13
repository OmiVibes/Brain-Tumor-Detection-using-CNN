"""Simple YAML configuration loading for Phase 1A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH_KEY_SUFFIXES = ("path", "dir", "root")


def get_project_root() -> Path:
    """Return the repository root derived from the src layout."""
    return PROJECT_ROOT


def resolve_project_path(path_value: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a project-relative or absolute path safely."""
    root = project_root or get_project_root()
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def make_project_relative_path(path_value: str | Path, project_root: Path | None = None) -> str:
    """Return a repository-relative POSIX path for portable metadata artifacts."""
    root = (project_root or get_project_root()).resolve()
    resolved = resolve_project_path(path_value, root)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the project root and cannot be made relative: {resolved}") from exc
    return relative.as_posix()


def _resolve_path_like_values(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        resolved: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, (str, Path)) and key.lower().endswith(PATH_KEY_SUFFIXES):
                resolved[key] = str(resolve_project_path(item, project_root))
            else:
                resolved[key] = _resolve_path_like_values(item, project_root)
        return resolved
    if isinstance(value, list):
        return [_resolve_path_like_values(item, project_root) for item in value]
    return value


def _validate_required_sections(config: dict[str, Any], required_sections: tuple[str, ...], source: Path) -> None:
    missing = [section for section in required_sections if section not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config section(s) in {source}: {joined}")


def load_config(config_path: str | Path, required_sections: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    resolved_path = resolve_project_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a top-level mapping: {resolved_path}")

    if required_sections:
        _validate_required_sections(config, required_sections, resolved_path)

    return _resolve_path_like_values(config, get_project_root())


def load_train_config(config_path: str | Path = "configs/train.yaml") -> dict[str, Any]:
    """Load the Phase 1A training configuration."""
    return load_config(config_path, required_sections=("training", "dataset", "model"))


def load_inference_config(config_path: str | Path = "configs/inference.yaml") -> dict[str, Any]:
    """Load the Phase 1A inference configuration."""
    return load_config(config_path, required_sections=("dataset", "model", "inference"))
