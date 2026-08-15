"""Helpers for loading frozen validation references from history or summary artifacts."""

from __future__ import annotations

from pathlib import Path
import json

from brainscan.core.config import resolve_project_path


def load_best_validation_reference(reference_path: str | Path) -> dict[str, float | int]:
    """Return best epoch and best validation macro F1 from a history or summary artifact."""
    resolved_path = resolve_project_path(reference_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        if "best_epoch" in payload and "best_validation_macro_f1" in payload:
            return {
                "best_epoch": int(payload["best_epoch"]),
                "best_validation_macro_f1": float(payload["best_validation_macro_f1"]),
            }
        raise ValueError(
            f"Unsupported summary payload at {resolved_path}. Expected best_epoch and best_validation_macro_f1."
        )

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Training history is empty: {resolved_path}")
        best_entry = max(payload, key=lambda row: float(row["val_macro_f1"]))
        return {
            "best_epoch": int(best_entry["epoch"]),
            "best_validation_macro_f1": float(best_entry["val_macro_f1"]),
        }

    raise TypeError(
        f"Unsupported frozen validation reference payload type at {resolved_path}: {type(payload).__name__}"
    )

