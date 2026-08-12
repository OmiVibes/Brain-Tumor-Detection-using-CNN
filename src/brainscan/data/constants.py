"""Canonical class mapping for BrainScanAI Phase 1A."""

from __future__ import annotations


CANONICAL_CLASS_NAMES: tuple[str, str, str, str] = (
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor",
)

# Mapping from physical folder names in the legacy dataset to the canonical
# semantic class names used by BrainScanAI.
DATASET_FOLDER_TO_CANONICAL: dict[str, str] = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
}

CANONICAL_CLASS_TO_ID: dict[str, int] = {
    class_name: class_id for class_id, class_name in enumerate(CANONICAL_CLASS_NAMES)
}

ID_TO_CANONICAL_CLASS: dict[int, str] = {
    class_id: class_name for class_name, class_id in CANONICAL_CLASS_TO_ID.items()
}

DATASET_FOLDER_TO_ID: dict[str, int] = {
    folder_name: CANONICAL_CLASS_TO_ID[canonical_name]
    for folder_name, canonical_name in DATASET_FOLDER_TO_CANONICAL.items()
}
