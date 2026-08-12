"""Manifest-backed PyTorch dataset for BrainScanAI MRI classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv

from PIL import Image, UnidentifiedImageError
import torch
from torch import Tensor
from torch.utils.data import Dataset

from brainscan.core.config import get_project_root, resolve_project_path
from brainscan.data.constants import CANONICAL_CLASS_TO_ID, DATASET_FOLDER_TO_CANONICAL


REQUIRED_MANIFEST_COLUMNS = {
    "relative_path",
    "physical_class",
    "canonical_class",
    "class_id",
    "split",
}


@dataclass(frozen=True)
class ManifestRecord:
    relative_path: str
    physical_class: str
    canonical_class: str
    class_id: int
    split: str
    sha256: str = ""
    perceptual_hash: str = ""
    subject_id: str = ""


def _validate_manifest_row(row: dict[str, str]) -> ManifestRecord:
    missing = REQUIRED_MANIFEST_COLUMNS.difference(row)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Manifest row is missing required column(s): {joined}")

    physical_class = row["physical_class"]
    canonical_class = row["canonical_class"]
    class_id = int(row["class_id"])

    if physical_class not in DATASET_FOLDER_TO_CANONICAL:
        raise ValueError(f"Unknown physical dataset class '{physical_class}' in manifest row")

    expected_canonical = DATASET_FOLDER_TO_CANONICAL[physical_class]
    if canonical_class != expected_canonical:
        raise ValueError(
            f"Manifest row canonical_class '{canonical_class}' does not match physical_class "
            f"'{physical_class}' -> '{expected_canonical}'"
        )

    expected_class_id = CANONICAL_CLASS_TO_ID[canonical_class]
    if class_id != expected_class_id:
        raise ValueError(
            f"Manifest row class_id '{class_id}' does not match canonical_class "
            f"'{canonical_class}' -> '{expected_class_id}'"
        )

    return ManifestRecord(
        relative_path=row["relative_path"],
        physical_class=physical_class,
        canonical_class=canonical_class,
        class_id=class_id,
        split=row["split"],
        sha256=row.get("sha256", ""),
        perceptual_hash=row.get("perceptual_hash", ""),
        subject_id=row.get("subject_id", ""),
    )


def load_manifest_records(manifest_path: str | Path) -> list[ManifestRecord]:
    resolved_path = resolve_project_path(manifest_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_validate_manifest_row(dict(row)) for row in reader]


class BrainMRIDataset(Dataset[tuple[Tensor, int] | tuple[Tensor, int, dict[str, str]]]):
    """Load MRI image samples strictly from manifest metadata."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        records: Iterable[ManifestRecord] | None = None,
        transform=None,
        return_metadata: bool = False,
        project_root: Path | None = None,
    ) -> None:
        if manifest_path is None and records is None:
            raise ValueError("Either manifest_path or records must be provided.")

        self.project_root = project_root or get_project_root()
        self.records = list(records) if records is not None else load_manifest_records(manifest_path)  # type: ignore[arg-type]
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, relative_path: str) -> Path:
        resolved = resolve_project_path(relative_path, self.project_root)
        if not resolved.exists():
            raise FileNotFoundError(f"Image file not found for manifest path: {relative_path}")
        return resolved

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self._resolve_image_path(record.relative_path)

        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
        except FileNotFoundError:
            raise
        except UnidentifiedImageError as exc:
            raise ValueError(f"Failed to decode image '{record.relative_path}': {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read image '{record.relative_path}': {exc}") from exc

        if self.transform is not None:
            image_tensor = self.transform(rgb_image)
        else:
            raise ValueError("A transform must be provided for BrainMRIDataset.")

        label = int(record.class_id)

        if not self.return_metadata:
            return image_tensor, label

        metadata = {
            "relative_path": record.relative_path,
            "canonical_class": record.canonical_class,
            "physical_class": record.physical_class,
            "sha256": record.sha256,
        }
        return image_tensor, label, metadata
