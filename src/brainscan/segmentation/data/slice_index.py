"""Slice-index helpers for BraTS multimodal 2D segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from brainscan.core.config import resolve_project_path


SLICE_INDEX_COLUMNS = (
    "subject_id",
    "split",
    "slice_index",
    "contains_tumor",
    "tumor_pixel_count",
    "t1c_path",
    "t1n_path",
    "t2f_path",
    "t2w_path",
    "mask_path",
)


@dataclass(frozen=True)
class SliceIndexRecord:
    """One axial slice reference for multimodal segmentation."""

    subject_id: str
    split: str
    slice_index: int
    contains_tumor: bool
    tumor_pixel_count: int
    modality_paths: dict[str, str]
    mask_path: str


def _assert_relative_path(path_value: str) -> None:
    candidate = Path(path_value)
    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in segmentation slice indices: {path_value}")


def slice_record_to_row(record: SliceIndexRecord, modalities: list[str]) -> dict[str, str]:
    row = {
        "subject_id": record.subject_id,
        "split": record.split,
        "slice_index": str(int(record.slice_index)),
        "contains_tumor": "1" if record.contains_tumor else "0",
        "tumor_pixel_count": str(int(record.tumor_pixel_count)),
        "mask_path": record.mask_path,
    }
    for modality in modalities:
        row[f"{modality}_path"] = record.modality_paths.get(modality, "")
    return row


def slice_record_from_row(row: dict[str, str], modalities: list[str]) -> SliceIndexRecord:
    missing = [column for column in SLICE_INDEX_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Slice index row is missing required column(s): {', '.join(missing)}")
    modality_paths = {modality: row[f"{modality}_path"] for modality in modalities}
    for path_value in [row["mask_path"], *modality_paths.values()]:
        if path_value:
            _assert_relative_path(path_value)
    return SliceIndexRecord(
        subject_id=row["subject_id"],
        split=row["split"],
        slice_index=int(row["slice_index"]),
        contains_tumor=row["contains_tumor"] == "1",
        tumor_pixel_count=int(row["tumor_pixel_count"]),
        modality_paths=modality_paths,
        mask_path=row["mask_path"],
    )


def write_slice_index_csv(
    output_path: str | Path,
    records: list[SliceIndexRecord],
    *,
    modalities: list[str],
) -> Path:
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SLICE_INDEX_COLUMNS)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(slice_record_to_row(record, modalities))
    return resolved


def load_slice_index_records(
    index_path: str | Path,
    *,
    modalities: list[str],
) -> list[SliceIndexRecord]:
    resolved = resolve_project_path(index_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Segmentation slice index not found: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [slice_record_from_row(dict(row), modalities) for row in reader]


def count_split_subjects(records: list[SliceIndexRecord]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for record in records:
        grouped.setdefault(record.split, set()).add(record.subject_id)
    return grouped
