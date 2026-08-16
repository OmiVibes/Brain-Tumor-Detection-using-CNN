"""Deterministic subject-level split helpers for segmentation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random


@dataclass(frozen=True)
class SegmentationManifestRow:
    subject_id: str
    split: str
    source_dataset: str
    source_version: str
    modality_paths: dict[str, str]
    mask_path: str


def choose_subject_split_strategy(subject_count: int) -> dict[str, object]:
    """Choose a simple deterministic subject-level split strategy."""
    if subject_count <= 0:
        raise ValueError("Subject count must be positive.")
    if subject_count >= 50:
        return {
            "strategy": "subject_level_70_15_15",
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "justification": "Enough subjects exist to reserve larger held-out validation and test cohorts.",
        }
    return {
        "strategy": "subject_level_80_10_10",
        "fractions": {"train": 0.80, "val": 0.10, "test": 0.10},
        "justification": "Smaller cohorts benefit from keeping more subjects in training while preserving held-out splits.",
    }


def _ensure_relative_portable(path_value: str) -> None:
    candidate = Path(path_value)
    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in segmentation manifests: {path_value}")


def _allocate_subject_counts(subject_count: int, fractions: dict[str, float]) -> dict[str, int]:
    split_order = ("train", "val", "test")
    counts = {split_name: int(round(subject_count * fractions[split_name])) for split_name in split_order}
    delta = subject_count - sum(counts.values())
    counts["train"] += delta

    if subject_count >= 3:
        for split_name in ("val", "test"):
            if counts[split_name] == 0:
                donor = max(split_order, key=lambda name: counts[name])
                if counts[donor] <= 1:
                    continue
                counts[donor] -= 1
                counts[split_name] += 1
    return counts


def create_subject_level_manifests(
    audited_rows: list[dict[str, object]],
    *,
    modalities: list[str],
    seed: int,
    fractions: dict[str, float] | None = None,
) -> dict[str, list[SegmentationManifestRow]]:
    """Create subject-level train/val/test manifests from audited rows."""
    if not audited_rows:
        raise ValueError("Cannot create manifests from an empty audited dataset.")

    chosen = choose_subject_split_strategy(len(audited_rows))
    split_fractions = fractions or chosen["fractions"]
    subject_ids = sorted(str(row["subject_id"]) for row in audited_rows)
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Duplicate subject IDs detected in segmentation audit rows.")

    shuffled = subject_ids[:]
    random.Random(seed).shuffle(shuffled)
    split_counts = _allocate_subject_counts(len(shuffled), split_fractions)

    split_subjects: dict[str, set[str]] = {}
    cursor = 0
    for split_name in ("train", "val", "test"):
        count = split_counts[split_name]
        split_subjects[split_name] = set(shuffled[cursor : cursor + count])
        cursor += count

    manifests: dict[str, list[SegmentationManifestRow]] = {"train": [], "val": [], "test": []}
    for row in sorted(audited_rows, key=lambda item: str(item["subject_id"])):
        subject_id = str(row["subject_id"])
        modality_paths = row["modality_paths"]
        assert isinstance(modality_paths, dict)
        mask_path = str(row["mask_path"])
        _ensure_relative_portable(mask_path)
        for modality in modalities:
            path_value = str(modality_paths.get(modality, ""))
            if path_value:
                _ensure_relative_portable(path_value)

        split_name = next(name for name, subjects in split_subjects.items() if subject_id in subjects)
        manifests[split_name].append(
            SegmentationManifestRow(
                subject_id=subject_id,
                split=split_name,
                source_dataset=str(row["source_dataset"]),
                source_version=str(row["source_version"]),
                modality_paths={modality: str(modality_paths.get(modality, "")) for modality in modalities},
                mask_path=mask_path,
            )
        )
    return manifests


def manifests_have_disjoint_subjects(manifests: dict[str, list[SegmentationManifestRow]]) -> bool:
    seen: set[str] = set()
    for rows in manifests.values():
        current = {row.subject_id for row in rows}
        if seen.intersection(current):
            return False
        seen.update(current)
    return True


def write_segmentation_manifest_csv(
    output_path: str | Path,
    rows: list[SegmentationManifestRow],
    *,
    modalities: list[str],
) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "split", "source_dataset", "source_version"] + [
        f"{modality}_path" for modality in modalities
    ] + ["mask_path"]
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "subject_id": row.subject_id,
                "split": row.split,
                "source_dataset": row.source_dataset,
                "source_version": row.source_version,
                "mask_path": row.mask_path,
            }
            for modality in modalities:
                record[f"{modality}_path"] = row.modality_paths.get(modality, "")
            writer.writerow(record)
    return resolved_path
