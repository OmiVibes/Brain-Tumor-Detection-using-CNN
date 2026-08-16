"""Build frozen slice-index manifests from Phase 3A subject splits."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, resolve_project_path
from brainscan.segmentation.data.audit import load_nifti_volume
from brainscan.segmentation.data.slice_index import SliceIndexRecord, write_slice_index_csv


def _load_subject_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    resolved = resolve_project_path(manifest_path)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _build_rows(rows: list[dict[str, str]], modalities: list[str]) -> list[SliceIndexRecord]:
    config = load_segmentation_config("configs/segmentation.yaml")
    dataset_root = resolve_project_path(config["dataset"]["root"])
    records: list[SliceIndexRecord] = []
    for row in rows:
        subject_id = row["subject_id"]
        split = row["split"]
        mask_path = row["mask_path"]
        mask_volume, _ = load_nifti_volume(mask_path)
        binary = mask_volume > 0
        depth = int(binary.shape[2])
        for slice_index in range(depth):
            mask_slice = binary[:, :, slice_index]
            tumor_pixel_count = int(mask_slice.sum())
            records.append(
                SliceIndexRecord(
                    subject_id=subject_id,
                    split=split,
                    slice_index=slice_index,
                    contains_tumor=bool(tumor_pixel_count > 0),
                    tumor_pixel_count=tumor_pixel_count,
                    modality_paths={
                        modality: resolve_project_path(row[f"{modality}_path"]).relative_to(dataset_root).as_posix()
                        for modality in modalities
                    },
                    mask_path=resolve_project_path(mask_path).relative_to(dataset_root).as_posix(),
                )
            )
    return records


def main() -> int:
    config = load_segmentation_config("configs/segmentation.yaml")
    modalities = list(config["dataset"]["modalities"])
    subject_dir = resolve_project_path(config["dataset"]["split_manifest_dir"])
    output_dir = resolve_project_path(config["dataset"]["slice_index_dir"])
    split_subjects: dict[str, set[str]] = {}
    for split in ["train", "val", "test"]:
        subject_rows = _load_subject_manifest(subject_dir / f"{split}.csv")
        split_subjects[split] = {row["subject_id"] for row in subject_rows}
        records = _build_rows(subject_rows, modalities)
        write_slice_index_csv(output_dir / f"{split}.csv", records, modalities=modalities)
        print(
            f"built_slice_index split={split} subjects={len(split_subjects[split])} slices={len(records)} "
            f"positive={sum(1 for record in records if record.contains_tumor)}"
        )

    if split_subjects["train"] & split_subjects["val"]:
        raise ValueError("Train/val subject leakage detected in segmentation slice index build.")
    if split_subjects["train"] & split_subjects["test"]:
        raise ValueError("Train/test subject leakage detected in segmentation slice index build.")
    if split_subjects["val"] & split_subjects["test"]:
        raise ValueError("Val/test subject leakage detected in segmentation slice index build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
