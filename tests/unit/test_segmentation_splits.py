from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import nibabel as nib
import numpy as np

from brainscan.segmentation.data.audit import audit_brats_subjects, discover_brats_subjects
from brainscan.segmentation.data.splits import (
    create_subject_level_manifests,
    manifests_have_disjoint_subjects,
    write_segmentation_manifest_csv,
)


def _workspace_tmp_dir() -> Path:
    path = Path(".tmp") / "segmentation_unit" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_nifti(path: Path, array: np.ndarray) -> None:
    image = nib.Nifti1Image(array, affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _build_dataset(root: Path, subject_count: int = 10) -> list[dict[str, object]]:
    for index in range(subject_count):
        subject_id = f"BraTS-GLI-{index:05d}-000"
        subject_dir = root / subject_id
        image = np.ones((4, 4, 3), dtype=np.float32) * index
        mask = np.zeros((4, 4, 3), dtype=np.uint8)
        mask[1:3, 1:3, 1] = 1
        for modality in ("t1c", "t1n", "t2f", "t2w"):
            _write_nifti(subject_dir / f"{subject_id}-{modality}.nii.gz", image)
        _write_nifti(subject_dir / f"{subject_id}-seg.nii.gz", mask)

    subjects = discover_brats_subjects(
        root,
        ["t1c", "t1n", "t2f", "t2w"],
        "seg",
        source_dataset="brats_adult_glioma",
        source_version="2023",
    )
    return audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])


def test_subject_ids_deterministic() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        rows = _build_dataset(tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma", subject_count=4)
        assert [row["subject_id"] for row in rows] == sorted(row["subject_id"] for row in rows)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_subject_level_splits_mutually_exclusive() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        rows = _build_dataset(tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma", subject_count=8)
        manifests = create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        assert manifests_have_disjoint_subjects(manifests) is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_no_subject_crosses_splits() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        rows = _build_dataset(tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma", subject_count=9)
        manifests = create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        membership: dict[str, str] = {}
        for split_name, split_rows in manifests.items():
            for row in split_rows:
                assert row.subject_id not in membership
                membership[row.subject_id] = split_name
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_same_seed_reproduces_manifests() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        rows = _build_dataset(tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma", subject_count=7)
        manifests_a = create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        manifests_b = create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        assert manifests_a == manifests_b
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_manifest_paths_portable() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        rows = _build_dataset(tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma", subject_count=5)
        manifests = create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        for split_rows in manifests.values():
            for row in split_rows:
                assert not Path(row.mask_path).is_absolute()
                assert all(not Path(path_value).is_absolute() for path_value in row.modality_paths.values() if path_value)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_absolute_paths_rejected() -> None:
    rows = [
        {
            "subject_id": "BraTS-GLI-00000-000",
            "source_dataset": "brats_adult_glioma",
            "source_version": "2023",
            "modality_paths": {"t1c": "C:/absolute/file.nii.gz", "t1n": "", "t2f": "", "t2w": ""},
            "mask_path": "relative-mask.nii.gz",
        }
    ]
    try:
        create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        assert False, "Expected absolute path rejection."
    except ValueError as exc:
        assert "Absolute paths" in str(exc)


def test_manifest_csv_is_portable() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        rows = _build_dataset(tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma", subject_count=6)
        manifests = create_subject_level_manifests(rows, modalities=["t1c", "t1n", "t2f", "t2w"], seed=42)
        output_path = tmp_dir / "data" / "segmentation" / "splits" / "train.csv"
        write_segmentation_manifest_csv(output_path, manifests["train"], modalities=["t1c", "t1n", "t2f", "t2w"])
        content = output_path.read_text(encoding="utf-8")
        assert "C:\\" not in content
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
