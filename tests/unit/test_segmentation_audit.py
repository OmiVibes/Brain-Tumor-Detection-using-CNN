from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import nibabel as nib
import numpy as np

from brainscan.segmentation.data.audit import (
    audit_brats_subjects,
    build_segmentation_dataset_fingerprint,
    compute_file_sha256,
    discover_brats_subjects,
    load_nifti_volume,
)


def _workspace_tmp_dir() -> Path:
    path = Path(".tmp") / "segmentation_unit" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_nifti(path: Path, array: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    image = nib.Nifti1Image(array, affine=affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _build_subject(root: Path, subject_id: str, *, mismatch_mask: bool = False, empty_mask: bool = False) -> None:
    subject_dir = root / subject_id
    base_image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    if not empty_mask:
        mask[1:3, 1:3, 1] = np.array([[1, 2], [3, 3]], dtype=np.uint8)
    if mismatch_mask:
        mask = np.zeros((5, 4, 3), dtype=np.uint8)
    for modality in ("t1c", "t1n", "t2f", "t2w"):
        _write_nifti(subject_dir / f"{subject_id}-{modality}.nii.gz", base_image + len(modality))
    _write_nifti(subject_dir / f"{subject_id}-seg.nii.gz", mask)


def test_nifti_loads() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        file_path = tmp_dir / "sample.nii.gz"
        original = np.ones((4, 4, 4), dtype=np.float32)
        _write_nifti(file_path, original)
        loaded, _image = load_nifti_volume(file_path)
        assert loaded.shape == original.shape
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_image_mask_shape_match_validation() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00000-000")
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        rows = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        assert rows[0]["shape_mismatch"] is False
        assert rows[0]["mask_readable"] is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_mismatch_detected() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00001-000", mismatch_mask=True)
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        rows = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        assert rows[0]["shape_mismatch"] is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_mask_labels_extracted() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00002-000")
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        rows = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        assert rows[0]["mask_label_values"] == [0, 1, 2, 3]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_empty_mask_detected() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00003-000", empty_mask=True)
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        rows = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        assert rows[0]["empty_mask"] is True
        assert rows[0]["tumor_voxel_count"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_tumor_voxel_count_correct() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00004-000")
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        rows = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        assert rows[0]["tumor_voxel_count"] == 4
        assert rows[0]["positive_slices"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dataset_fingerprint_deterministic() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00005-000")
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        rows = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        fp_a = build_segmentation_dataset_fingerprint(
            rows,
            label_mapping={"background": 0, "ncr_net": 1, "edema": 2, "enhancing_tumor": 3},
            dataset_name="brats_adult_glioma",
            dataset_version="2023",
        )
        fp_b = build_segmentation_dataset_fingerprint(
            rows,
            label_mapping={"background": 0, "ncr_net": 1, "edema": 2, "enhancing_tumor": 3},
            dataset_name="brats_adult_glioma",
            dataset_version="2023",
        )
        assert fp_a["dataset_fingerprint"] == fp_b["dataset_fingerprint"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_source_files_never_modified() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        dataset_root = tmp_dir / "data" / "segmentation" / "raw" / "brats2023_glioma"
        _build_subject(dataset_root, "BraTS-GLI-00006-000")
        target_file = dataset_root / "BraTS-GLI-00006-000" / "BraTS-GLI-00006-000-seg.nii.gz"
        before = compute_file_sha256(target_file)
        subjects = discover_brats_subjects(
            dataset_root,
            ["t1c", "t1n", "t2f", "t2w"],
            "seg",
            source_dataset="brats_adult_glioma",
            source_version="2023",
        )
        _ = audit_brats_subjects(subjects, ["t1c", "t1n", "t2f", "t2w"])
        after = compute_file_sha256(target_file)
        assert before == after
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
