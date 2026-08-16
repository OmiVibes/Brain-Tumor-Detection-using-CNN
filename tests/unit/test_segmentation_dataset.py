from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import nibabel as nib
import numpy as np

from brainscan.segmentation.data.audit import compute_file_sha256
from brainscan.segmentation.data.dataset import BraTSSliceDataset, select_training_slice_records
from brainscan.segmentation.data.slice_index import SliceIndexRecord, load_slice_index_records, write_slice_index_csv


def _workspace_tmp_dir() -> Path:
    path = (Path.cwd() / ".tmp" / "segmentation_dataset_unit" / uuid.uuid4().hex).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_nifti(path: Path, array: np.ndarray) -> None:
    image = nib.Nifti1Image(array, affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _build_subject(root: Path, subject_id: str) -> tuple[dict[str, str], str]:
    subject_dir = root / subject_id
    volume = np.zeros((8, 8, 3), dtype=np.float32)
    volume[2:6, 2:6, 1] = 5.0
    mask = np.zeros((8, 8, 3), dtype=np.uint8)
    mask[2:6, 2:6, 1] = 2
    modality_paths: dict[str, str] = {}
    for channel, modality in enumerate(("t1c", "t1n", "t2f", "t2w"), start=1):
        path = subject_dir / f"{subject_id}-{modality}.nii.gz"
        _write_nifti(path, volume + channel)
        modality_paths[modality] = path.as_posix()
    mask_path = subject_dir / f"{subject_id}-seg.nii.gz"
    _write_nifti(mask_path, mask)
    return modality_paths, mask_path.as_posix()


def test_modalities_stack_into_four_channel_slice_and_binary_target() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        modality_paths, mask_path = _build_subject(tmp_dir, "BraTS-GLI-00000-000")
        records = [
            SliceIndexRecord(
                subject_id="BraTS-GLI-00000-000",
                split="train",
                slice_index=1,
                contains_tumor=True,
                tumor_pixel_count=16,
                modality_paths={key: Path(value).resolve().relative_to(Path.cwd()).as_posix() for key, value in modality_paths.items()},
                mask_path=Path(mask_path).resolve().relative_to(Path.cwd()).as_posix(),
            )
        ]
        dataset = BraTSSliceDataset(records=records, modalities=["t1c", "t1n", "t2f", "t2w"])
        image, mask = dataset[0]
        assert tuple(image.shape) == (4, 8, 8)
        assert tuple(mask.shape) == (1, 8, 8)
        assert float(mask.max().item()) == 1.0
        assert float(mask.min().item()) == 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_source_mask_file_is_unchanged_after_dataset_access() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        modality_paths, mask_path = _build_subject(tmp_dir, "BraTS-GLI-00001-000")
        relative_mask = Path(mask_path).resolve().relative_to(Path.cwd()).as_posix()
        record = SliceIndexRecord(
            subject_id="BraTS-GLI-00001-000",
            split="train",
            slice_index=1,
            contains_tumor=True,
            tumor_pixel_count=16,
            modality_paths={key: Path(value).resolve().relative_to(Path.cwd()).as_posix() for key, value in modality_paths.items()},
            mask_path=relative_mask,
        )
        before = compute_file_sha256(relative_mask)
        dataset = BraTSSliceDataset(records=[record], modalities=["t1c", "t1n", "t2f", "t2w"])
        _ = dataset[0]
        after = compute_file_sha256(relative_mask)
        assert before == after
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_slice_index_roundtrip_rejects_absolute_paths_and_preserves_portability() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        output_path = tmp_dir / "slices.csv"
        relative_record = SliceIndexRecord(
            subject_id="BraTS-GLI-00002-000",
            split="val",
            slice_index=0,
            contains_tumor=False,
            tumor_pixel_count=0,
            modality_paths={modality: f"data/{modality}.nii.gz" for modality in ("t1c", "t1n", "t2f", "t2w")},
            mask_path="data/seg.nii.gz",
        )
        write_slice_index_csv(output_path, [relative_record], modalities=["t1c", "t1n", "t2f", "t2w"])
        loaded = load_slice_index_records(output_path, modalities=["t1c", "t1n", "t2f", "t2w"])
        assert loaded[0] == relative_record
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_training_record_selection_keeps_all_positives_and_deterministic_negative_subset() -> None:
    positives = [
        SliceIndexRecord(
            subject_id=f"subject-{index}",
            split="train",
            slice_index=0,
            contains_tumor=True,
            tumor_pixel_count=10,
            modality_paths={modality: f"{modality}-{index}" for modality in ("t1c", "t1n", "t2f", "t2w")},
            mask_path=f"mask-{index}",
        )
        for index in range(4)
    ]
    negatives = [
        SliceIndexRecord(
            subject_id=f"subject-neg-{index}",
            split="train",
            slice_index=0,
            contains_tumor=False,
            tumor_pixel_count=0,
            modality_paths={modality: f"{modality}-neg-{index}" for modality in ("t1c", "t1n", "t2f", "t2w")},
            mask_path=f"mask-neg-{index}",
        )
        for index in range(8)
    ]
    selected_a = select_training_slice_records(positives + negatives, positive_negative_ratio=1.0, seed=42)
    selected_b = select_training_slice_records(positives + negatives, positive_negative_ratio=1.0, seed=42)
    assert selected_a == selected_b
    assert sum(1 for record in selected_a if record.contains_tumor) == len(positives)
    assert sum(1 for record in selected_a if not record.contains_tumor) == len(positives)
