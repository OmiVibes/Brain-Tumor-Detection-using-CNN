from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import nibabel as nib
import numpy as np

from brainscan.segmentation.data.dataset import (
    BraTSSliceDataset,
    extract_slice_context,
    resolve_neighbor_indices,
)
from brainscan.segmentation.data.slice_index import SliceIndexRecord


def _workspace_tmp_dir() -> Path:
    path = (Path.cwd() / ".tmp" / "segmentation_2p5d_dataset_unit" / uuid.uuid4().hex).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_nifti(path: Path, array: np.ndarray) -> None:
    image = nib.Nifti1Image(array, affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _build_subject(root: Path, subject_id: str) -> tuple[dict[str, str], str]:
    subject_dir = root / subject_id
    mask = np.zeros((6, 6, 3), dtype=np.uint8)
    mask[1:5, 1:5, 1] = 1
    modality_paths: dict[str, str] = {}
    for modality_index, modality in enumerate(("t1c", "t1n", "t2f", "t2w"), start=1):
        volume = np.zeros((6, 6, 3), dtype=np.float32)
        volume[:, :, 0] = modality_index * 10 + 1
        volume[:, :, 1] = modality_index * 10 + 2
        volume[:, :, 2] = modality_index * 10 + 3
        path = subject_dir / f"{subject_id}-{modality}.nii.gz"
        _write_nifti(path, volume)
        modality_paths[modality] = path.resolve().relative_to(Path.cwd()).as_posix()
    mask_path = subject_dir / f"{subject_id}-seg.nii.gz"
    _write_nifti(mask_path, mask)
    return modality_paths, mask_path.resolve().relative_to(Path.cwd()).as_posix()


def test_neighbor_indices_replicate_volume_edges() -> None:
    assert resolve_neighbor_indices(center_index=0, depth=155, slice_offsets=[-1, 0, 1], neighbor_policy="edge_replicate") == [0, 0, 1]
    assert resolve_neighbor_indices(center_index=154, depth=155, slice_offsets=[-1, 0, 1], neighbor_policy="edge_replicate") == [153, 154, 154]


def test_extract_slice_context_is_modality_major_and_offset_minor() -> None:
    modalities_volume = np.zeros((2, 2, 2, 3), dtype=np.float32)
    modalities_volume[0, :, :, 0] = 10
    modalities_volume[0, :, :, 1] = 11
    modalities_volume[0, :, :, 2] = 12
    modalities_volume[1, :, :, 0] = 20
    modalities_volume[1, :, :, 1] = 21
    modalities_volume[1, :, :, 2] = 22
    context = extract_slice_context(
        modalities_volume,
        center_index=1,
        slice_offsets=[-1, 0, 1],
        neighbor_policy="edge_replicate",
    )
    assert tuple(context.shape) == (6, 2, 2)
    assert float(context[0, 0, 0]) == 10
    assert float(context[1, 0, 0]) == 11
    assert float(context[2, 0, 0]) == 12
    assert float(context[3, 0, 0]) == 20
    assert float(context[4, 0, 0]) == 21
    assert float(context[5, 0, 0]) == 22


def test_2p5d_dataset_returns_twelve_channels_and_center_slice_target() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        modality_paths, mask_path = _build_subject(tmp_dir, "BraTS-GLI-10000-000")
        dataset = BraTSSliceDataset(
            records=[
                SliceIndexRecord(
                    subject_id="BraTS-GLI-10000-000",
                    split="train",
                    slice_index=1,
                    contains_tumor=True,
                    tumor_pixel_count=16,
                    modality_paths=modality_paths,
                    mask_path=mask_path,
                )
            ],
            modalities=["t1c", "t1n", "t2f", "t2w"],
            slice_offsets=[-1, 0, 1],
            neighbor_policy="edge_replicate",
        )
        image, mask = dataset[0]
        assert tuple(image.shape) == (12, 6, 6)
        assert tuple(mask.shape) == (1, 6, 6)
        assert float(mask[:, 2, 2].item()) == 1.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_2p5d_dataset_boundary_context_stays_within_same_subject() -> None:
    tmp_dir = _workspace_tmp_dir()
    try:
        modality_paths, mask_path = _build_subject(tmp_dir, "BraTS-GLI-10001-000")
        dataset = BraTSSliceDataset(
            records=[
                SliceIndexRecord(
                    subject_id="BraTS-GLI-10001-000",
                    split="val",
                    slice_index=0,
                    contains_tumor=False,
                    tumor_pixel_count=0,
                    modality_paths=modality_paths,
                    mask_path=mask_path,
                )
            ],
            modalities=["t1c", "t1n", "t2f", "t2w"],
            slice_offsets=[-1, 0, 1],
            neighbor_policy="edge_replicate",
        )
        image, _mask = dataset[0]
        assert tuple(image.shape) == (12, 6, 6)
        assert np.allclose(image[0].numpy(), image[1].numpy())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
