from __future__ import annotations

from pathlib import Path
import json

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from brainscan.segmentation.data.dataset import (
    BraTSSliceDataset,
    SubjectVolumeLRUCache,
    load_normalization_stats_map,
)
from brainscan.segmentation.data.preprocessing import VolumeNormalizationStats
from brainscan.segmentation.data.slice_index import SliceIndexRecord
from brainscan.segmentation.training import (
    BCEDiceLoss,
    build_checkpoint_metadata,
    build_optimizer,
    build_scheduler,
    fit_segmenter,
)


class _RecordingProxy:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.reads: list[object] = []

    def __getitem__(self, item):
        self.reads.append(item)
        if not isinstance(item, tuple) or len(item) != 3 or not isinstance(item[2], int):
            raise AssertionError(f"Expected lazy axial slice access only, got {item!r}")
        return self.array[item]


class _FakeImage:
    def __init__(self, array: np.ndarray) -> None:
        self.dataobj = _RecordingProxy(array)
        self.shape = array.shape


class _ConstantDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.zeros((12, 16, 16), dtype=torch.float32)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, 4:8, 4:8] = 1.0 + index
        mask[:, 4:8, 4:8] = 1.0
        return image, mask


class _ConstantValDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.zeros((12, 16, 16), dtype=torch.float32)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, 4:8, 4:8] = 1.0
        mask[:, 4:8, 4:8] = 1.0
        metadata = {
            "subject_id": "subject-a" if index < 2 else "subject-b",
            "split": "val",
            "slice_index": index % 2,
            "contains_tumor": True,
            "tumor_pixel_count": 16,
            "mask_path": "relative-mask.nii.gz",
            "modality_paths": {modality: f"{modality}.nii.gz" for modality in ("t1c", "t1n", "t2f", "t2w")},
        }
        return image, mask, metadata


def _config() -> dict[str, object]:
    return {
        "dataset": {
            "modalities": ["t1c", "t1n", "t2f", "t2w"],
            "slice_offsets": [-1, 0, 1],
            "neighbor_policy": "edge_replicate",
        },
        "model": {"architecture": "unet2p5d", "input_channels": 12, "output_channels": 1},
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "epochs": 1,
            "early_stopping_patience": 10,
            "seed": 42,
            "train_negative_sampling_seed": 42,
            "log_every_batches": 0,
            "checkpoint_every_batches": 1,
        },
        "sampling": {"positive_negative_ratio": 1.0},
        "checkpoint": {"monitor": "val_subject_dice", "mode": "max"},
        "inference": {"threshold": 0.5},
    }


def test_lazy_slice_loading_reads_only_requested_neighbor_slices(monkeypatch) -> None:
    modality_arrays = {
        "t1c": np.ones((6, 6, 3), dtype=np.float32),
        "t1n": np.ones((6, 6, 3), dtype=np.float32) * 2,
        "t2f": np.ones((6, 6, 3), dtype=np.float32) * 3,
        "t2w": np.ones((6, 6, 3), dtype=np.float32) * 4,
    }
    mask_array = np.zeros((6, 6, 3), dtype=np.uint8)
    mask_array[2:4, 2:4, 1] = 1
    image_lookup = {f"{modality}.nii.gz": _FakeImage(array) for modality, array in modality_arrays.items()}
    image_lookup["seg.nii.gz"] = _FakeImage(mask_array)

    monkeypatch.setattr(
        "brainscan.segmentation.data.dataset.load_nifti_image",
        lambda path_value: image_lookup[Path(path_value).name],
    )

    dataset = BraTSSliceDataset(
        records=[
            SliceIndexRecord(
                subject_id="subject-a",
                split="train",
                slice_index=1,
                contains_tumor=True,
                tumor_pixel_count=4,
                modality_paths={modality: f"{modality}.nii.gz" for modality in modality_arrays},
                mask_path="seg.nii.gz",
            )
        ],
        modalities=["t1c", "t1n", "t2f", "t2w"],
        slice_offsets=[-1, 0, 1],
        normalization_stats={
            "subject-a": {
                modality: VolumeNormalizationStats(mean_nonzero=0.0, std_nonzero=1.0, nonzero_count=108)
                for modality in modality_arrays
            }
        },
    )

    _image, mask = dataset[0]
    assert tuple(mask.shape) == (1, 6, 6)
    for modality in modality_arrays:
        assert image_lookup[f"{modality}.nii.gz"].dataobj.reads == [
            (slice(None, None, None), slice(None, None, None), 0),
            (slice(None, None, None), slice(None, None, None), 1),
            (slice(None, None, None), slice(None, None, None), 2),
        ]
    assert image_lookup["seg.nii.gz"].dataobj.reads == [
        (slice(None, None, None), slice(None, None, None), 1)
    ]


def test_subject_cache_evicts_oldest_entry() -> None:
    cache = SubjectVolumeLRUCache(max_size=1)
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4))
    entry_a = object()
    entry_b = object()
    cache.put("a", entry_a)  # type: ignore[arg-type]
    cache.put("b", entry_b)  # type: ignore[arg-type]
    assert cache.get("a") is None
    assert cache.get("b") is entry_b


def test_load_normalization_stats_map_reads_portable_json(tmp_path) -> None:
    payload = {
        "subjects": {
            "subject-a": {
                "t1c": {"mean_nonzero": 1.0, "std_nonzero": 2.0, "nonzero_count": 3},
            }
        }
    }
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    stats_map = load_normalization_stats_map(stats_path)
    assert stats_map["subject-a"]["t1c"].mean_nonzero == 1.0
    assert stats_map["subject-a"]["t1c"].std_nonzero == 2.0
    assert stats_map["subject-a"]["t1c"].nonzero_count == 3


def test_fit_segmenter_writes_recovery_checkpoint_and_clears_it_after_epoch(tmp_path) -> None:
    model = nn.Conv2d(12, 1, kernel_size=1)
    config = _config()
    optimizer = build_optimizer(model, config)
    criterion = BCEDiceLoss()
    history_json = tmp_path / "history.json"
    fit_segmenter(
        model=model,
        train_loader=DataLoader(_ConstantDataset(), batch_size=2, shuffle=False),
        val_loader=DataLoader(_ConstantValDataset(), batch_size=2, shuffle=False),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer),
        device=torch.device("cpu"),
        config=config,
        checkpoint_metadata=build_checkpoint_metadata(
            config=config,
            dataset_fingerprint="fp123",
            train_subject_count=2,
            val_subject_count=1,
        ),
        best_checkpoint_path=tmp_path / "best.pt",
        last_checkpoint_path=tmp_path / "last.pt",
        history_json_path=history_json,
        history_csv_path=tmp_path / "history.csv",
        model_selection_path=tmp_path / "selection.json",
        max_train_batches=1,
        max_val_batches=1,
    )
    assert not history_json.with_name("recovery.pt").exists()


def test_full_training_refuses_to_resume_smoke_checkpoint(tmp_path) -> None:
    smoke_checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "model_state_dict": nn.Conv2d(12, 1, kernel_size=1).state_dict(),
            "optimizer_state_dict": build_optimizer(nn.Conv2d(12, 1, kernel_size=1), _config()).state_dict(),
            "scheduler_state_dict": None,
            "epoch": 1,
            "metrics": {},
            "metadata": {"run_type": "smoke_test"},
        },
        smoke_checkpoint,
    )
    history_json = tmp_path / "history.json"
    history_json.write_text("[]", encoding="utf-8")
    model = nn.Conv2d(12, 1, kernel_size=1)
    optimizer = build_optimizer(model, _config())
    try:
        fit_segmenter(
            model=model,
            train_loader=DataLoader(_ConstantDataset(), batch_size=2, shuffle=False),
            val_loader=DataLoader(_ConstantValDataset(), batch_size=2, shuffle=False),
            criterion=BCEDiceLoss(),
            optimizer=optimizer,
            scheduler=build_scheduler(optimizer),
            device=torch.device("cpu"),
            config=_config(),
            checkpoint_metadata=build_checkpoint_metadata(
                config=_config(),
                dataset_fingerprint="fp123",
                train_subject_count=2,
                val_subject_count=1,
            ),
            best_checkpoint_path=tmp_path / "best.pt",
            last_checkpoint_path=smoke_checkpoint,
            history_json_path=history_json,
            history_csv_path=tmp_path / "history.csv",
            model_selection_path=tmp_path / "selection.json",
            max_train_batches=1,
            max_val_batches=1,
        )
    except ValueError as exc:
        assert "run type" in str(exc)
    else:
        raise AssertionError("Expected full training resume to reject smoke checkpoint artifacts.")


def test_phase3c_config_uses_memory_safe_defaults() -> None:
    config_text = Path("configs/segmentation_2p5d.yaml").read_text(encoding="utf-8")
    assert "batch_size: 8" in config_text
    assert "num_workers: 0" in config_text
    assert "pin_memory: false" in config_text
