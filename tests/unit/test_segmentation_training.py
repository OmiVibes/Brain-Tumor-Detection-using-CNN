from __future__ import annotations

from copy import deepcopy
from inspect import signature
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from brainscan.segmentation.training import (
    BCEDiceLoss,
    build_checkpoint_metadata,
    build_optimizer,
    build_scheduler,
    fit_segmenter,
)
from brainscan.training import load_checkpoint


class _TrainDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.zeros((4, 16, 16), dtype=torch.float32)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, 4:8, 4:8] = 1.0 + index
        mask[:, 4:8, 4:8] = 1.0
        return image, mask


class _ValDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        subject_id = "subject-a" if index < 2 else "subject-b"
        slice_index = index % 2
        image = torch.zeros((4, 16, 16), dtype=torch.float32)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, 4:8, 4:8] = 1.0
        mask[:, 4:8, 4:8] = 1.0
        metadata = {
            "subject_id": subject_id,
            "split": "val",
            "slice_index": slice_index,
            "contains_tumor": True,
            "tumor_pixel_count": 16,
            "mask_path": "relative-mask.nii.gz",
            "modality_paths": {modality: f"{modality}.nii.gz" for modality in ("t1c", "t1n", "t2f", "t2w")},
        }
        return image, mask, metadata


def _config(tmp_path) -> dict[str, object]:
    return {
        "dataset": {
            "modalities": ["t1c", "t1n", "t2f", "t2w"],
            "slice_offsets": [0],
            "neighbor_policy": "edge_replicate",
        },
        "model": {"architecture": "unet2d", "input_channels": 4, "output_channels": 1},
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "epochs": 2,
            "early_stopping_patience": 10,
            "seed": 42,
            "train_negative_sampling_seed": 42,
        },
        "sampling": {"positive_negative_ratio": 1.0},
        "checkpoint": {"monitor": "val_subject_dice", "mode": "max"},
        "inference": {"threshold": 0.5},
    }


def test_checkpoint_metadata_contains_segmentation_fingerprint() -> None:
    metadata = build_checkpoint_metadata(
        config=_config(None),
        dataset_fingerprint="fp123",
        train_subject_count=10,
        val_subject_count=2,
    )
    assert metadata["dataset_fingerprint"] == "fp123"
    assert metadata["binary_mask_rule"] == "mask > 0"


def test_fit_segmenter_saves_checkpoint_and_selection_artifact(tmp_path) -> None:
    model = nn.Conv2d(4, 1, kernel_size=1)
    loader_train = DataLoader(_TrainDataset(), batch_size=2, shuffle=False)
    loader_val = DataLoader(_ValDataset(), batch_size=2, shuffle=False)
    criterion = BCEDiceLoss()
    optimizer = build_optimizer(model, _config(tmp_path))
    scheduler = build_scheduler(optimizer)
    results = fit_segmenter(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        config=_config(tmp_path),
        checkpoint_metadata={
            "architecture": "unet2d",
            "input_channels": 4,
            "dataset_fingerprint": "fp123",
            "input_modalities": ["t1c", "t1n", "t2f", "t2w"],
            "normalization": "zscore_nonzero_per_modality",
            "binary_mask_rule": "mask > 0",
            "neighbor_policy": "edge_replicate",
            "slice_offsets": [0],
            "sampling": {"positive_negative_ratio": 1.0, "train_negative_sampling_seed": 42},
            "test_used": False,
            "git_commit": "deadbeef",
        },
        best_checkpoint_path=tmp_path / "best.pt",
        last_checkpoint_path=tmp_path / "last.pt",
        history_json_path=tmp_path / "history.json",
        history_csv_path=tmp_path / "history.csv",
        model_selection_path=tmp_path / "model_selection.json",
        max_train_batches=1,
        max_val_batches=1,
    )
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    selection = json.loads((tmp_path / "model_selection.json").read_text(encoding="utf-8"))
    assert selection["dataset_fingerprint"] == "fp123"
    assert not Path(selection["selected_checkpoint"]).is_absolute()

    reloaded_model = nn.Conv2d(4, 1, kernel_size=1)
    load_checkpoint(tmp_path / "best.pt", reloaded_model)
    sample = next(iter(loader_train))[0]
    with torch.no_grad():
        original = model(sample)
        reloaded = reloaded_model(sample)
    assert original.shape == reloaded.shape
    assert results["epochs_completed"] >= 1


def test_segmentation_training_api_does_not_accept_test_loader() -> None:
    assert "test_loader" not in signature(fit_segmenter).parameters


def test_phase3c_training_script_does_not_open_test_manifest() -> None:
    source = Path("scripts/train_segmentation.py").read_text(encoding="utf-8")
    assert 'slice_dir / "test.csv"' not in source
