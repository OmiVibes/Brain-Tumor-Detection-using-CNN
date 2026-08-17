from __future__ import annotations

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
    load_training_history,
    load_training_state,
)


class _ResumeTrainDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.zeros((12, 16, 16), dtype=torch.float32)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, 3:9, 3:9] = 0.5 + index
        mask[:, 4:8, 4:8] = 1.0
        return image, mask


class _ResumeValDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        subject_id = "subject-a" if index < 2 else "subject-b"
        slice_index = index % 2
        image = torch.zeros((12, 16, 16), dtype=torch.float32)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, 3:9, 3:9] = 1.0
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
            "epochs": 2,
            "early_stopping_patience": 10,
            "seed": 42,
            "train_negative_sampling_seed": 42,
        },
        "sampling": {"positive_negative_ratio": 1.0},
        "checkpoint": {"monitor": "val_subject_dice", "mode": "max"},
        "inference": {"threshold": 0.5},
    }


def test_segmentation_resume_restores_history_and_state(tmp_path) -> None:
    history_json = tmp_path / "history.json"
    history_csv = tmp_path / "history.csv"
    best_checkpoint = tmp_path / "best.pt"
    last_checkpoint = tmp_path / "last.pt"
    selection_path = tmp_path / "model_selection_2p5d.json"

    first_model = nn.Conv2d(12, 1, kernel_size=1)
    first_optimizer = build_optimizer(first_model, _config())
    criterion = BCEDiceLoss()
    fit_segmenter(
        model=first_model,
        train_loader=DataLoader(_ResumeTrainDataset(), batch_size=2, shuffle=False),
        val_loader=DataLoader(_ResumeValDataset(), batch_size=2, shuffle=False),
        criterion=criterion,
        optimizer=first_optimizer,
        scheduler=build_scheduler(first_optimizer),
        device=torch.device("cpu"),
        config=_config(),
        checkpoint_metadata=build_checkpoint_metadata(
            config=_config(),
            dataset_fingerprint="fp123",
            train_subject_count=10,
            val_subject_count=2,
        ),
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        history_json_path=history_json,
        history_csv_path=history_csv,
        model_selection_path=selection_path,
        max_train_batches=1,
        max_val_batches=1,
    )

    resumed_config = _config()
    resumed_config["training"]["epochs"] = 4
    resumed_model = nn.Conv2d(12, 1, kernel_size=1)
    resumed_optimizer = build_optimizer(resumed_model, resumed_config)
    resumed_results = fit_segmenter(
        model=resumed_model,
        train_loader=DataLoader(_ResumeTrainDataset(), batch_size=2, shuffle=False),
        val_loader=DataLoader(_ResumeValDataset(), batch_size=2, shuffle=False),
        criterion=criterion,
        optimizer=resumed_optimizer,
        scheduler=build_scheduler(resumed_optimizer),
        device=torch.device("cpu"),
        config=resumed_config,
        checkpoint_metadata=build_checkpoint_metadata(
            config=resumed_config,
            dataset_fingerprint="fp123",
            train_subject_count=10,
            val_subject_count=2,
        ),
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        history_json_path=history_json,
        history_csv_path=history_csv,
        model_selection_path=selection_path,
        max_train_batches=1,
        max_val_batches=1,
    )

    persisted_history = load_training_history(history_json)
    persisted_state = load_training_state(history_json)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert resumed_results["resumed_from_epoch"] == 2
    assert resumed_results["epochs_completed"] == 4
    assert len(persisted_history) == 4
    assert persisted_history[-1]["epoch"] == 4
    assert persisted_state["last_completed_epoch"] == 4
    assert "early_stopping" in persisted_state
    assert "random_state" in persisted_state
    assert selection["test_used"] is False
