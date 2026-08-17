"""Train the BrainScanAI 2D U-Net whole-tumor segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, resolve_project_path
from brainscan.core.reproducibility import make_torch_generator, seed_worker, set_global_seed
from brainscan.segmentation.data.dataset import BraTSSliceDataset, select_training_slice_records
from brainscan.segmentation.data.preprocessing import SegmentationAugmentationConfig, SegmentationAugmenter
from brainscan.segmentation.data.slice_index import count_split_subjects, load_slice_index_records
from brainscan.segmentation.models import UNet2D, count_parameters
from brainscan.segmentation.training import (
    BCEDiceLoss,
    build_checkpoint_metadata,
    build_optimizer,
    build_scheduler,
    fit_segmenter,
)
from brainscan.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BrainScanAI whole-tumor segmentation experiment.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_segmentation_config(args.config)
    set_global_seed(int(config["training"]["seed"]))
    device, device_info = resolve_device(prefer_cuda=True)
    print(f"device={device}")
    print(f"device_info={device_info}")
    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("CUDA is unavailable. Full Phase 3 segmentation training must not degrade into a long CPU run.")

    modalities = list(config["dataset"]["modalities"])
    slice_dir = resolve_project_path(config["dataset"]["slice_index_dir"])
    train_records_full = load_slice_index_records(slice_dir / "train.csv", modalities=modalities)
    val_records = load_slice_index_records(slice_dir / "val.csv", modalities=modalities)
    train_subjects = count_split_subjects(train_records_full)["train"]
    val_subjects = count_split_subjects(val_records)["val"]
    if train_subjects & val_subjects:
        raise ValueError("Segmentation subject leakage detected before training.")

    train_records = select_training_slice_records(
        train_records_full,
        positive_negative_ratio=float(config["sampling"]["positive_negative_ratio"]),
        seed=int(config["training"]["train_negative_sampling_seed"]),
    )
    train_positive = sum(1 for record in train_records if record.contains_tumor)
    train_negative = len(train_records) - train_positive

    augmenter = SegmentationAugmenter(
        SegmentationAugmentationConfig(
            horizontal_flip_probability=float(config["augmentation"]["horizontal_flip_probability"]),
            max_rotation_degrees=float(config["augmentation"]["max_rotation_degrees"]),
            max_translation_pixels=int(config["augmentation"]["max_translation_pixels"]),
        )
    )
    train_dataset = BraTSSliceDataset(
        records=train_records,
        modalities=modalities,
        augmenter=augmenter,
        max_subject_cache_size=int(config["training"]["max_subject_cache_size"]),
        dataset_root=config["dataset"]["root"],
        slice_offsets=list(config["dataset"].get("slice_offsets", [0])),
        neighbor_policy=str(config["dataset"].get("neighbor_policy", "edge_replicate")),
    )
    val_dataset = BraTSSliceDataset(
        records=val_records,
        modalities=modalities,
        return_metadata=True,
        max_subject_cache_size=int(config["training"]["max_subject_cache_size"]),
        dataset_root=config["dataset"]["root"],
        slice_offsets=list(config["dataset"].get("slice_offsets", [0])),
        neighbor_policy=str(config["dataset"].get("neighbor_policy", "edge_replicate")),
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(int(config["training"]["seed"])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    model = UNet2D(
        input_channels=int(config["model"]["input_channels"]),
        output_channels=int(config["model"]["output_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer)

    dataset_fingerprint = "5bc6c1fe8868265cc8df6042ba353891aff6707f9b1acb65c4816c03920314db"
    checkpoint_metadata = build_checkpoint_metadata(
        config=config,
        dataset_fingerprint=dataset_fingerprint,
        train_subject_count=len(train_subjects),
        val_subject_count=len(val_subjects),
    )

    checkpoint_config = config["checkpoint"]
    best_checkpoint_path = Path(checkpoint_config["model_dir"]) / str(checkpoint_config["best_filename"])
    last_checkpoint_path = Path(checkpoint_config["model_dir"]) / str(checkpoint_config["last_filename"])
    history_json_path = Path(checkpoint_config["training_dir"]) / str(checkpoint_config["history_json"])
    history_csv_path = Path(checkpoint_config["training_dir"]) / str(checkpoint_config["history_csv"])
    model_selection_path = Path(checkpoint_config["selection_path"])

    if args.smoke_test:
        config["training"]["epochs"] = min(int(config["training"]["epochs"]), 2)
        args.max_train_batches = args.max_train_batches or 4
        args.max_val_batches = args.max_val_batches or 2

    loader_smoke_start = time.perf_counter()
    sample_batch = next(iter(train_loader))
    loader_smoke_seconds = time.perf_counter() - loader_smoke_start
    sample_inputs, sample_targets = sample_batch[:2]
    print(
        "segmentation_setup="
        f"parameters={count_parameters(model)} "
        f"train_selected_slices={len(train_records)} "
        f"train_positive_slices={train_positive} "
        f"train_negative_slices={train_negative} "
        f"val_slices={len(val_records)} "
        f"sample_shape={tuple(sample_inputs.shape)} "
        f"target_shape={tuple(sample_targets.shape)} "
        f"loader_smoke_seconds={loader_smoke_seconds:.4f} "
        f"test_used=false"
    )

    results = fit_segmenter(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_metadata=checkpoint_metadata,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        history_json_path=history_json_path,
        history_csv_path=history_csv_path,
        model_selection_path=model_selection_path,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    print("segmentation_training_summary=")
    print(f"  epochs_completed={results['epochs_completed']}")
    print(f"  best_epoch={results['best_epoch']}")
    print(f"  best_val_subject_dice={results['best_score']}")
    print(f"  peak_vram_bytes={results['peak_vram_bytes']}")
    print(f"  best_checkpoint={resolve_project_path(results['best_checkpoint_path'])}")
    print(f"  last_checkpoint={resolve_project_path(results['last_checkpoint_path'])}")
    print(f"  history_json={resolve_project_path(results['history_json_path'])}")
    print(f"  history_csv={resolve_project_path(results['history_csv_path'])}")
    print(f"  model_selection={resolve_project_path(model_selection_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
