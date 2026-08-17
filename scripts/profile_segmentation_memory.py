"""Profile Phase 3C segmentation memory usage without launching a full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, resolve_project_path
from brainscan.core.reproducibility import make_torch_generator, seed_worker, set_global_seed
from brainscan.core.system_metrics import capture_memory_snapshot
from brainscan.segmentation.data.dataset import BraTSSliceDataset, load_normalization_stats_map, select_training_slice_records
from brainscan.segmentation.data.preprocessing import SegmentationAugmentationConfig, SegmentationAugmenter
from brainscan.segmentation.data.slice_index import load_slice_index_records
from brainscan.segmentation.models import UNet2D
from brainscan.segmentation.training import BCEDiceLoss, build_optimizer
from brainscan.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile segmentation memory usage for the 2.5D Phase 3C pipeline.")
    parser.add_argument("--config", default="configs/segmentation_2p5d.yaml")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument(
        "--output",
        default="artifacts/segmentation/profiling/unet2p5d_whole_tumor/memory_profile.json",
    )
    return parser.parse_args()


def _snapshot(label: str, *, device: torch.device) -> dict[str, object]:
    payload = capture_memory_snapshot(device).to_dict()
    payload["label"] = label
    payload["timestamp_seconds"] = time.perf_counter()
    return payload


def main() -> int:
    args = parse_args()
    config = load_segmentation_config(args.config)
    set_global_seed(int(config["training"]["seed"]))
    device, device_info = resolve_device(prefer_cuda=True)
    modalities = list(config["dataset"]["modalities"])
    slice_dir = resolve_project_path(config["dataset"]["slice_index_dir"])
    stats_path = config["dataset"]["normalization_stats_path"]

    snapshots: list[dict[str, object]] = [_snapshot("before_dataset_init", device=device)]

    train_records_full = load_slice_index_records(slice_dir / "train.csv", modalities=modalities)
    val_records = load_slice_index_records(slice_dir / "val.csv", modalities=modalities)
    train_records = select_training_slice_records(
        train_records_full,
        positive_negative_ratio=float(config["sampling"]["positive_negative_ratio"]),
        seed=int(config["training"]["train_negative_sampling_seed"]),
    )
    normalization_stats = load_normalization_stats_map(stats_path)
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
        normalization_stats=normalization_stats,
    )
    _val_dataset = BraTSSliceDataset(
        records=val_records,
        modalities=modalities,
        return_metadata=True,
        max_subject_cache_size=int(config["training"]["max_subject_cache_size"]),
        dataset_root=config["dataset"]["root"],
        slice_offsets=list(config["dataset"].get("slice_offsets", [0])),
        neighbor_policy=str(config["dataset"].get("neighbor_policy", "edge_replicate")),
        normalization_stats=normalization_stats,
    )
    snapshots.append(_snapshot("after_dataset_init", device=device))

    _ = train_dataset[0]
    snapshots.append(_snapshot("after_one_sample", device=device))

    for index in range(min(10, len(train_dataset))):
        _ = train_dataset[index]
    snapshots.append(_snapshot("after_ten_samples", device=device))

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    pin_memory = bool(config["training"].get("pin_memory", False)) and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(int(config["training"]["seed"])),
    )

    first_batch = next(iter(train_loader))
    snapshots.append(_snapshot("after_one_batch_loaded", device=device))

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

    inputs, targets = first_batch[:2]
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    snapshots.append(_snapshot("after_forward_pass", device=device))
    loss.backward()
    optimizer.step()
    snapshots.append(_snapshot("after_backward_pass", device=device))

    batch_losses: list[float] = [float(loss.item())]
    peak_process_rss_mb = None
    peak_process_private_mb = None
    peak_cuda_allocated_mb = None
    peak_cuda_reserved_mb = None

    model.train()
    batch_start_time = time.perf_counter()
    for batch_index, batch in enumerate(train_loader, start=1):
        if batch_index > int(args.max_batches):
            break
        inputs, targets = batch[:2]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Encountered non-finite loss during profiling: {loss.item()}")
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.item()))
        if batch_index in {1, min(25, int(args.max_batches)), min(50, int(args.max_batches)), int(args.max_batches)}:
            snapshot = _snapshot(f"after_{batch_index}_training_batches", device=device)
            snapshots.append(snapshot)
            process_rss_mb = snapshot.get("process_rss_mb")
            process_private_mb = snapshot.get("process_private_mb")
            cuda_allocated_mb = snapshot.get("cuda_allocated_mb")
            cuda_reserved_mb = snapshot.get("cuda_reserved_mb")
            if process_rss_mb is not None:
                peak_process_rss_mb = max(process_rss_mb, peak_process_rss_mb or process_rss_mb)
            if process_private_mb is not None:
                peak_process_private_mb = max(process_private_mb, peak_process_private_mb or process_private_mb)
            if cuda_allocated_mb is not None:
                peak_cuda_allocated_mb = max(cuda_allocated_mb, peak_cuda_allocated_mb or cuda_allocated_mb)
            if cuda_reserved_mb is not None:
                peak_cuda_reserved_mb = max(cuda_reserved_mb, peak_cuda_reserved_mb or cuda_reserved_mb)

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(resolve_project_path(args.config)),
        "device": str(device),
        "device_info": device_info,
        "test_used": False,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "max_batches_profiled": int(args.max_batches),
        "elapsed_seconds": time.perf_counter() - batch_start_time,
        "loss_first_batch": batch_losses[0],
        "loss_last_batch": batch_losses[-1],
        "loss_finite": all(torch.isfinite(torch.tensor(batch_losses)).tolist()),
        "snapshots": snapshots,
        "peak_process_rss_mb": peak_process_rss_mb,
        "peak_process_private_mb": peak_process_private_mb,
        "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
