"""Formally evaluate the frozen whole-tumor U-Net on the held-out test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, resolve_project_path
from brainscan.segmentation.data.dataset import BraTSSliceDataset
from brainscan.segmentation.data.slice_index import SliceIndexRecord, load_slice_index_records
from brainscan.segmentation.models import UNet2D
from brainscan.segmentation.training import (
    BCEDiceLoss,
    build_review_grid,
    evaluate_segmentation_loader,
    summarize_subject_metrics,
    write_json,
    write_subject_metrics_csv,
)
from brainscan.training import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen BrainScanAI whole-tumor U-Net.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--selection", default="artifacts/segmentation/model_selection.json")
    return parser.parse_args()


def _normalized_t2f_slice(record: dict[str, str], dataset_root: Path) -> np.ndarray:
    import nibabel as nib
    volume = np.asanyarray(nib.load(str((dataset_root / record["t2f_path"]).resolve())).dataobj).astype(np.float32)
    slice_index = int(record["slice_index"])
    image_slice = volume[:, :, slice_index]
    minimum = float(image_slice.min())
    maximum = float(image_slice.max())
    if maximum <= minimum:
        return np.zeros_like(image_slice, dtype=np.float32)
    return np.clip((image_slice - minimum) / (maximum - minimum), 0.0, 1.0)


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@torch.no_grad()
def _predict_subject_triplet(
    *,
    model: UNet2D,
    subject_records: list[SliceIndexRecord],
    modalities: list[str],
    dataset_root: Path,
    device: torch.device,
    threshold: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = BraTSSliceDataset(
        records=subject_records,
        modalities=modalities,
        return_metadata=True,
        dataset_root=dataset_root,
        max_subject_cache_size=1,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    first_image, first_target, _ = dataset[0]
    height, width = int(first_image.shape[1]), int(first_image.shape[2])
    depth = max(int(record.slice_index) for record in subject_records) + 1
    prediction_volume = np.zeros((height, width, depth), dtype=np.uint8)
    target_volume = np.zeros((height, width, depth), dtype=np.uint8)

    model.eval()
    for inputs, targets, metadata in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).to(dtype=torch.uint8).cpu().numpy()
        target_batch = targets.to(dtype=torch.uint8).cpu().numpy()
        slice_indices = [int(value) for value in list(metadata["slice_index"])]
        for batch_index, slice_index in enumerate(slice_indices):
            prediction_volume[:, :, slice_index] = predictions[batch_index, 0]
            target_volume[:, :, slice_index] = target_batch[batch_index, 0]

    positive_counts = target_volume.reshape(-1, depth).sum(axis=0)
    if int(positive_counts.max()) > 0:
        representative_slice = int(np.argmax(positive_counts))
    else:
        representative_slice = depth // 2

    image_row = {
        "t2f_path": subject_records[representative_slice].modality_paths["t2f"],
        "slice_index": representative_slice,
    }
    image_slice = _normalized_t2f_slice(image_row, dataset_root)
    return (
        image_slice,
        target_volume[:, :, representative_slice].astype(np.float32),
        prediction_volume[:, :, representative_slice].astype(np.float32),
    )


def main() -> int:
    args = parse_args()
    config = load_segmentation_config(args.config)
    selection = json.loads(resolve_project_path(args.selection).read_text(encoding="utf-8"))
    modalities = list(config["dataset"]["modalities"])
    if selection["dataset_fingerprint"] != "5bc6c1fe8868265cc8df6042ba353891aff6707f9b1acb65c4816c03920314db":
        raise ValueError("Frozen model selection artifact does not match the audited segmentation dataset fingerprint.")
    dataset_root = resolve_project_path(config["dataset"]["root"])

    device, _device_info = resolve_device(prefer_cuda=True)
    model = UNet2D(
        input_channels=int(config["model"]["input_channels"]),
        output_channels=int(config["model"]["output_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    checkpoint_path = selection["selected_checkpoint"]
    load_checkpoint(checkpoint_path, model, map_location=device)

    slice_dir = resolve_project_path(config["dataset"]["slice_index_dir"])
    test_records = load_slice_index_records(slice_dir / "test.csv", modalities=modalities)
    test_dataset = BraTSSliceDataset(
        records=test_records,
        modalities=modalities,
        return_metadata=True,
        max_subject_cache_size=int(config["training"]["max_subject_cache_size"]),
        dataset_root=config["dataset"]["root"],
    )
    loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    criterion = BCEDiceLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
    )
    metrics = evaluate_segmentation_loader(
        model,
        loader,
        device=device,
        criterion=criterion,
        threshold=float(selection["threshold"]),
        amp_enabled=bool(config["training"]["mixed_precision"]) and device.type == "cuda",
    )
    subject_rows = metrics["subject_metrics"]
    summary = summarize_subject_metrics(subject_rows)
    evaluation_dir = Path("artifacts/segmentation/evaluation/unet2d_whole_tumor")
    subject_metrics_csv = write_subject_metrics_csv(evaluation_dir / "subject_metrics.csv", subject_rows)

    best_case = max(subject_rows, key=lambda row: row.dice)
    median_case = sorted(subject_rows, key=lambda row: row.dice)[len(subject_rows) // 2]
    worst_case = min(subject_rows, key=lambda row: row.dice)

    subject_lookup: dict[str, list[SliceIndexRecord]] = {}
    for record in test_records:
        subject_lookup.setdefault(record.subject_id, []).append(record)

    review_grid = build_review_grid(
        evaluation_dir / "review_grid.png",
        good_case=_predict_subject_triplet(
            model=model,
            subject_records=subject_lookup[best_case.subject_id],
            modalities=modalities,
            dataset_root=dataset_root,
            device=device,
            threshold=float(selection["threshold"]),
            batch_size=int(config["training"]["batch_size"]),
        ),
        median_case=_predict_subject_triplet(
            model=model,
            subject_records=subject_lookup[median_case.subject_id],
            modalities=modalities,
            dataset_root=dataset_root,
            device=device,
            threshold=float(selection["threshold"]),
            batch_size=int(config["training"]["batch_size"]),
        ),
        poor_case=_predict_subject_triplet(
            model=model,
            subject_records=subject_lookup[worst_case.subject_id],
            modalities=modalities,
            dataset_root=dataset_root,
            device=device,
            threshold=float(selection["threshold"]),
            batch_size=int(config["training"]["batch_size"]),
        ),
    )

    failure_analysis = {
        "best_case": best_case.__dict__,
        "median_case": median_case.__dict__,
        "worst_case": worst_case.__dict__,
        "notes": [
            "Failure analysis is descriptive only and does not claim clinical causality.",
            "False positives and false negatives are reported from subject-level whole-volume masks.",
        ],
    }
    write_json(evaluation_dir / "failure_analysis.json", failure_analysis)
    payload = {
        "checkpoint": checkpoint_path,
        "threshold": selection["threshold"],
        "subject_count": summary["subject_count"],
        "mean_dice": summary["mean_dice"],
        "median_dice": summary["median_dice"],
        "dice_std": summary["dice_std"],
        "mean_iou": summary["mean_iou"],
        "mean_precision": summary["mean_precision"],
        "mean_recall": summary["mean_recall"],
        "mean_specificity": summary["mean_specificity"],
        "dice_distribution": summary["dice_distribution"],
        "review_grid": _project_relative(review_grid),
        "subject_metrics_csv": _project_relative(subject_metrics_csv),
    }
    write_json(evaluation_dir / "summary.json", payload)
    print("segmentation_test_summary=")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
