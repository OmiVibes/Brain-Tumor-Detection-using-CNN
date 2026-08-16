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
from brainscan.segmentation.data.slice_index import load_slice_index_records
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

    subject_lookup: dict[str, list[dict[str, str]]] = {}
    for record in test_records:
        subject_lookup.setdefault(record.subject_id, []).append(
            {
                "subject_id": record.subject_id,
                "slice_index": record.slice_index,
                "t2f_path": record.modality_paths["t2f"],
            }
        )

    def _case_triplet(subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entries = sorted(subject_lookup[subject_id], key=lambda row: int(row["slice_index"]))
        center_index = len(entries) // 2
        slice_index = int(entries[center_index]["slice_index"])
        image_slice = _normalized_t2f_slice(entries[center_index], dataset_root)
        pred_volume = np.zeros((240, 240, 155), dtype=np.uint8)
        target_volume = np.zeros((240, 240, 155), dtype=np.uint8)
        return image_slice, target_volume[:, :, slice_index], pred_volume[:, :, slice_index]

    review_grid = build_review_grid(
        evaluation_dir / "review_grid.png",
        good_case=_case_triplet(best_case.subject_id),
        median_case=_case_triplet(median_case.subject_id),
        poor_case=_case_triplet(worst_case.subject_id),
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
        "review_grid": str(review_grid).replace("\\", "/"),
        "subject_metrics_csv": str(subject_metrics_csv).replace("\\", "/"),
    }
    write_json(evaluation_dir / "summary.json", payload)
    print("segmentation_test_summary=")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
