"""Build a frozen-train OOD reference and validation-derived robustness thresholds."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch

from brainscan.core.config import load_train_config, make_project_relative_path, resolve_project_path
from brainscan.data import BrainMRIDataset, CANONICAL_CLASS_NAMES, CANONICAL_CLASS_TO_ID
from brainscan.data.preprocessing import build_eval_transform
from brainscan.evaluation.classification import verify_checkpoint_metadata
from brainscan.models import build_classifier
from brainscan.robustness import (
    build_diagonal_mahalanobis_reference,
    compute_quality_metrics,
    derive_quality_thresholds,
    extract_features_from_dataloader,
    save_ood_reference,
)
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device
from brainscan.training.train_classifier import get_git_commit


CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
TRAINING_HISTORY_PATH = Path("artifacts/training/resnet18_baseline_history.json")
ROBUSTNESS_DIR = Path("artifacts/robustness/resnet18_baseline")
OOD_REFERENCE_NPZ_PATH = ROBUSTNESS_DIR / "ood_reference.npz"
OOD_REFERENCE_JSON_PATH = ROBUSTNESS_DIR / "ood_reference.json"
QUALITY_THRESHOLDS_PATH = ROBUSTNESS_DIR / "quality_thresholds.json"


def _save_json(payload: object, path: Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved


def _build_split_loader(config: dict[str, object], split_name: str) -> torch.utils.data.DataLoader:
    split_manifest_dir = Path(config["dataset"]["split_manifest_dir"])
    dataset = BrainMRIDataset(
        manifest_path=split_manifest_dir / f"{split_name}.csv",
        transform=build_eval_transform(config["training"]["image_size"]),
        return_metadata=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )


def _compute_quality_thresholds_from_splits(config: dict[str, object]) -> tuple[dict[str, float | str], int, int]:
    split_manifest_dir = Path(config["dataset"]["split_manifest_dir"])
    metrics_rows = []
    split_counts = {"train": 0, "val": 0}
    for split_name in ("train", "val"):
        dataset = BrainMRIDataset(
            manifest_path=split_manifest_dir / f"{split_name}.csv",
            transform=build_eval_transform(config["training"]["image_size"]),
            return_metadata=False,
        )
        split_counts[split_name] = len(dataset.records)
        for record in dataset.records:
            image_path = resolve_project_path(record.relative_path)
            from PIL import Image

            with Image.open(image_path) as image:
                metrics_rows.append(compute_quality_metrics(image.convert("RGB")))
    return derive_quality_thresholds(metrics_rows), split_counts["train"], split_counts["val"]


def main() -> None:
    config = load_train_config("configs/train.yaml")
    history = json.loads(resolve_project_path(TRAINING_HISTORY_PATH).read_text(encoding="utf-8"))
    best_history_entry = next(entry for entry in history if int(entry["epoch"]) == 9)

    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(CHECKPOINT_PATH, model, map_location="cpu")
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]
    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_validation_score=float(best_history_entry["val_macro_f1"]),
    )

    device, _ = resolve_device(prefer_cuda=True)
    model.to(device)
    model.eval()

    train_loader = _build_split_loader(config, "train")
    val_loader = _build_split_loader(config, "val")
    train_bundle = extract_features_from_dataloader(model, train_loader, device, expected_split="train")
    val_bundle = extract_features_from_dataloader(model, val_loader, device, expected_split="val")

    reference, validation_scores = build_diagonal_mahalanobis_reference(
        train_bundle["features"],  # type: ignore[arg-type]
        train_bundle["labels"],  # type: ignore[arg-type]
        val_bundle["features"],  # type: ignore[arg-type]
        threshold_quantile=0.95,
    )
    save_ood_reference(
        reference,
        npz_path=OOD_REFERENCE_NPZ_PATH,
        json_path=OOD_REFERENCE_JSON_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_epoch=int(checkpoint["epoch"]),
        dataset_fingerprint=dataset_fingerprint,
        architecture=str(checkpoint_metadata["architecture"]),
        class_mapping=dict(CANONICAL_CLASS_TO_ID),
    )

    quality_thresholds, quality_train_count, quality_val_count = _compute_quality_thresholds_from_splits(config)
    _save_json(
        {
            "dataset_fingerprint": dataset_fingerprint,
            "checkpoint": make_project_relative_path(CHECKPOINT_PATH),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "architecture": str(checkpoint_metadata["architecture"]),
            "thresholds": quality_thresholds,
            "derivation_split_counts": {
                "train": quality_train_count,
                "validation": quality_val_count,
            },
            "generated_timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
        },
        QUALITY_THRESHOLDS_PATH,
    )

    print("ood_reference_summary=")
    print(f"  checkpoint={resolve_project_path(CHECKPOINT_PATH)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  feature_layer={reference.feature_layer}")
    print(f"  feature_dimension={reference.feature_dimension}")
    print(f"  training_reference_sample_count={reference.training_sample_count}")
    print(f"  validation_threshold_sample_count={reference.validation_sample_count}")
    print(f"  ood_warning_threshold={reference.warning_threshold:.6f}")
    print(f"  warning_threshold_source={reference.warning_threshold_source}")
    print(f"  ood_threshold={reference.threshold:.6f}")
    print(f"  threshold_source={reference.threshold_source}")
    print(f"  ood_reference_json={resolve_project_path(OOD_REFERENCE_JSON_PATH)}")
    print(f"  quality_thresholds_json={resolve_project_path(QUALITY_THRESHOLDS_PATH)}")


if __name__ == "__main__":
    main()
