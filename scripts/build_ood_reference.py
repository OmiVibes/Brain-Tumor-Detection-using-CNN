"""Build a frozen-train OOD reference and validation-derived robustness thresholds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch

from brainscan.core import load_best_validation_reference
from brainscan.core.config import load_train_config, make_project_relative_path, resolve_project_path
from brainscan.data import BrainMRIDataset, CANONICAL_CLASS_TO_ID
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


DEFAULT_CONFIG_PATH = Path("configs/train.yaml")
DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
DEFAULT_REFERENCE_METRICS_PATH = Path("artifacts/training/resnet18_baseline_history.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/robustness/resnet18_baseline")
DEFAULT_QUALITY_SOURCE_PATH = DEFAULT_OUTPUT_DIR / "quality_thresholds.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BrainScanAI OOD and quality artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Training-style config path.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Frozen checkpoint path.")
    parser.add_argument(
        "--reference-metrics-path",
        default=str(DEFAULT_REFERENCE_METRICS_PATH),
        help="History JSON or summary JSON used to verify best epoch and validation macro F1.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for OOD and quality artifacts.",
    )
    parser.add_argument(
        "--quality-thresholds-source",
        default=None,
        help="Optional existing quality_thresholds.json to reuse when thresholds are data-derived.",
    )
    parser.add_argument("--prefer-cuda", action="store_true", help="Prefer CUDA when available.")
    return parser.parse_args()


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


def _reuse_quality_thresholds(
    source_path: str | Path,
    *,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    architecture: str,
    dataset_fingerprint: str,
) -> dict[str, object]:
    source_payload = json.loads(resolve_project_path(source_path).read_text(encoding="utf-8"))
    if str(source_payload["dataset_fingerprint"]) != dataset_fingerprint:
        raise ValueError("Quality-threshold source artifact dataset fingerprint does not match the frozen checkpoint.")
    split_counts = dict(source_payload["derivation_split_counts"])
    if int(split_counts["train"]) <= 0 or int(split_counts["validation"]) <= 0:
        raise ValueError("Quality-threshold source artifact does not record valid train/validation derivation counts.")

    return {
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint": make_project_relative_path(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "architecture": architecture,
        "thresholds": dict(source_payload["thresholds"]),
        "derivation_split_counts": {
            "train": int(split_counts["train"]),
            "validation": int(split_counts["validation"]),
        },
        "derivation": "reused existing data-derived thresholds from train + validation image statistics",
        "reuse_note": (
            "Quality thresholds were reused because they depend only on train/validation image statistics, "
            "not on classifier architecture."
        ),
        "source_artifact": make_project_relative_path(resolve_project_path(source_path)),
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
    }


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    reference_metrics = load_best_validation_reference(args.reference_metrics_path)

    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(args.checkpoint_path, model, map_location="cpu")
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]
    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_architecture=str(config["model"]["architecture"]),
        expected_epoch=int(reference_metrics["best_epoch"]),
        expected_validation_score=float(reference_metrics["best_validation_macro_f1"]),
    )

    output_dir = Path(args.output_dir)
    ood_reference_npz_path = output_dir / "ood_reference.npz"
    ood_reference_json_path = output_dir / "ood_reference.json"
    quality_thresholds_path = output_dir / "quality_thresholds.json"

    device, _ = resolve_device(prefer_cuda=bool(args.prefer_cuda))
    model.to(device)
    model.eval()

    architecture = str(checkpoint_metadata["architecture"])
    train_loader = _build_split_loader(config, "train")
    val_loader = _build_split_loader(config, "val")
    train_bundle = extract_features_from_dataloader(
        model,
        train_loader,
        device,
        expected_split="train",
        architecture=architecture,
    )
    val_bundle = extract_features_from_dataloader(
        model,
        val_loader,
        device,
        expected_split="val",
        architecture=architecture,
    )

    reference, _validation_scores = build_diagonal_mahalanobis_reference(
        train_bundle["features"],  # type: ignore[arg-type]
        train_bundle["labels"],  # type: ignore[arg-type]
        val_bundle["features"],  # type: ignore[arg-type]
        feature_layer=str(train_bundle["feature_layer"]),
        threshold_quantile=0.95,
    )
    save_ood_reference(
        reference,
        npz_path=ood_reference_npz_path,
        json_path=ood_reference_json_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_epoch=int(checkpoint["epoch"]),
        dataset_fingerprint=dataset_fingerprint,
        architecture=architecture,
        class_mapping=dict(CANONICAL_CLASS_TO_ID),
    )

    if args.quality_thresholds_source:
        quality_payload = _reuse_quality_thresholds(
            args.quality_thresholds_source,
            checkpoint_path=Path(args.checkpoint_path),
            checkpoint_epoch=int(checkpoint["epoch"]),
            architecture=architecture,
            dataset_fingerprint=dataset_fingerprint,
        )
    else:
        quality_thresholds, quality_train_count, quality_val_count = _compute_quality_thresholds_from_splits(config)
        quality_payload = {
            "dataset_fingerprint": dataset_fingerprint,
            "checkpoint": make_project_relative_path(args.checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "architecture": architecture,
            "thresholds": quality_thresholds,
            "derivation_split_counts": {
                "train": quality_train_count,
                "validation": quality_val_count,
            },
            "derivation": "freshly derived from train + validation image statistics",
            "reuse_note": None,
            "source_artifact": None,
            "generated_timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
        }
    _save_json(quality_payload, quality_thresholds_path)

    print("ood_reference_summary=")
    print(f"  checkpoint={resolve_project_path(args.checkpoint_path)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  architecture={architecture}")
    print(f"  feature_layer={reference.feature_layer}")
    print(f"  feature_dimension={reference.feature_dimension}")
    print(f"  training_reference_sample_count={reference.training_sample_count}")
    print(f"  validation_threshold_sample_count={reference.validation_sample_count}")
    print(f"  ood_warning_threshold={reference.warning_threshold:.6f}")
    print(f"  warning_threshold_source={reference.warning_threshold_source}")
    print(f"  ood_threshold={reference.threshold:.6f}")
    print(f"  threshold_source={reference.threshold_source}")
    print(f"  quality_threshold_reuse={bool(args.quality_thresholds_source)}")
    print(f"  ood_reference_json={resolve_project_path(ood_reference_json_path)}")
    print(f"  quality_thresholds_json={resolve_project_path(quality_thresholds_path)}")


if __name__ == "__main__":
    main()
