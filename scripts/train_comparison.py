"""Train comparison architectures against the frozen ResNet18 baseline recipe."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import json
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.comparison import (
    build_comparison_paths,
    build_experiment_name,
    create_comparison_config,
    save_json_artifact,
    sanitize_experiment_name,
)
from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.core.reproducibility import set_global_seed
from brainscan.data import CANONICAL_CLASS_TO_ID, create_train_val_dataloaders_from_config
from brainscan.evaluation.classification import benchmark_model_inference
from brainscan.models import (
    build_classifier,
    count_trainable_parameters,
    get_model_metadata,
    get_pretrained_weights_name,
)
from brainscan.training import (
    build_checkpoint_metadata,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    fit_classifier,
    load_checkpoint,
    load_dataset_fingerprint,
    reload_and_validate_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a comparison architecture for BrainScanAI.")
    parser.add_argument("--config", default="configs/train.yaml", help="Base training config.")
    parser.add_argument("--architecture", required=True, help="Architecture name.")
    parser.add_argument("--seed", type=int, required=True, help="Training seed.")
    parser.add_argument("--experiment-name", default=None, help="Optional explicit experiment name.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for this run.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional minor LR override.")
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision for numerically unstable architectures.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a short smoke test only.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional train batch cap.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional val batch cap.")
    return parser.parse_args()


def _estimate_checkpoint_size_mb(checkpoint_path: Path) -> float:
    resolved = resolve_project_path(checkpoint_path)
    return round(resolved.stat().st_size / (1024**2), 2)


def _memory_mb_from_bytes(total_bytes: int) -> float:
    return round(total_bytes / (1024**2), 2)


def main() -> None:
    args = parse_args()
    base_config = load_train_config(args.config)
    experiment_name = (
        sanitize_experiment_name(args.experiment_name)
        if args.experiment_name is not None
        else build_experiment_name(args.architecture, args.seed)
    )
    config = create_comparison_config(
        base_config,
        architecture=args.architecture,
        seed=args.seed,
        experiment_name=experiment_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    paths = build_comparison_paths(experiment_name)
    if args.disable_amp:
        config["training"]["mixed_precision"] = False

    set_global_seed(int(config["training"]["seed"]))
    device, device_info = resolve_device(prefer_cuda=True)
    print(f"device={device}")
    print(f"device_info={device_info}")

    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("CUDA is unavailable. Full comparison training must not fall back to a long CPU run.")

    dataloaders = create_train_val_dataloaders_from_config(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
    )
    model.to(device)

    model_metadata = get_model_metadata(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
        image_size=config["training"]["image_size"],
        class_mapping=dict(CANONICAL_CLASS_TO_ID),
        model=model,
    )

    criterion = build_loss_function()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    dataset_fingerprint = load_dataset_fingerprint()
    checkpoint_metadata = build_checkpoint_metadata(
        config=config,
        weights_name=get_pretrained_weights_name(
            str(config["model"]["architecture"]),
            bool(config["model"]["pretrained"]),
        ),
        dataset_fingerprint=dataset_fingerprint,
    )
    checkpoint_metadata["experiment_name"] = experiment_name
    checkpoint_metadata["model_metadata"] = model_metadata

    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    if args.smoke_test:
        config["training"]["epochs"] = min(int(config["training"]["epochs"]), 2)
        max_train_batches = max_train_batches or 2
        max_val_batches = max_val_batches or 2

    print(
        "comparison_training_setup="
        f"experiment_name={experiment_name} "
        f"architecture={config['model']['architecture']} "
        f"trainable_parameters={count_trainable_parameters(model)} "
        f"image_size={config['training']['image_size']} "
        f"batch_size={config['training']['batch_size']} "
        f"mixed_precision={config['training']['mixed_precision']} "
        f"classes={list(CANONICAL_CLASS_TO_ID.keys())}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    results = fit_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_metadata=checkpoint_metadata,
        best_checkpoint_path=paths["best_checkpoint"],
        last_checkpoint_path=paths["last_checkpoint"],
        history_json_path=paths["history_json"],
        history_csv_path=paths["history_csv"],
        curve_output_dir=paths["training_dir"],
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    train_seconds = time.perf_counter() - train_start
    peak_vram_mb = (
        _memory_mb_from_bytes(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    sample_batch = next(iter(val_loader))
    checkpoint_load_start = time.perf_counter()
    reloaded_model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    _ = load_checkpoint(paths["best_checkpoint"], reloaded_model, map_location=device)
    checkpoint_load_time_ms = (time.perf_counter() - checkpoint_load_start) * 1000.0

    validation_result = reload_and_validate_checkpoint(
        results["best_checkpoint_path"],
        model=build_classifier(
            architecture=str(config["model"]["architecture"]),
            num_classes=int(config["model"]["num_classes"]),
            pretrained=False,
        ),
        device=device,
        sample_batch=sample_batch,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    benchmark = benchmark_model_inference(reloaded_model.to(device), sample_batch[0], device)
    peak_benchmark_vram_mb = (
        _memory_mb_from_bytes(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    benchmark["checkpoint_load_time_ms"] = checkpoint_load_time_ms

    best_checkpoint_mb = _estimate_checkpoint_size_mb(paths["best_checkpoint"])
    summary_payload = {
        "experiment_name": experiment_name,
        "architecture": str(config["model"]["architecture"]),
        "seed": int(config["training"]["seed"]),
        "batch_size": int(config["training"]["batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "mixed_precision": bool(config["training"]["mixed_precision"]),
        "dataset_fingerprint": dataset_fingerprint["dataset_fingerprint"],
        "model_metadata": model_metadata,
        "device_info": device_info,
        "best_epoch": results["best_epoch"],
        "best_validation_macro_f1": results["best_score"],
        "epochs_completed": results["epochs_completed"],
        "resumed_from_epoch": results.get("resumed_from_epoch"),
        "training_duration_seconds": train_seconds,
        "best_checkpoint_path": paths["best_checkpoint"].as_posix(),
        "last_checkpoint_path": paths["last_checkpoint"].as_posix(),
        "history_json_path": paths["history_json"].as_posix(),
        "history_csv_path": paths["history_csv"].as_posix(),
        "best_checkpoint_size_mb": best_checkpoint_mb,
        "final_history": results["history"][-1],
        "history": results["history"],
        "benchmark": benchmark,
        "peak_vram_mb": peak_vram_mb,
        "peak_benchmark_vram_mb": peak_benchmark_vram_mb,
        "reload_validation": validation_result,
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "smoke_test": bool(args.smoke_test),
    }
    summary_path = save_json_artifact(summary_payload, paths["summary_json"])

    smoke_payload = {
        "experiment_name": experiment_name,
        "architecture": str(config["model"]["architecture"]),
        "seed": int(config["training"]["seed"]),
        "batch_size": int(config["training"]["batch_size"]),
        "mixed_precision": bool(config["training"]["mixed_precision"]),
        "device_info": device_info,
        "benchmark": benchmark,
        "best_checkpoint_size_mb": best_checkpoint_mb,
        "training_duration_seconds": train_seconds,
        "peak_vram_mb": peak_vram_mb,
        "peak_benchmark_vram_mb": peak_benchmark_vram_mb,
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if args.smoke_test:
        save_json_artifact(smoke_payload, paths["smoke_json"])

    print("comparison_final_summary=")
    print(f"  experiment_name={experiment_name}")
    print(f"  architecture={config['model']['architecture']}")
    print(f"  best_epoch={results['best_epoch']}")
    print(f"  best_val_macro_f1={results['best_score']}")
    print(f"  epochs_completed={results['epochs_completed']}")
    print(f"  resumed_from_epoch={results.get('resumed_from_epoch')}")
    print(f"  training_duration_seconds={train_seconds:.2f}")
    print(f"  best_checkpoint={resolve_project_path(paths['best_checkpoint'])}")
    print(f"  best_checkpoint_size_mb={best_checkpoint_mb}")
    print(f"  peak_vram_mb={peak_vram_mb}")
    print(f"  peak_benchmark_vram_mb={peak_benchmark_vram_mb}")
    print(f"  summary_json={resolve_project_path(summary_path)}")
    print(f"  benchmark={benchmark}")


if __name__ == "__main__":
    main()
