"""Train the Phase 1D BrainScanAI baseline classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.core.reproducibility import set_global_seed
from brainscan.data import CANONICAL_CLASS_TO_ID, create_train_val_dataloaders
from brainscan.models import build_classifier, count_trainable_parameters, get_pretrained_weights_name
from brainscan.training import (
    build_checkpoint_metadata,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    fit_classifier,
    load_dataset_fingerprint,
    reload_and_validate_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BrainScanAI Phase 1D baseline.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short 2-epoch smoke test.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional batch cap for training.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional batch cap for validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    set_global_seed(int(config["training"]["seed"]))

    device, device_info = resolve_device(prefer_cuda=True)
    print(f"device={device}")
    print(f"device_info={device_info}")

    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("CUDA is unavailable. Phase 1D full training must not fall back to a long CPU run.")

    dataloaders = create_train_val_dataloaders(args.config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = build_classifier(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
    )
    model.to(device)

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

    checkpoint_config = config["checkpoint"]
    best_checkpoint_path = Path(checkpoint_config["model_dir"]) / str(checkpoint_config["best_filename"])
    last_checkpoint_path = Path(checkpoint_config["model_dir"]) / str(checkpoint_config["last_filename"])
    history_json_path = Path(checkpoint_config["training_dir"]) / str(checkpoint_config["history_json"])
    history_csv_path = Path(checkpoint_config["training_dir"]) / str(checkpoint_config["history_csv"])

    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    if args.smoke_test:
        config["training"]["epochs"] = min(int(config["training"]["epochs"]), 2)
        max_train_batches = max_train_batches or 2
        max_val_batches = max_val_batches or 2

    print(
        "training_setup="
        f"architecture={config['model']['architecture']} "
        f"trainable_parameters={count_trainable_parameters(model)} "
        f"image_size={config['training']['image_size']} "
        f"batch_size={config['training']['batch_size']} "
        f"classes={list(CANONICAL_CLASS_TO_ID.keys())}"
    )

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
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        history_json_path=history_json_path,
        history_csv_path=history_csv_path,
        curve_output_dir=checkpoint_config["training_dir"],
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )

    sample_batch = next(iter(val_loader))
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

    final_history = results["history"][-1]
    print("final_summary=")
    print(f"  epochs_completed={results['epochs_completed']}")
    print(f"  best_epoch={results['best_epoch']}")
    print(f"  best_val_macro_f1={results['best_score']}")
    print(f"  final_train_loss={final_history['train_loss']}")
    print(f"  final_train_accuracy={final_history['train_accuracy']}")
    print(f"  final_val_loss={final_history['val_loss']}")
    print(f"  final_val_accuracy={final_history['val_accuracy']}")
    print(f"  final_val_macro_f1={final_history['val_f1_macro']}")
    print(f"  best_checkpoint={resolve_project_path(results['best_checkpoint_path'])}")
    print(f"  last_checkpoint={resolve_project_path(results['last_checkpoint_path'])}")
    print(f"  history_json={resolve_project_path(results['history_json_path'])}")
    print(f"  history_csv={resolve_project_path(results['history_csv_path'])}")
    print(f"  curve_paths={[str(path) for path in results['curve_paths']]}")
    print(f"  checkpoint_reload={validation_result}")


if __name__ == "__main__":
    main()
