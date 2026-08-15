"""Reusable training loop for BrainScanAI classifiers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from brainscan.core.config import resolve_project_path
from brainscan.data import CANONICAL_CLASS_TO_ID, IMAGENET_MEAN, IMAGENET_STD


matplotlib.use("Agg")


HistoryEntry = dict[str, float | int]


@dataclass
class EarlyStoppingMonitor:
    """Track improvements for checkpointing and early stopping."""

    mode: str = "max"
    patience: int = 5
    best_score: float | None = None
    best_epoch: int = -1
    bad_epochs: int = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        if self.mode == "min":
            return score < self.best_score
        raise ValueError(f"Unsupported monitor mode '{self.mode}'. Expected 'max' or 'min'.")

    def update(self, score: float, epoch: int) -> tuple[bool, bool]:
        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return False, should_stop


def resolve_device(prefer_cuda: bool = True) -> tuple[torch.device, dict[str, object]]:
    """Select a device and expose lightweight hardware metadata."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        info = {
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(device),
            "cuda_available": True,
            "cuda_runtime": torch.version.cuda,
            "vram_bytes": int(properties.total_memory),
            "vram_gb": round(properties.total_memory / (1024**3), 2),
        }
        return device, info

    return torch.device("cpu"), {
        "device_type": "cpu",
        "device_name": "CPU",
        "cuda_available": False,
        "cuda_runtime": torch.version.cuda,
        "vram_bytes": 0,
        "vram_gb": 0.0,
    }


def resolve_amp_enabled(requested: bool, device: torch.device) -> bool:
    """Enable AMP only when CUDA is active."""
    return bool(requested and device.type == "cuda")


def build_loss_function() -> nn.Module:
    """Return the baseline classification loss."""
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, config: dict[str, object]) -> Optimizer:
    """Create the configured optimizer."""
    optimizer_name = str(config["optimizer"]["name"]).lower()
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Phase 1D supports only AdamW.")

    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(optimizer: Optimizer, config: dict[str, object]):
    """Create the configured scheduler."""
    scheduler_config = config.get("scheduler", {})
    scheduler_name = str(scheduler_config.get("name", "none")).lower()

    if scheduler_name in {"none", ""}:
        return None

    if scheduler_name != "reduce_on_plateau":
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. Phase 1D supports only ReduceLROnPlateau."
        )

    return ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_config.get("mode", "max")),
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 2)),
        min_lr=float(scheduler_config.get("min_lr", 1e-6)),
    )


def compute_classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    num_classes: int,
) -> dict[str, float]:
    """Compute accuracy and macro metrics."""
    target_array = np.asarray(list(targets), dtype=np.int64)
    prediction_array = np.asarray(list(predictions), dtype=np.int64)

    if target_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty target set.")

    accuracy = float((target_array == prediction_array).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def _move_batch_to_device(batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, str]], device: torch.device):
    inputs = batch[0].to(device, non_blocking=True)
    labels = batch[1].to(device, non_blocking=True)
    return inputs, labels


def _current_learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _finalize_epoch(
    *,
    total_loss: float,
    sample_count: int,
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> dict[str, float]:
    metrics = compute_classification_metrics(targets, predictions, num_classes)
    metrics["loss"] = total_loss / max(sample_count, 1)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    num_classes: int,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training epoch and update model parameters."""
    model.train()
    total_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []
    grad_scaler = scaler or torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs, labels = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise ValueError(f"Encountered non-finite training loss: {loss.item()}")

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        total_loss += float(loss.item()) * labels.size(0)
        sample_count += int(labels.size(0))
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    metrics = _finalize_epoch(
        total_loss=total_loss,
        sample_count=sample_count,
        targets=targets,
        predictions=predictions,
        num_classes=num_classes,
    )
    metrics["learning_rate"] = _current_learning_rate(optimizer)
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    amp_enabled: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one validation epoch without parameter updates."""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs, labels = _move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise ValueError(f"Encountered non-finite validation loss: {loss.item()}")

        total_loss += float(loss.item()) * labels.size(0)
        sample_count += int(labels.size(0))
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return _finalize_epoch(
        total_loss=total_loss,
        sample_count=sample_count,
        targets=targets,
        predictions=predictions,
        num_classes=num_classes,
    )


def load_dataset_fingerprint(
    fingerprint_path: str | Path = "artifacts/data_audit/dataset_fingerprint.json",
) -> dict[str, object]:
    """Load the deterministic dataset fingerprint artifact."""
    resolved_path = resolve_project_path(fingerprint_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_git_commit() -> str | None:
    """Return the current git commit if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_checkpoint_metadata(
    *,
    config: dict[str, object],
    weights_name: str | None,
    dataset_fingerprint: dict[str, object],
) -> dict[str, object]:
    """Build checkpoint metadata required for reproducibility."""
    return {
        "architecture": config["model"]["architecture"],
        "num_classes": int(config["model"]["num_classes"]),
        "canonical_class_mapping": dict(CANONICAL_CLASS_TO_ID),
        "pretrained_weights": weights_name,
        "image_size": list(config["training"]["image_size"]),
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "seed": int(config["training"]["seed"]),
        "training_configuration": config,
        "dataset_fingerprint": dataset_fingerprint["dataset_fingerprint"],
        "dataset_fingerprint_source": dataset_fingerprint,
        "git_commit": get_git_commit(),
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict[str, object],
    scheduler=None,
) -> Path:
    """Persist a training checkpoint."""
    resolved_path = resolve_project_path(checkpoint_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "metadata": metadata,
    }
    torch.save(payload, resolved_path)
    return resolved_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load a checkpoint and optionally restore optimizer/scheduler state."""
    resolved_path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_training_history(
    history: list[HistoryEntry],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Save structured training history to JSON and CSV."""
    resolved_json = resolve_project_path(json_path)
    resolved_csv = resolve_project_path(csv_path)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_csv.parent.mkdir(parents=True, exist_ok=True)

    with resolved_json.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    fieldnames = list(history[0].keys()) if history else ["epoch"]
    with resolved_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return resolved_json, resolved_csv


def load_training_history(history_json_path: str | Path) -> list[HistoryEntry]:
    """Load persisted epoch history if it exists."""
    resolved_json = resolve_project_path(history_json_path)
    if not resolved_json.exists():
        return []
    with resolved_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Training history JSON must contain a list of epoch entries.")
    return payload


def plot_training_curves(history: list[HistoryEntry], output_dir: str | Path) -> list[Path]:
    """Plot loss, accuracy, and F1 curves from saved history."""
    if not history:
        return []

    resolved_dir = resolve_project_path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_accuracy = [float(row["train_accuracy"]) for row in history]
    val_accuracy = [float(row["val_accuracy"]) for row in history]
    val_f1 = [float(row["val_macro_f1"]) for row in history]

    figures = [
        (
            "loss_curve.png",
            "Loss",
            train_loss,
            val_loss,
            "train_loss",
            "val_loss",
        ),
        (
            "accuracy_curve.png",
            "Accuracy",
            train_accuracy,
            val_accuracy,
            "train_accuracy",
            "val_accuracy",
        ),
        (
            "f1_curve.png",
            "Macro F1",
            val_f1,
            None,
            "val_macro_f1",
            None,
        ),
    ]

    created_paths: list[Path] = []
    for filename, ylabel, first_series, second_series, first_label, second_label in figures:
        figure_path = resolved_dir / filename
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, first_series, label=first_label)
        if second_series is not None:
            plt.plot(epochs, second_series, label=second_label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"BrainScanAI {ylabel} Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()
        created_paths.append(figure_path)

    return created_paths


def check_model_parameters_finite(model: nn.Module) -> bool:
    """Return whether all trainable parameters are finite."""
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def fit_classifier(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler,
    device: torch.device,
    config: dict[str, object],
    checkpoint_metadata: dict[str, object],
    best_checkpoint_path: str | Path,
    last_checkpoint_path: str | Path,
    history_json_path: str | Path,
    history_csv_path: str | Path,
    curve_output_dir: str | Path,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, object]:
    """Train a classifier and track the best validation checkpoint."""
    monitor_name = str(config["checkpoint"]["monitor"])
    monitor_mode = str(config["checkpoint"]["mode"])
    max_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    amp_enabled = resolve_amp_enabled(bool(config["training"]["mixed_precision"]), device)
    num_classes = int(config["model"]["num_classes"])
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    early_stopping = EarlyStoppingMonitor(mode=monitor_mode, patience=patience)
    history: list[HistoryEntry] = []
    resumed_from_epoch: int | None = None
    best_checkpoint_resolved = resolve_project_path(best_checkpoint_path)
    last_checkpoint_resolved = resolve_project_path(last_checkpoint_path)
    history_json_resolved = resolve_project_path(history_json_path)

    if last_checkpoint_resolved.exists() and history_json_resolved.exists():
        checkpoint = load_checkpoint(
            last_checkpoint_resolved,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        history = load_training_history(history_json_resolved)
        if history and int(history[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise ValueError(
                "Persisted history and last checkpoint are out of sync; refusing automatic resume."
            )
        if not best_checkpoint_resolved.exists():
            raise FileNotFoundError(
                f"Best checkpoint missing for resumable run: {best_checkpoint_resolved}"
            )
        for history_entry in history:
            early_stopping.update(float(history_entry[monitor_name]), int(history_entry["epoch"]))
        resumed_from_epoch = int(checkpoint["epoch"])
        print(
            f"Resuming training from epoch {resumed_from_epoch + 1} "
            f"using {last_checkpoint_resolved}."
        )
    elif last_checkpoint_resolved.exists() or history_json_resolved.exists():
        print(
            "Detected incomplete resume artifacts without both last checkpoint and history; "
            "starting this run from scratch and overwriting stale partial outputs."
        )

    start_epoch = resumed_from_epoch + 1 if resumed_from_epoch is not None else 1
    if start_epoch > max_epochs or early_stopping.bad_epochs >= patience:
        history_json, history_csv = save_training_history(
            history,
            json_path=history_json_path,
            csv_path=history_csv_path,
        )
        curve_paths = plot_training_curves(history, curve_output_dir)
        return {
            "history": history,
            "history_json_path": history_json,
            "history_csv_path": history_csv,
            "curve_paths": curve_paths,
            "best_checkpoint_path": best_checkpoint_resolved,
            "last_checkpoint_path": last_checkpoint_resolved,
            "best_epoch": early_stopping.best_epoch,
            "best_score": early_stopping.best_score,
            "epochs_completed": len(history),
            "amp_enabled": amp_enabled,
            "resumed_from_epoch": resumed_from_epoch,
        }

    for epoch in range(start_epoch, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
            scaler=scaler,
            max_batches=max_train_batches,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
            max_batches=max_val_batches,
        )

        history_entry: HistoryEntry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_macro_f1": val_metrics["f1_macro"],
            "learning_rate": train_metrics["learning_rate"],
        }
        history.append(history_entry)
        save_training_history(
            history,
            json_path=history_json_path,
            csv_path=history_csv_path,
        )

        current_score = float(history_entry[monitor_name])
        improved, should_stop = early_stopping.update(current_score, epoch)

        if scheduler is not None:
            scheduler.step(current_score if monitor_name == "val_macro_f1" else history_entry["val_loss"])

        checkpoint_metadata_with_epoch = {
            **checkpoint_metadata,
            "epoch": epoch,
            "validation_score": current_score,
        }

        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            metadata=checkpoint_metadata_with_epoch,
        )

        if improved:
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                metadata=checkpoint_metadata_with_epoch,
            )

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"lr={train_metrics['learning_rate']:.6f}"
        )

        if should_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"after {early_stopping.bad_epochs} non-improving epoch(s)."
            )
            break

    history_json, history_csv = save_training_history(
        history,
        json_path=history_json_path,
        csv_path=history_csv_path,
    )
    curve_paths = plot_training_curves(history, curve_output_dir)

    return {
        "history": history,
        "history_json_path": history_json,
        "history_csv_path": history_csv,
        "curve_paths": curve_paths,
        "best_checkpoint_path": resolve_project_path(best_checkpoint_path),
        "last_checkpoint_path": resolve_project_path(last_checkpoint_path),
        "best_epoch": early_stopping.best_epoch,
        "best_score": early_stopping.best_score,
        "epochs_completed": len(history),
        "amp_enabled": amp_enabled,
        "resumed_from_epoch": resumed_from_epoch,
    }


@torch.no_grad()
def reload_and_validate_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    device: torch.device,
    sample_batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, str]],
) -> dict[str, object]:
    """Reload a checkpoint and verify prediction sanity on one batch."""
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()

    inputs, _ = _move_batch_to_device(sample_batch, device)
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=1)
    prediction_ids = logits.argmax(dim=1)

    probability_sums = probabilities.sum(dim=1)
    if not torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-4):
        raise ValueError("Reloaded checkpoint produced invalid probability sums.")

    if prediction_ids.min().item() < 0 or prediction_ids.max().item() >= logits.shape[1]:
        raise ValueError("Reloaded checkpoint produced out-of-range class IDs.")

    metadata = checkpoint["metadata"]
    if metadata["canonical_class_mapping"] != dict(CANONICAL_CLASS_TO_ID):
        raise ValueError("Checkpoint class mapping does not match canonical constants.")

    if not check_model_parameters_finite(model):
        raise ValueError("Checkpoint reload detected non-finite model parameters.")

    return {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_metric_keys": sorted(checkpoint["metrics"].keys()),
        "logits_shape": tuple(logits.shape),
        "probability_sums_close_to_one": True,
        "prediction_range_valid": True,
    }
