"""Training loop and checkpoint orchestration for 2D whole-tumor segmentation."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from pathlib import Path
import pickle
import random
import csv
import json
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from brainscan.core.config import make_project_relative_path, resolve_project_path
from brainscan.core.system_metrics import capture_memory_snapshot
from brainscan.segmentation.training.evaluation import evaluate_segmentation_loader, write_json
from brainscan.training import EarlyStoppingMonitor, load_checkpoint, resolve_amp_enabled, save_checkpoint


HistoryEntry = dict[str, float | int]


def _resolve_training_state_path(history_json_path: str | Path) -> Path:
    resolved_history = resolve_project_path(history_json_path)
    return resolved_history.with_name("training_state.json")


def _resolve_recovery_path(history_json_path: str | Path) -> Path:
    resolved_history = resolve_project_path(history_json_path)
    return resolved_history.with_name("recovery.pt")


def save_training_history(
    history: list[HistoryEntry],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
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
    resolved_json = resolve_project_path(history_json_path)
    if not resolved_json.exists():
        return []
    with resolved_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Training history JSON must contain a list.")
    return payload


def save_training_state(history_json_path: str | Path, payload: dict[str, object]) -> Path:
    resolved = _resolve_training_state_path(history_json_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def load_training_state(history_json_path: str | Path) -> dict[str, object]:
    resolved = _resolve_training_state_path(history_json_path)
    if not resolved.exists():
        return {
            "cumulative_training_duration_seconds": 0.0,
            "last_completed_epoch": 0,
            "early_stopping": {"best_score": None, "best_epoch": -1, "bad_epochs": 0},
        }
    return json.loads(resolved.read_text(encoding="utf-8"))


def _load_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    resolved_path = resolve_project_path(checkpoint_path)
    return torch.load(resolved_path, map_location=map_location, weights_only=False)


def save_recovery_checkpoint(
    recovery_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    epoch: int,
    next_batch_index: int,
    history: list[HistoryEntry],
    best_score: float | None,
    best_epoch: int,
    bad_epochs: int,
    cumulative_training_duration_seconds: float,
    peak_vram_bytes: int,
    epoch_random_state: dict[str, object],
    metadata: dict[str, object],
) -> Path:
    resolved_path = resolve_project_path(recovery_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "history": history,
        "best_score": None if best_score is None else float(best_score),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "cumulative_training_duration_seconds": float(cumulative_training_duration_seconds),
        "peak_vram_bytes": int(peak_vram_bytes),
        "epoch_random_state": epoch_random_state,
        "metadata": metadata,
    }
    torch.save(payload, resolved_path)
    return resolved_path


def load_recovery_checkpoint(
    recovery_path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer,
    scheduler,
    map_location: str | torch.device,
) -> dict[str, object]:
    resolved_path = resolve_project_path(recovery_path)
    checkpoint = _load_checkpoint_payload(resolved_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def _encode_state(state: object) -> str:
    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def _decode_state(value: str) -> object:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def capture_random_state(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {
        "python": _encode_state(random.getstate()),
        "numpy": _encode_state(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().tolist(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["torch_cuda"] = [state.tolist() for state in torch.cuda.get_rng_state_all()]
    return payload


def restore_random_state(payload: dict[str, object] | None, device: torch.device) -> None:
    if not payload:
        return
    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    torch_cpu_state = payload.get("torch_cpu")
    if isinstance(python_state, str):
        random.setstate(_decode_state(python_state))
    if isinstance(numpy_state, str):
        np.random.set_state(_decode_state(numpy_state))
    if isinstance(torch_cpu_state, list):
        torch.set_rng_state(torch.tensor(torch_cpu_state, dtype=torch.uint8))
    torch_cuda_state = payload.get("torch_cuda")
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
        and isinstance(torch_cuda_state, list)
        and torch_cuda_state
    ):
        torch.cuda.set_rng_state_all([torch.tensor(state, dtype=torch.uint8) for state in torch_cuda_state])


def build_optimizer(model: nn.Module, config: dict[str, object]) -> Optimizer:
    return AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def build_scheduler(optimizer: Optimizer):
    return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _portable_value(value):
    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return make_project_relative_path(candidate)
            except ValueError:
                return value.replace("\\", "/")
        return value.replace("\\", "/")
    return value


def _portable_path(path_value: str | Path, *, relative_to: str | Path | None = None) -> str:
    resolved = resolve_project_path(path_value)
    try:
        return make_project_relative_path(resolved)
    except ValueError:
        if relative_to is not None:
            base_path = resolve_project_path(relative_to)
            if base_path.suffix:
                base_path = base_path.parent
            try:
                return str(resolved.relative_to(base_path)).replace("\\", "/")
            except ValueError:
                try:
                    return str(Path(resolved).relative_to(base_path)).replace("\\", "/")
                except ValueError:
                    return str(Path(resolved).relative_to(base_path.anchor)).replace("\\", "/")
        return str(resolved).replace("\\", "/")


def build_checkpoint_metadata(
    *,
    config: dict[str, object],
    dataset_fingerprint: str,
    train_subject_count: int,
    val_subject_count: int,
) -> dict[str, object]:
    return {
        "architecture": str(config["model"]["architecture"]),
        "input_modalities": list(config["dataset"]["modalities"]),
        "input_channels": int(config["model"]["input_channels"]),
        "output_channels": int(config["model"]["output_channels"]),
        "binary_mask_rule": "mask > 0",
        "normalization": "zscore_nonzero_per_modality",
        "neighbor_policy": str(config["dataset"].get("neighbor_policy", "edge_replicate")),
        "slice_offsets": list(config["dataset"].get("slice_offsets", [0])),
        "sampling": {
            "positive_negative_ratio": float(config["sampling"]["positive_negative_ratio"]),
            "train_negative_sampling_seed": int(config["training"]["train_negative_sampling_seed"]),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "subject_split": {
            "train_subjects": train_subject_count,
            "val_subjects": val_subject_count,
        },
        "seed": int(config["training"]["seed"]),
        "run_type": "full_training",
        "test_used": False,
        "config": _portable_value(config),
        "git_commit": get_git_commit(),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
    epoch_index: int,
    start_batch_index: int = 0,
    max_batches: int | None = None,
    log_every_batches: int | None = None,
    checkpoint_every_batches: int | None = None,
    recovery_path: str | Path | None = None,
    history: list[HistoryEntry] | None = None,
    best_score: float | None = None,
    best_epoch: int = -1,
    bad_epochs: int = 0,
    cumulative_training_duration_seconds: float = 0.0,
    peak_vram_bytes: int = 0,
    checkpoint_metadata: dict[str, object] | None = None,
    scheduler=None,
    epoch_random_state: dict[str, object] | None = None,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    total_loss = 0.0
    batch_count = 0
    dice_values: list[float] = []
    max_memory_bytes = 0
    start_time = time.perf_counter()
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if batch_index < start_batch_index:
            continue
        inputs, targets = batch[:2]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Encountered non-finite segmentation training loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).to(dtype=torch.float32)
        intersection = float((predictions * targets).sum().item())
        denominator = float(predictions.sum().item() + targets.sum().item())
        dice_values.append((2.0 * intersection + 1e-6) / (denominator + 1e-6))
        total_loss += float(loss.item()) * int(inputs.shape[0])
        batch_count += int(inputs.shape[0])
        if device.type == "cuda":
            max_memory_bytes = max(max_memory_bytes, int(torch.cuda.max_memory_allocated(device)))
        completed_batches = batch_index + 1
        if log_every_batches and completed_batches % int(log_every_batches) == 0:
            snapshot = capture_memory_snapshot(device)
            print(
                f"[seg epoch {epoch_index:02d} batch {completed_batches}/{total_batches}] "
                f"loss={float(loss.item()):.4f} "
                f"elapsed_seconds={time.perf_counter() - start_time:.1f} "
                f"gpu_allocated_mb={snapshot.cuda_allocated_mb} "
                f"gpu_reserved_mb={snapshot.cuda_reserved_mb} "
                f"process_rss_mb={snapshot.process_rss_mb} "
                f"process_private_mb={snapshot.process_private_mb}"
            )
        if (
            checkpoint_every_batches
            and recovery_path is not None
            and completed_batches % int(checkpoint_every_batches) == 0
            and history is not None
            and checkpoint_metadata is not None
            and epoch_random_state is not None
        ):
            save_recovery_checkpoint(
                recovery_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                next_batch_index=completed_batches,
                history=history,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                cumulative_training_duration_seconds=cumulative_training_duration_seconds,
                peak_vram_bytes=max(peak_vram_bytes, max_memory_bytes),
                epoch_random_state=epoch_random_state,
                metadata=checkpoint_metadata,
            )
    return {
        "loss": float(total_loss / max(batch_count, 1)),
        "dice": _mean(dice_values),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "max_memory_bytes": max_memory_bytes,
    }


def fit_segmenter(
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
    model_selection_path: str | Path,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, object]:
    monitor_name = str(config["checkpoint"]["monitor"])
    monitor_mode = str(config["checkpoint"]["mode"])
    max_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    threshold = float(config["inference"]["threshold"])
    amp_enabled = resolve_amp_enabled(bool(config["training"]["mixed_precision"]), device)

    early_stopping = EarlyStoppingMonitor(mode=monitor_mode, patience=patience)
    history: list[HistoryEntry] = []
    resumed_from_epoch: int | None = None
    cumulative_training_duration_seconds = 0.0
    peak_vram_bytes = 0
    best_checkpoint_resolved = resolve_project_path(best_checkpoint_path)
    last_checkpoint_resolved = resolve_project_path(last_checkpoint_path)
    history_json_resolved = resolve_project_path(history_json_path)
    recovery_checkpoint_resolved = _resolve_recovery_path(history_json_path)
    log_every_batches = int(config["training"].get("log_every_batches", 0) or 0)
    checkpoint_every_batches = int(config["training"].get("checkpoint_every_batches", 0) or 0)
    requested_run_type = str(checkpoint_metadata.get("run_type", "full_training"))
    start_batch_index = 0

    if recovery_checkpoint_resolved.exists():
        recovery_payload = _load_checkpoint_payload(recovery_checkpoint_resolved, map_location=device)
        recovery_run_type = str(recovery_payload.get("metadata", {}).get("run_type", "unknown"))
        if recovery_run_type != requested_run_type:
            raise ValueError(
                "Found segmentation recovery checkpoint from a different run type; refusing automatic resume."
            )

    if last_checkpoint_resolved.exists() and history_json_resolved.exists():
        last_payload = _load_checkpoint_payload(last_checkpoint_resolved, map_location=device)
        last_run_type = str(last_payload.get("metadata", {}).get("run_type", "unknown"))
        if last_run_type != requested_run_type:
            raise ValueError(
                "Found segmentation checkpoint artifacts that are not marked for this run type; refusing resume."
            )
        checkpoint = load_checkpoint(
            last_checkpoint_resolved,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        history = load_training_history(history_json_resolved)
        if history and int(history[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise ValueError("Persisted segmentation history and checkpoint are out of sync.")
        if not best_checkpoint_resolved.exists():
            raise FileNotFoundError(f"Missing best segmentation checkpoint for resume: {best_checkpoint_resolved}")
        training_state = load_training_state(history_json_resolved)
        cumulative_training_duration_seconds = float(training_state.get("cumulative_training_duration_seconds", 0.0))
        peak_vram_bytes = int(training_state.get("peak_vram_bytes", 0))
        early_stopping_state = training_state.get("early_stopping", {})
        if isinstance(early_stopping_state, dict):
            early_stopping.best_score = (
                float(early_stopping_state["best_score"])
                if early_stopping_state.get("best_score") is not None
                else None
            )
            early_stopping.best_epoch = int(early_stopping_state.get("best_epoch", -1))
            early_stopping.bad_epochs = int(early_stopping_state.get("bad_epochs", 0))
        else:
            for history_entry in history:
                early_stopping.update(float(history_entry[monitor_name]), int(history_entry["epoch"]))
        restore_random_state(training_state.get("random_state"), device)
        resumed_from_epoch = int(checkpoint["epoch"])
        print(f"Resuming segmentation training from epoch {resumed_from_epoch + 1} using {last_checkpoint_resolved}.")

    if recovery_checkpoint_resolved.exists():
        recovery_checkpoint = load_recovery_checkpoint(
            recovery_checkpoint_resolved,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        history = list(recovery_checkpoint.get("history", history))
        cumulative_training_duration_seconds = float(
            recovery_checkpoint.get("cumulative_training_duration_seconds", cumulative_training_duration_seconds)
        )
        peak_vram_bytes = int(recovery_checkpoint.get("peak_vram_bytes", peak_vram_bytes))
        early_stopping.best_score = (
            float(recovery_checkpoint["best_score"])
            if recovery_checkpoint.get("best_score") is not None
            else early_stopping.best_score
        )
        early_stopping.best_epoch = int(recovery_checkpoint.get("best_epoch", early_stopping.best_epoch))
        early_stopping.bad_epochs = int(recovery_checkpoint.get("bad_epochs", early_stopping.bad_epochs))
        resumed_from_epoch = int(recovery_checkpoint["epoch"]) - 1
        start_batch_index = int(recovery_checkpoint.get("next_batch_index", 0))
        restore_random_state(recovery_checkpoint.get("epoch_random_state"), device)
        print(
            f"Resuming segmentation epoch {int(recovery_checkpoint['epoch'])} "
            f"from batch {start_batch_index} using {recovery_checkpoint_resolved}."
        )

    start_epoch = resumed_from_epoch + 1 if resumed_from_epoch is not None else 1
    if start_epoch > max_epochs or early_stopping.bad_epochs >= patience:
        return {
            "history": history,
            "best_epoch": early_stopping.best_epoch,
            "best_score": early_stopping.best_score,
            "epochs_completed": len(history),
            "amp_enabled": amp_enabled,
            "peak_vram_bytes": peak_vram_bytes,
            "best_checkpoint_path": resolve_project_path(best_checkpoint_path),
            "last_checkpoint_path": resolve_project_path(last_checkpoint_path),
            "history_json_path": resolve_project_path(history_json_path),
            "history_csv_path": resolve_project_path(history_csv_path),
            "resumed_from_epoch": resumed_from_epoch,
            "cumulative_training_duration_seconds": cumulative_training_duration_seconds,
        }

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.perf_counter()
        epoch_random_state = capture_random_state(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
            threshold=threshold,
            epoch_index=epoch,
            start_batch_index=start_batch_index if epoch == start_epoch else 0,
            max_batches=max_train_batches,
            log_every_batches=log_every_batches,
            checkpoint_every_batches=checkpoint_every_batches,
            recovery_path=recovery_checkpoint_resolved,
            history=history,
            best_score=early_stopping.best_score,
            best_epoch=early_stopping.best_epoch,
            bad_epochs=early_stopping.bad_epochs,
            cumulative_training_duration_seconds=cumulative_training_duration_seconds,
            peak_vram_bytes=peak_vram_bytes,
            checkpoint_metadata=checkpoint_metadata,
            scheduler=scheduler,
            epoch_random_state=epoch_random_state,
        )
        start_batch_index = 0
        val_metrics = evaluate_segmentation_loader(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            threshold=threshold,
            amp_enabled=amp_enabled,
            max_batches=max_val_batches,
        )
        epoch_duration = time.perf_counter() - epoch_start
        cumulative_training_duration_seconds += epoch_duration
        peak_vram_bytes = max(peak_vram_bytes, int(train_metrics["max_memory_bytes"]))

        history_entry: HistoryEntry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": float(val_metrics["loss"] or 0.0),
            "val_slice_dice": float(val_metrics["slice_dice"]),
            "val_subject_dice": float(val_metrics["subject_dice"]),
            "val_iou": float(val_metrics["subject_iou"]),
            "learning_rate": train_metrics["learning_rate"],
        }
        history.append(history_entry)
        save_training_history(history, json_path=history_json_path, csv_path=history_csv_path)
        save_training_state(
            history_json_path,
            {
                "cumulative_training_duration_seconds": cumulative_training_duration_seconds,
                "last_completed_epoch": epoch,
                "peak_vram_bytes": peak_vram_bytes,
                "early_stopping": {
                    "best_score": early_stopping.best_score,
                    "best_epoch": early_stopping.best_epoch,
                    "bad_epochs": early_stopping.bad_epochs,
                },
                "random_state": capture_random_state(device),
            },
        )
        if recovery_checkpoint_resolved.exists():
            recovery_checkpoint_resolved.unlink()

        current_score = float(history_entry[monitor_name])
        improved, should_stop = early_stopping.update(current_score, epoch)
        if scheduler is not None:
            scheduler.step(current_score)

        checkpoint_metadata_with_epoch = {
            **checkpoint_metadata,
            "epoch": epoch,
            "best_validation_subject_dice_so_far": float(early_stopping.best_score or current_score),
            "threshold": threshold,
        }
        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={key: value for key, value in val_metrics.items() if key != "subject_metrics"},
            metadata=checkpoint_metadata_with_epoch,
        )
        if improved:
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={key: value for key, value in val_metrics.items() if key != "subject_metrics"},
                metadata=checkpoint_metadata_with_epoch,
            )
            write_json(
                model_selection_path,
                {
                    "selected_checkpoint": _portable_path(
                        best_checkpoint_path,
                        relative_to=model_selection_path,
                    ),
                    "best_epoch": epoch,
                    "validation_metrics": {
                        "subject_dice": float(val_metrics["subject_dice"]),
                        "subject_iou": float(val_metrics["subject_iou"]),
                        "slice_dice": float(val_metrics["slice_dice"]),
                    },
                    "dataset_fingerprint": checkpoint_metadata["dataset_fingerprint"],
                    "architecture": checkpoint_metadata["architecture"],
                    "input_channels": checkpoint_metadata["input_channels"],
                    "neighbor_policy": checkpoint_metadata["neighbor_policy"],
                    "slice_offsets": checkpoint_metadata["slice_offsets"],
                    "threshold": threshold,
                    "normalization": checkpoint_metadata["normalization"],
                    "modalities": checkpoint_metadata["input_modalities"],
                    "binary_mask_rule": checkpoint_metadata["binary_mask_rule"],
                    "sampling": checkpoint_metadata["sampling"],
                    "test_used": checkpoint_metadata["test_used"],
                    "git_commit": checkpoint_metadata["git_commit"],
                },
            )

        print(
            f"[seg epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={float(val_metrics['loss'] or 0.0):.4f} "
            f"val_slice_dice={float(val_metrics['slice_dice']):.4f} "
            f"val_subject_dice={float(val_metrics['subject_dice']):.4f} "
            f"val_iou={float(val_metrics['subject_iou']):.4f} "
            f"lr={train_metrics['learning_rate']:.6f} "
            f"test_used=false"
        )
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    return {
        "history": history,
        "best_epoch": early_stopping.best_epoch,
        "best_score": early_stopping.best_score,
        "epochs_completed": len(history),
        "amp_enabled": amp_enabled,
        "peak_vram_bytes": peak_vram_bytes,
        "best_checkpoint_path": resolve_project_path(best_checkpoint_path),
        "last_checkpoint_path": resolve_project_path(last_checkpoint_path),
        "history_json_path": resolve_project_path(history_json_path),
        "history_csv_path": resolve_project_path(history_csv_path),
        "resumed_from_epoch": resumed_from_epoch,
        "cumulative_training_duration_seconds": cumulative_training_duration_seconds,
    }
