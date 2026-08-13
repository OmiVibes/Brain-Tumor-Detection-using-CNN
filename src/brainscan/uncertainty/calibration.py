"""Temperature scaling and logits collection for BrainScanAI uncertainty analysis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from brainscan.core.config import make_project_relative_path, resolve_project_path


class TemperatureScaler(nn.Module):
    """Single-scalar temperature scaling for frozen multiclass logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive.")
        log_temperature = torch.log(torch.tensor([initial_temperature], dtype=torch.float32))
        self.log_temperature = nn.Parameter(log_temperature)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
        return logits / self.temperature.clamp_min(1e-6)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(logits), dim=1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        max_iterations: int = 100,
    ) -> dict[str, float]:
        """Fit the temperature on frozen validation logits by minimizing NLL."""
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"Expected 1D labels, got shape {tuple(labels.shape)}")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("Logits and labels must contain the same number of samples.")

        logits = logits.detach()
        labels = labels.detach()
        self.to(logits.device)

        nll_before = float(F.cross_entropy(logits, labels, reduction="mean").item())
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.1,
            max_iter=max_iterations,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self(logits), labels, reduction="mean")
            if not torch.isfinite(loss):
                raise ValueError("Encountered non-finite loss while fitting temperature scaling.")
            loss.backward()
            return loss

        optimizer.step(_closure)
        if float(self.temperature.item()) <= 0.0:
            raise ValueError("Fitted temperature is not strictly positive.")

        calibrated_logits = self(logits)
        nll_after = float(F.cross_entropy(calibrated_logits, labels, reduction="mean").item())
        return {
            "temperature": float(self.temperature.item()),
            "nll_before": nll_before,
            "nll_after": nll_after,
        }


def collect_logits_from_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    expected_split: str,
) -> dict[str, object]:
    """Collect frozen-model logits and labels from one manifest-backed split."""
    dataset = getattr(dataloader, "dataset", None)
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("Calibration logits collection requires a manifest-backed dataset with records.")

    observed_splits = {record.split for record in records}
    if observed_splits != {expected_split}:
        raise ValueError(
            f"Expected only '{expected_split}' records for calibration collection, found {sorted(observed_splits)}."
        )

    model.eval()
    logits_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    relative_paths: list[str] = []

    for batch in dataloader:
        if len(batch) != 3:
            raise ValueError("Calibration dataloader must yield image, label, metadata tuples.")
        inputs, labels, metadata = batch
        inputs = inputs.to(device, non_blocking=True)

        with torch.inference_mode():
            logits = model(inputs)

        logits_rows.append(logits.detach().cpu())
        labels_rows.append(labels.detach().cpu())
        relative_paths.extend(list(metadata["relative_path"]))

    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Duplicate relative_path values detected during logits collection.")

    return {
        "logits": torch.cat(logits_rows, dim=0),
        "labels": torch.cat(labels_rows, dim=0).to(dtype=torch.int64),
        "relative_paths": relative_paths,
        "split": expected_split,
    }


def fit_temperature_from_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    expected_split: str = "val",
    initial_temperature: float = 1.0,
    max_iterations: int = 100,
) -> tuple[TemperatureScaler, dict[str, object], dict[str, float]]:
    """Collect validation logits, fit a scalar temperature, and verify frozen model parameters."""
    before = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    logits_bundle = collect_logits_from_dataloader(
        model,
        dataloader,
        device,
        expected_split=expected_split,
    )
    scaler = TemperatureScaler(initial_temperature=initial_temperature)
    fit_result = scaler.fit(
        logits_bundle["logits"],  # type: ignore[arg-type]
        logits_bundle["labels"],  # type: ignore[arg-type]
        max_iterations=max_iterations,
    )

    after = model.state_dict()
    for name, previous_value in before.items():
        if not torch.equal(previous_value, after[name]):
            raise RuntimeError("Model parameters changed during temperature-scaling fitting.")

    return scaler, logits_bundle, fit_result


def save_temperature_artifact(payload: dict[str, object], output_path: str | Path) -> Path:
    """Persist a portable temperature-scaling artifact."""
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_path


def load_temperature_artifact(output_path: str | Path) -> dict[str, object]:
    """Load a temperature-scaling artifact from disk."""
    resolved_path = resolve_project_path(output_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_temperature_artifact_payload(
    *,
    temperature: float,
    checkpoint_path: str | Path,
    checkpoint_epoch: int,
    dataset_fingerprint: str,
    validation_metrics_before: dict[str, object],
    validation_metrics_after: dict[str, object],
    git_commit: str | None,
    generated_timestamp_utc: str,
    method: str = "temperature_scaling",
    fitting_split: str = "validation",
) -> dict[str, object]:
    """Build a compact portable artifact payload for the learned temperature."""
    return {
        "temperature": temperature,
        "fitting_split": fitting_split,
        "checkpoint": make_project_relative_path(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "dataset_fingerprint": dataset_fingerprint,
        "calibration_method": method,
        "validation_nll_before": validation_metrics_before["nll"],
        "validation_nll_after": validation_metrics_after["nll"],
        "validation_ece_before": validation_metrics_before["ece"],
        "validation_ece_after": validation_metrics_after["ece"],
        "validation_brier_before": validation_metrics_before["brier_score"],
        "validation_brier_after": validation_metrics_after["brier_score"],
        "git_commit": git_commit,
        "generated_timestamp_utc": generated_timestamp_utc,
    }
