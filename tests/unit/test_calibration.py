from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from brainscan.uncertainty.calibration import (
    TemperatureScaler,
    build_temperature_artifact_payload,
    collect_logits_from_dataloader,
    fit_temperature_from_dataloader,
    load_temperature_artifact,
    save_temperature_artifact,
)


class TinyCalibrationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class DummyManifestDataset(torch.utils.data.Dataset):
    def __init__(self, split: str) -> None:
        self.records = [type("Record", (), {"split": split})() for _ in range(4)]
        self._features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self._labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        return (
            self._features[index],
            int(self._labels[index]),
            {"relative_path": f"dataset/{index}.jpg"},
        )


def test_temperature_is_always_positive() -> None:
    scaler = TemperatureScaler(initial_temperature=1.5)
    assert float(scaler.temperature.item()) > 0.0


def test_temperature_one_preserves_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    scaler = TemperatureScaler(initial_temperature=1.0)
    assert torch.allclose(scaler(logits), logits)


def test_calibration_preserves_argmax() -> None:
    logits = torch.tensor([[4.0, 1.0, -1.0], [0.5, 2.5, 1.0]], dtype=torch.float32)
    scaler = TemperatureScaler(initial_temperature=2.0)
    assert torch.equal(logits.argmax(dim=1), scaler(logits).argmax(dim=1))


def test_calibrated_probabilities_sum_to_one() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    scaler = TemperatureScaler(initial_temperature=1.7)
    probabilities = scaler.probabilities(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6)


def test_collect_logits_from_dataloader_returns_expected_shapes() -> None:
    model = TinyCalibrationModel()
    loader = DataLoader(DummyManifestDataset(split="val"), batch_size=2, shuffle=False)
    bundle = collect_logits_from_dataloader(model, loader, torch.device("cpu"), expected_split="val")

    assert tuple(bundle["logits"].shape) == (4, 4)
    assert tuple(bundle["labels"].shape) == (4,)
    assert bundle["split"] == "val"
    assert len(bundle["relative_paths"]) == 4


def test_calibration_fitting_does_not_modify_model_parameters() -> None:
    model = TinyCalibrationModel()
    before = deepcopy(model.state_dict())
    loader = DataLoader(DummyManifestDataset(split="val"), batch_size=2, shuffle=False)

    scaler, bundle, fit_result = fit_temperature_from_dataloader(model, loader, torch.device("cpu"))

    assert float(scaler.temperature.item()) > 0.0
    assert bundle["split"] == "val"
    assert fit_result["nll_after"] <= fit_result["nll_before"] + 1e-6
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_calibration_only_consumes_validation_loader_during_fitting() -> None:
    model = TinyCalibrationModel()
    loader = DataLoader(DummyManifestDataset(split="test"), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="Expected only 'val' records"):
        fit_temperature_from_dataloader(model, loader, torch.device("cpu"), expected_split="val")


def test_test_data_is_not_used_during_fitting() -> None:
    model = TinyCalibrationModel()
    loader = DataLoader(DummyManifestDataset(split="test"), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="Expected only 'val' records"):
        collect_logits_from_dataloader(model, loader, torch.device("cpu"), expected_split="val")


def test_calibration_artifact_contains_dataset_fingerprint_and_no_absolute_paths(tmp_path) -> None:
    payload = build_temperature_artifact_payload(
        temperature=1.25,
        checkpoint_path="artifacts/models/resnet18_baseline_best.pt",
        checkpoint_epoch=9,
        dataset_fingerprint="fingerprint-123",
        validation_metrics_before={"nll": 0.5, "ece": 0.2, "brier_score": 0.3},
        validation_metrics_after={"nll": 0.4, "ece": 0.1, "brier_score": 0.2},
        git_commit="abc123",
        generated_timestamp_utc="2026-08-13T00:00:00+00:00",
    )
    path = tmp_path / "temperature.json"
    save_temperature_artifact(payload, path)
    loaded = load_temperature_artifact(path)

    assert loaded["dataset_fingerprint"] == "fingerprint-123"
    assert str(loaded["checkpoint"]) == "artifacts/models/resnet18_baseline_best.pt"
    assert ":\\" not in json_string(loaded)


def test_save_load_temperature_reproduces_calibrated_probabilities(tmp_path) -> None:
    scaler = TemperatureScaler(initial_temperature=2.5)
    logits = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
    path = tmp_path / "temperature.json"
    payload = {"temperature": float(scaler.temperature.item())}
    save_temperature_artifact(payload, path)
    loaded = load_temperature_artifact(path)
    restored = TemperatureScaler(initial_temperature=float(loaded["temperature"]))

    assert torch.allclose(scaler.probabilities(logits), restored.probabilities(logits), atol=1e-7)


def json_string(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload)
