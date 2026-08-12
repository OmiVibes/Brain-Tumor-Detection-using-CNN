from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brainscan.training.train_classifier import (
    EarlyStoppingMonitor,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    compute_classification_metrics,
    resolve_amp_enabled,
    train_one_epoch,
    validate_one_epoch,
)


def _make_loader(batch_size: int = 4) -> DataLoader:
    inputs = torch.randn(12, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.ReLU(), nn.Linear(16, 4))


def _make_config() -> dict[str, object]:
    return {
        "optimizer": {"name": "adamw"},
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "mixed_precision": True,
        },
        "scheduler": {
            "name": "reduce_on_plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
            "min_lr": 1e-6,
        },
    }


def test_training_step_updates_parameters() -> None:
    model = _make_model()
    loader = _make_loader()
    optimizer = build_optimizer(model, _make_config())
    criterion = build_loss_function()
    before = deepcopy(model.state_dict())

    metrics = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        num_classes=4,
        amp_enabled=False,
    )

    after = model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_validation_step_does_not_update_parameters() -> None:
    model = _make_model()
    loader = _make_loader()
    criterion = build_loss_function()
    before = deepcopy(model.state_dict())

    metrics = validate_one_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        num_classes=4,
        amp_enabled=False,
    )

    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_metric_calculation_matches_known_labels() -> None:
    metrics = compute_classification_metrics(
        targets=[0, 1, 2, 3],
        predictions=[0, 1, 2, 0],
        num_classes=4,
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["precision_macro"] == 0.625
    assert metrics["recall_macro"] == 0.75
    assert metrics["f1_macro"] == 2.0 / 3.0


def test_best_checkpoint_monitor_and_early_stopping_behavior() -> None:
    monitor = EarlyStoppingMonitor(mode="max", patience=2)
    assert monitor.update(0.4, epoch=1) == (True, False)
    assert monitor.update(0.5, epoch=2) == (True, False)
    assert monitor.update(0.49, epoch=3) == (False, False)
    assert monitor.update(0.48, epoch=4) == (False, True)
    assert monitor.best_epoch == 2
    assert monitor.best_score == 0.5


def test_scheduler_builds_and_steps() -> None:
    model = _make_model()
    optimizer = build_optimizer(model, _make_config())
    scheduler = build_scheduler(optimizer, _make_config())
    assert scheduler is not None
    scheduler.step(0.5)


def test_amp_path_is_disabled_on_cpu() -> None:
    assert resolve_amp_enabled(requested=True, device=torch.device("cpu")) is False


def test_training_api_does_not_take_test_loader() -> None:
    import inspect

    signature = inspect.signature(train_one_epoch)
    assert "test_loader" not in signature.parameters
