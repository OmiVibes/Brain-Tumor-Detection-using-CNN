from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brainscan.training.train_classifier import (
    build_loss_function,
    build_optimizer,
    load_checkpoint,
    reload_and_validate_checkpoint,
    save_checkpoint,
    validate_one_epoch,
)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))


def _make_loader() -> DataLoader:
    inputs = torch.randn(8, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    return DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)


def _metadata() -> dict[str, object]:
    return {
        "architecture": "resnet18",
        "num_classes": 4,
        "canonical_class_mapping": {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3},
        "pretrained_weights": None,
        "image_size": [224, 224],
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "seed": 42,
        "training_configuration": {},
        "dataset_fingerprint": "abc123",
        "dataset_fingerprint_source": {"dataset_fingerprint": "abc123"},
        "git_commit": "deadbeef",
    }


def test_checkpoint_save_and_reload_preserves_predictions(tmp_path) -> None:
    model = _make_model()
    loader = _make_loader()
    optimizer = build_optimizer(
        model,
        {
            "optimizer": {"name": "adamw"},
            "training": {"learning_rate": 0.001, "weight_decay": 0.0},
        },
    )
    criterion = build_loss_function()
    batch = next(iter(loader))
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"loss": 1.0, "accuracy": 0.5},
        metadata=_metadata(),
    )

    with torch.no_grad():
        original_outputs = model(batch[0]).clone()

    reloaded_model = _make_model()
    load_checkpoint(checkpoint_path, reloaded_model)

    with torch.no_grad():
        reloaded_outputs = reloaded_model(batch[0])

    assert torch.allclose(original_outputs, reloaded_outputs)


def test_checkpoint_reload_validation_passes(tmp_path) -> None:
    model = _make_model()
    loader = _make_loader()
    optimizer = build_optimizer(
        model,
        {
            "optimizer": {"name": "adamw"},
            "training": {"learning_rate": 0.001, "weight_decay": 0.0},
        },
    )
    criterion = build_loss_function()
    validate_one_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        num_classes=4,
        amp_enabled=False,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"loss": 1.0, "accuracy": 0.5, "precision_macro": 0.5, "recall_macro": 0.5, "f1_macro": 0.5},
        metadata=_metadata(),
    )

    result = reload_and_validate_checkpoint(
        checkpoint_path,
        model=_make_model(),
        device=torch.device("cpu"),
        sample_batch=next(iter(loader)),
    )

    assert result["logits_shape"] == (4, 4)
    assert result["probability_sums_close_to_one"] is True
    assert result["prediction_range_valid"] is True
