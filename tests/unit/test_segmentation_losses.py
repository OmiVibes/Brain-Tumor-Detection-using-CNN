from __future__ import annotations

import torch

from brainscan.segmentation.training.losses import BCEDiceLoss, soft_dice_score, thresholded_binary_metrics


def test_bce_plus_dice_loss_is_finite() -> None:
    criterion = BCEDiceLoss()
    logits = torch.randn(2, 1, 8, 8)
    targets = torch.randint(0, 2, (2, 1, 8, 8), dtype=torch.float32)
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)


def test_perfect_prediction_has_near_one_dice() -> None:
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    dice = soft_dice_score(logits, targets)
    assert float(dice) > 0.99


def test_empty_prediction_behavior_stays_finite() -> None:
    probabilities = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    targets = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    metrics = thresholded_binary_metrics(probabilities, targets, threshold=0.5)
    assert 0.0 <= metrics["dice"] < 0.1
    assert 0.0 <= metrics["recall"] < 0.1
