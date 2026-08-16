"""Losses and metrics for binary whole-tumor segmentation."""

from __future__ import annotations

import torch
from torch import nn


def soft_dice_score(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.reshape(probabilities.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 1.0 - soft_dice_score(logits, targets, epsilon=self.epsilon)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def thresholded_binary_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    true_positive = float((predictions * targets).sum().item())
    true_negative = float(((1.0 - predictions) * (1.0 - targets)).sum().item())
    false_positive = float((predictions * (1.0 - targets)).sum().item())
    false_negative = float(((1.0 - predictions) * targets).sum().item())

    dice = (2.0 * true_positive + epsilon) / (2.0 * true_positive + false_positive + false_negative + epsilon)
    iou = (true_positive + epsilon) / (true_positive + false_positive + false_negative + epsilon)
    precision = (true_positive + epsilon) / (true_positive + false_positive + epsilon)
    recall = (true_positive + epsilon) / (true_positive + false_negative + epsilon)
    specificity = (true_negative + epsilon) / (true_negative + false_positive + epsilon)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }
