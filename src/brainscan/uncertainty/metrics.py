"""Calibration and uncertainty metrics for BrainScanAI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class CalibrationBin:
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


def softmax_probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits into numerically stable softmax probabilities."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
    probabilities = torch.softmax(logits, dim=1)
    if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
        raise ValueError("Softmax probabilities contain NaN or Inf values.")
    return probabilities


def negative_log_likelihood_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the mean multiclass negative log-likelihood."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Logits and labels must contain the same number of samples.")
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def multiclass_brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the mean multiclass Brier score with one-hot targets."""
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probabilities.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {tuple(labels.shape)}")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Probabilities and labels must contain the same number of samples.")

    num_classes = probabilities.shape[1]
    targets = F.one_hot(labels, num_classes=num_classes).to(dtype=probabilities.dtype)
    per_sample = torch.sum((probabilities - targets) ** 2, dim=1)
    return float(per_sample.mean().item())


def predictive_entropy(
    probabilities: torch.Tensor,
    *,
    normalize: bool = False,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compute predictive entropy from probabilities."""
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probabilities.shape)}")

    clamped = probabilities.clamp(min=epsilon, max=1.0)
    entropy = -(clamped * torch.log(clamped)).sum(dim=1)
    if normalize:
        normalizer = math.log(probabilities.shape[1])
        if normalizer <= 0.0:
            raise ValueError("Probability tensor must contain at least two classes for normalized entropy.")
        entropy = entropy / normalizer
    if torch.isnan(entropy).any() or torch.isinf(entropy).any():
        raise ValueError("Predictive entropy contains NaN or Inf values.")
    return entropy


def top1_top2_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """Return the top-1 minus top-2 probability margin for each sample."""
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probabilities.shape)}")
    if probabilities.shape[1] < 2:
        raise ValueError("At least two classes are required to compute a top-1/top-2 margin.")

    top2 = torch.topk(probabilities, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
) -> dict[str, object]:
    """Compute Expected Calibration Error and detailed reliability-bin statistics."""
    if n_bins <= 0:
        raise ValueError("Expected Calibration Error requires at least one bin.")
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probabilities.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {tuple(labels.shape)}")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Probabilities and labels must contain the same number of samples.")

    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, dtype=probabilities.dtype)
    total = probabilities.shape[0]
    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0

    for index in range(n_bins):
        lower = float(bin_edges[index].item())
        upper = float(bin_edges[index + 1].item())
        if index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        sample_count = int(mask.sum().item())
        if sample_count == 0:
            mean_confidence = 0.0
            empirical_accuracy = 0.0
            calibration_gap = 0.0
        else:
            mean_confidence = float(confidences[mask].mean().item())
            empirical_accuracy = float(correct[mask].float().mean().item())
            calibration_gap = abs(mean_confidence - empirical_accuracy)
            ece += (sample_count / total) * calibration_gap
            mce = max(mce, calibration_gap)

        bins.append(
            CalibrationBin(
                lower_bound=lower,
                upper_bound=upper,
                sample_count=sample_count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                calibration_gap=calibration_gap,
            )
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": [bin_info.__dict__ for bin_info in bins],
        "bin_count": n_bins,
    }


def compute_probability_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
) -> dict[str, object]:
    """Compute classification, calibration, and uncertainty metrics from logits."""
    probabilities = softmax_probabilities_from_logits(logits)
    predicted_ids = probabilities.argmax(dim=1)
    calibration = expected_calibration_error(probabilities, labels, n_bins=n_bins)
    entropy = predictive_entropy(probabilities)
    normalized_entropy = predictive_entropy(probabilities, normalize=True)
    margin = top1_top2_margin(probabilities)

    return {
        "accuracy": float(accuracy_score(labels.detach().cpu().numpy(), predicted_ids.detach().cpu().numpy())),
        "macro_f1": float(
            f1_score(labels.detach().cpu().numpy(), predicted_ids.detach().cpu().numpy(), average="macro")
        ),
        "nll": negative_log_likelihood_from_logits(logits, labels),
        "brier_score": multiclass_brier_score(probabilities, labels),
        "ece": float(calibration["ece"]),
        "mce": float(calibration["mce"]),
        "calibration_bins": calibration["bins"],
        "probabilities": probabilities,
        "predicted_ids": predicted_ids,
        "confidence": probabilities.max(dim=1).values,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top1_top2_margin": margin,
    }


def summarize_uncertainty_by_correctness(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compare confidence, entropy, and margin between correct and incorrect predictions."""
    predicted_ids = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    entropy = predictive_entropy(probabilities)
    margin = top1_top2_margin(probabilities)
    correct_mask = predicted_ids.eq(labels)
    incorrect_mask = ~correct_mask

    def _mean(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else 0.0

    def _median(values: torch.Tensor) -> float:
        return float(values.median().item()) if values.numel() else 0.0

    return {
        "mean_confidence_correct": _mean(confidences[correct_mask]),
        "mean_confidence_incorrect": _mean(confidences[incorrect_mask]),
        "median_confidence_correct": _median(confidences[correct_mask]),
        "median_confidence_incorrect": _median(confidences[incorrect_mask]),
        "mean_entropy_correct": _mean(entropy[correct_mask]),
        "mean_entropy_incorrect": _mean(entropy[incorrect_mask]),
        "median_entropy_correct": _median(entropy[correct_mask]),
        "median_entropy_incorrect": _median(entropy[incorrect_mask]),
        "mean_margin_correct": _mean(margin[correct_mask]),
        "mean_margin_incorrect": _mean(margin[incorrect_mask]),
        "median_margin_correct": _median(margin[correct_mask]),
        "median_margin_incorrect": _median(margin[incorrect_mask]),
    }


def error_detection_auroc(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    score_kind: str,
) -> float | None:
    """Compute AUROC for detecting incorrect predictions from an uncertainty score."""
    predicted_ids = probabilities.argmax(dim=1)
    incorrect_targets = (~predicted_ids.eq(labels)).to(dtype=torch.int64).cpu().numpy()
    if len(np.unique(incorrect_targets)) < 2:
        return None

    if score_kind == "entropy":
        scores = predictive_entropy(probabilities).detach().cpu().numpy()
    elif score_kind == "one_minus_confidence":
        scores = (1.0 - probabilities.max(dim=1).values).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported error-detection score kind '{score_kind}'.")

    return float(roc_auc_score(incorrect_targets, scores))


def derive_uncertainty_thresholds(
    normalized_entropy_values: Sequence[float],
    margin_values: Sequence[float],
) -> dict[str, float | str]:
    """Derive transparent LOW/MEDIUM/HIGH uncertainty thresholds from validation quantiles."""
    entropy_array = np.asarray(normalized_entropy_values, dtype=np.float64)
    margin_array = np.asarray(margin_values, dtype=np.float64)
    if entropy_array.size == 0 or margin_array.size == 0:
        raise ValueError("Cannot derive uncertainty thresholds from empty validation statistics.")

    thresholds = {
        "entropy_q25": float(np.quantile(entropy_array, 0.25)),
        "entropy_q75": float(np.quantile(entropy_array, 0.75)),
        "margin_q25": float(np.quantile(margin_array, 0.25)),
        "margin_q75": float(np.quantile(margin_array, 0.75)),
        "strategy": (
            "HIGH if normalized_entropy >= entropy_q75 or top1_top2_margin <= margin_q25; "
            "LOW if normalized_entropy <= entropy_q25 and top1_top2_margin >= margin_q75; "
            "otherwise MEDIUM"
        ),
    }
    return thresholds


def assign_uncertainty_levels(
    normalized_entropy: Sequence[float],
    margins: Sequence[float],
    thresholds: dict[str, float | str],
) -> list[str]:
    """Assign LOW/MEDIUM/HIGH uncertainty levels using validation-derived thresholds."""
    entropy_q25 = float(thresholds["entropy_q25"])
    entropy_q75 = float(thresholds["entropy_q75"])
    margin_q25 = float(thresholds["margin_q25"])
    margin_q75 = float(thresholds["margin_q75"])

    levels: list[str] = []
    for entropy_value, margin_value in zip(normalized_entropy, margins):
        if entropy_value >= entropy_q75 or margin_value <= margin_q25:
            levels.append("HIGH")
        elif entropy_value <= entropy_q25 and margin_value >= margin_q75:
            levels.append("LOW")
        else:
            levels.append("MEDIUM")
    return levels
