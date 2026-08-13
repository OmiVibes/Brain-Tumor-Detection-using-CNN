from __future__ import annotations

import torch

from brainscan.uncertainty.metrics import (
    assign_uncertainty_levels,
    derive_uncertainty_thresholds,
    error_detection_auroc,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood_from_logits,
    predictive_entropy,
    softmax_probabilities_from_logits,
    top1_top2_margin,
)


def test_ece_known_toy_example() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.6, 0.4],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    result = expected_calibration_error(probabilities, labels, n_bins=2)

    assert result["bin_count"] == 2
    assert result["ece"] >= 0.0
    assert len(result["bins"]) == 2


def test_perfect_predictions_produce_low_ece() -> None:
    probabilities = torch.tensor([[0.999, 0.001], [0.001, 0.999]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    result = expected_calibration_error(probabilities, labels, n_bins=2)
    assert result["ece"] < 0.01


def test_overconfident_predictions_produce_larger_ece() -> None:
    probabilities = torch.tensor([[0.99, 0.01], [0.99, 0.01]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    result = expected_calibration_error(probabilities, labels, n_bins=2)
    assert result["ece"] > 0.2


def test_multiclass_brier_score_matches_expected_value() -> None:
    probabilities = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.int64)
    score = multiclass_brier_score(probabilities, labels)
    assert abs(score - 0.14) < 1e-6


def test_nll_implementation_is_finite() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    value = negative_log_likelihood_from_logits(logits, labels)
    assert value >= 0.0


def test_entropy_is_finite() -> None:
    probabilities = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)
    entropy = predictive_entropy(probabilities)
    assert torch.isfinite(entropy).all()


def test_entropy_minimum_for_near_one_hot_probabilities() -> None:
    near_one_hot = torch.tensor([[0.999, 0.001]], dtype=torch.float32)
    uniform = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    assert predictive_entropy(near_one_hot).item() < predictive_entropy(uniform).item()


def test_entropy_higher_for_uniform_probabilities() -> None:
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float32)
    assert predictive_entropy(uniform).item() > predictive_entropy(peaked).item()


def test_top1_top2_margin_calculation() -> None:
    probabilities = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    margin = top1_top2_margin(probabilities)
    assert torch.allclose(margin, torch.tensor([0.5]))


def test_normalized_entropy_in_valid_range() -> None:
    probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    normalized = predictive_entropy(probabilities, normalize=True)
    assert 0.0 <= normalized.item() <= 1.0


def test_softmax_probabilities_sum_to_one() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    probabilities = softmax_probabilities_from_logits(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6)


def test_error_detection_auroc_returns_value() -> None:
    probabilities = torch.tensor(
        [
            [0.95, 0.05],
            [0.55, 0.45],
            [0.10, 0.90],
            [0.51, 0.49],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    entropy_auroc = error_detection_auroc(probabilities, labels, score_kind="entropy")
    confidence_auroc = error_detection_auroc(probabilities, labels, score_kind="one_minus_confidence")

    assert entropy_auroc is not None
    assert confidence_auroc is not None
    assert 0.0 <= entropy_auroc <= 1.0
    assert 0.0 <= confidence_auroc <= 1.0


def test_uncertainty_threshold_derivation_and_assignment() -> None:
    thresholds = derive_uncertainty_thresholds([0.1, 0.2, 0.8, 0.9], [0.9, 0.8, 0.2, 0.1])
    levels = assign_uncertainty_levels([0.05, 0.5, 0.95], [0.95, 0.5, 0.05], thresholds)
    assert levels == ["LOW", "MEDIUM", "HIGH"]
