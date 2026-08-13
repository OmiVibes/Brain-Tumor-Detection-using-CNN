from __future__ import annotations

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brainscan.evaluation.classification import (
    CONFIDENCE_MARGIN_THRESHOLD,
    annotate_predictions_with_near_duplicates,
    benchmark_model_inference,
    build_confusion_pairs,
    build_predictions_rows,
    compute_classification_metrics_bundle,
    compute_confidence_statistics,
    create_error_rows,
    evaluate_classifier,
    load_near_duplicate_annotations,
    summarize_filtered_subset,
)


def _toy_probabilities() -> list[list[float]]:
    return [
        [0.90, 0.05, 0.03, 0.02],
        [0.10, 0.80, 0.05, 0.05],
        [0.05, 0.10, 0.70, 0.15],
        [0.55, 0.25, 0.10, 0.10],
    ]


def test_metric_bundle_accuracy_macro_and_support() -> None:
    bundle = compute_classification_metrics_bundle(
        true_ids=[0, 1, 2, 3],
        predicted_ids=[0, 1, 2, 0],
        probabilities=_toy_probabilities(),
    )
    assert bundle["overall_metrics"]["accuracy"] == 0.75
    assert bundle["overall_metrics"]["balanced_accuracy"] == 0.75
    assert bundle["per_class_metrics"]["glioma"]["support"] == 1
    assert bundle["per_class_metrics"]["no_tumor"]["support"] == 1


def test_specificity_and_confusion_matrix_class_order() -> None:
    bundle = compute_classification_metrics_bundle(
        true_ids=[0, 1, 2, 3],
        predicted_ids=[0, 1, 2, 0],
        probabilities=_toy_probabilities(),
    )
    confusion = bundle["confusion_matrix"]
    assert confusion[3][0] == 1
    assert bundle["per_class_metrics"]["glioma"]["specificity"] < 1.0


def test_predictions_csv_schema_and_error_filtering() -> None:
    rows = build_predictions_rows(
        relative_paths=["dataset/test/a.jpg", "dataset/test/b.jpg"],
        true_ids=[0, 1],
        predicted_ids=[0, 0],
        probabilities=[[0.8, 0.1, 0.05, 0.05], [0.7, 0.2, 0.05, 0.05]],
    )
    assert rows[0]["probability_glioma"] == 0.8
    errors = create_error_rows(rows)
    assert len(errors) == 1
    assert errors[0]["true_class"] == "meningioma"
    assert "model_confidence" in errors[0]


def test_confidence_statistics_and_confusion_pairs() -> None:
    rows = build_predictions_rows(
        relative_paths=["dataset/test/a.jpg", "dataset/test/b.jpg", "dataset/test/c.jpg"],
        true_ids=[0, 1, 1],
        predicted_ids=[0, 0, 0],
        probabilities=[
            [0.95, 0.03, 0.01, 0.01],
            [0.92, 0.05, 0.02, 0.01],
            [0.52, 0.40, 0.04, 0.04],
        ],
    )
    stats = compute_confidence_statistics(rows)
    assert stats["incorrect_confidence_ge_090"] == 1
    assert stats["predictions_top1_top2_margin_lt_010"] == 0
    pairs = build_confusion_pairs(rows)
    assert pairs[0]["true_class"] == "meningioma"
    assert pairs[0]["predicted_class"] == "glioma"
    assert pairs[0]["count"] == 2


def test_near_duplicate_annotation_logic(tmp_path) -> None:
    report = tmp_path / "near_duplicates.csv"
    report.write_text(
        "image_a,image_b,split_a,split_b,class_a,class_b,hash_distance\n"
        "dataset/test/a.jpg,dataset/train/x.jpg,test,train,glioma,glioma,1\n"
        "dataset/test/b.jpg,dataset/val/y.jpg,test,val,meningioma,meningioma,2\n",
        encoding="utf-8",
    )
    annotations = load_near_duplicate_annotations(report, test_relative_paths=["dataset/test/a.jpg", "dataset/test/b.jpg"])
    rows = build_predictions_rows(
        relative_paths=["dataset/test/a.jpg", "dataset/test/b.jpg"],
        true_ids=[0, 1],
        predicted_ids=[0, 1],
        probabilities=[[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05]],
    )
    enriched = annotate_predictions_with_near_duplicates(rows, annotations)
    assert enriched[0]["near_duplicate_to_train"] is True
    assert enriched[1]["near_duplicate_to_val"] is True
    assert enriched[0]["minimum_phash_distance"] == 1


def test_missing_near_duplicate_report_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Near-duplicate report not found"):
        load_near_duplicate_annotations(tmp_path / "missing.csv", test_relative_paths=["dataset/test/a.jpg"])


def test_filtered_subset_summary_and_row_count() -> None:
    rows = build_predictions_rows(
        relative_paths=["dataset/test/a.jpg", "dataset/test/b.jpg"],
        true_ids=[0, 1],
        predicted_ids=[0, 1],
        probabilities=[[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05]],
    )
    enriched = annotate_predictions_with_near_duplicates(
        rows,
        {
            "dataset/test/a.jpg": type("A", (), {"near_duplicate_to_train": True, "near_duplicate_to_val": False, "minimum_phash_distance": 1})(),
            "dataset/test/b.jpg": type("B", (), {"near_duplicate_to_train": False, "near_duplicate_to_val": False, "minimum_phash_distance": None})(),
        },
    )
    summary = summarize_filtered_subset(enriched)
    assert summary["subset_size"] == 1
    assert summary["metrics"] is None


def test_evaluation_keeps_model_in_eval_mode_and_does_not_update_parameters() -> None:
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, inputs):
            return self.linear(inputs)

    model = ToyModel()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    dataset = [
        (
            torch.randn(4),
            torch.tensor(0, dtype=torch.long),
            {
                "relative_path": "dataset/test/a.jpg",
                "canonical_class": "glioma",
                "physical_class": "glioma",
                "sha256": "x",
            },
        )
    ]

    def _collate(batch):
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return inputs, labels, {
            "relative_path": [item[2]["relative_path"] for item in batch],
            "canonical_class": [item[2]["canonical_class"] for item in batch],
            "physical_class": [item[2]["physical_class"] for item in batch],
            "sha256": [item[2]["sha256"] for item in batch],
        }

    loader = DataLoader(dataset, batch_size=1, collate_fn=_collate)
    results = evaluate_classifier(model, loader, torch.device("cpu"))
    after = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    assert results["relative_paths"] == ["dataset/test/a.jpg"]
    assert model.training is False
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_benchmark_returns_positive_latency() -> None:
    model = nn.Linear(4, 4)
    inputs = torch.randn(2, 4)
    metrics = benchmark_model_inference(model, inputs, torch.device("cpu"), warmup_iterations=1, timed_iterations=2)
    assert metrics["mean_batch_latency_ms"] >= 0.0
    assert metrics["throughput_images_per_second"] > 0.0
