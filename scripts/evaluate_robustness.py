"""Formal robustness and abstention evaluation for the frozen BrainScanAI pipeline."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from brainscan.core.config import resolve_project_path
from brainscan.data import load_manifest_records
from brainscan.inference import BrainScanInferencePipeline
from brainscan.robustness import (
    build_abstention_policy_artifact,
    build_risk_coverage_curve,
    compute_classwise_status_counts,
    compute_correctness_breakdown,
    compute_entropy_ood_relationship,
    compute_error_capture,
    compute_quality_flag_frequencies,
    compute_selective_metrics,
    compute_status_summary,
    plot_risk_coverage_curve,
    save_csv,
    save_json,
    summarize_numeric_distribution,
    summarize_perturbation_rows,
    summarize_synthetic_ood,
)
from brainscan.training.train_classifier import get_git_commit


ROBUSTNESS_DIR = Path("artifacts/robustness/resnet18_baseline")
ABSTENTION_POLICY_PATH = ROBUSTNESS_DIR / "abstention_policy.json"
ROBUSTNESS_METRICS_PATH = ROBUSTNESS_DIR / "robustness_metrics.json"
TEST_POLICY_RESULTS_PATH = ROBUSTNESS_DIR / "test_policy_results.csv"
ERROR_CAPTURE_PATH = ROBUSTNESS_DIR / "error_capture.csv"
PERTURBATION_SUMMARY_PATH = ROBUSTNESS_DIR / "perturbation_summary.csv"
SYNTHETIC_OOD_SUMMARY_PATH = ROBUSTNESS_DIR / "synthetic_ood_summary.csv"
RISK_COVERAGE_PATH = ROBUSTNESS_DIR / "risk_coverage.json"
RISK_COVERAGE_CURVE_PATH = ROBUSTNESS_DIR / "risk_coverage_curve.png"

VALIDATION_MANIFEST = Path("data/splits/val.csv")
TEST_MANIFEST = Path("data/splits/test.csv")
CLASSIFIER_ONLY_PER_IMAGE_LATENCY_MS = 7.3789092499995945
PERTURBATION_SAMPLES_PER_CLASS = 20
PERTURBATION_SEED = 42
RISK_COVERAGE_QUANTILES = np.linspace(0.05, 1.0, 20)


def _serialize_flags(flags: list[str] | None) -> str:
    if not flags:
        return ""
    return "|".join(flags)


def _serialize_reasons(reasons: list[str] | None) -> str:
    if not reasons:
        return ""
    return "|".join(reasons)


def _evaluate_manifest(
    pipeline: BrainScanInferencePipeline,
    manifest_path: Path,
    *,
    split_name: str,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    records = load_manifest_records(manifest_path)
    if any(record.split != split_name for record in records):
        raise ValueError(f"Manifest {manifest_path} contains non-{split_name} rows.")

    rows: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for record in records:
        start = time.perf_counter()
        result = pipeline.predict(record.relative_path)
        latency_ms = (time.perf_counter() - start) * 1000.0
        prediction = result.analysis_prediction
        rows.append(
            {
                "relative_path": record.relative_path,
                "split": record.split,
                "true_class": record.canonical_class,
                "true_class_id": record.class_id,
                "classifier_prediction": prediction.class_name if prediction is not None else "",
                "classifier_prediction_id": prediction.class_id if prediction is not None else "",
                "classifier_correct": (
                    prediction.class_id == record.class_id if prediction is not None else False
                ),
                "raw_confidence": prediction.raw_confidence if prediction is not None else "",
                "public_prediction": result.prediction.class_name if result.prediction is not None else "",
                "public_prediction_withheld": result.prediction is None,
                "entropy": result.uncertainty.entropy if result.uncertainty is not None else "",
                "normalized_entropy": (
                    result.uncertainty.normalized_entropy if result.uncertainty is not None else ""
                ),
                "margin": result.uncertainty.margin if result.uncertainty is not None else "",
                "uncertainty_level": result.uncertainty.level if result.uncertainty is not None else "",
                "ood_score": result.ood.score if result.ood is not None else "",
                "ood_status": result.ood.status if result.ood is not None else "",
                "quality_flags": _serialize_flags(result.quality.flags if result.quality is not None else []),
                "safe_status": result.status,
                "decision_reasons": _serialize_reasons(result.decision.reasons),
                "latency_ms": latency_ms,
            }
        )
    total_seconds = time.perf_counter() - total_start
    performance = {
        "mean_latency_ms": float(np.mean([float(row["latency_ms"]) for row in rows])) if rows else 0.0,
        "throughput_images_per_second": len(rows) / total_seconds if total_seconds else 0.0,
        "total_wall_time_seconds": total_seconds,
    }
    return rows, performance


def _create_synthetic_probe_images(base_dir: Path) -> list[tuple[str, Path]]:
    probe_paths: list[tuple[str, Path]] = []
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(base_dir / "black.png")
    probe_paths.append(("black", base_dir / "black.png"))
    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(base_dir / "white.png")
    probe_paths.append(("white", base_dir / "white.png"))
    Image.new("RGB", (224, 224), color=(127, 127, 127)).save(base_dir / "gray.png")
    probe_paths.append(("gray", base_dir / "gray.png"))

    gradient = np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 1))
    Image.fromarray(np.stack([gradient, gradient, gradient], axis=2), mode="RGB").save(base_dir / "gradient.png")
    probe_paths.append(("gradient", base_dir / "gradient.png"))

    checker = (((np.indices((224, 224)).sum(axis=0) % 2) * 255)).astype(np.uint8)
    Image.fromarray(np.stack([checker, checker, checker], axis=2), mode="RGB").save(base_dir / "checkerboard.png")
    probe_paths.append(("checkerboard", base_dir / "checkerboard.png"))

    rng = np.random.default_rng(42)
    random_noise = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    Image.fromarray(random_noise, mode="RGB").save(base_dir / "random_noise.png")
    probe_paths.append(("random_noise", base_dir / "random_noise.png"))

    base_gray = np.full((224, 224, 3), 127, dtype=np.float32)
    gaussian_low = np.clip(base_gray + rng.normal(0.0, 10.0, size=base_gray.shape), 0, 255).astype(np.uint8)
    Image.fromarray(gaussian_low, mode="RGB").save(base_dir / "gaussian_noise_low.png")
    probe_paths.append(("gaussian_noise_low", base_dir / "gaussian_noise_low.png"))

    gaussian_high = np.clip(base_gray + rng.normal(0.0, 35.0, size=base_gray.shape), 0, 255).astype(np.uint8)
    Image.fromarray(gaussian_high, mode="RGB").save(base_dir / "gaussian_noise_high.png")
    probe_paths.append(("gaussian_noise_high", base_dir / "gaussian_noise_high.png"))

    return probe_paths


def _evaluate_synthetic_ood(pipeline: BrainScanInferencePipeline) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="brainscan_synth_ood_") as tmp_dir:
        for probe_name, image_path in _create_synthetic_probe_images(Path(tmp_dir)):
            result = pipeline.predict(image_path)
            prediction = result.analysis_prediction
            rows.append(
                {
                    "probe_name": probe_name,
                    "classifier_prediction": prediction.class_name if prediction is not None else "",
                    "raw_confidence": prediction.raw_confidence if prediction is not None else "",
                    "normalized_entropy": (
                        result.uncertainty.normalized_entropy if result.uncertainty is not None else ""
                    ),
                    "ood_score": result.ood.score if result.ood is not None else "",
                    "ood_status": result.ood.status if result.ood is not None else "",
                    "quality_flags": _serialize_flags(result.quality.flags if result.quality is not None else []),
                    "safe_status": result.status,
                    "decision_reasons": _serialize_reasons(result.decision.reasons),
                }
            )
    return rows


def _apply_perturbation(image: Image.Image, perturbation_name: str, rng: np.random.Generator) -> Image.Image:
    rgb = image.convert("RGB")
    if perturbation_name == "mild_gaussian_noise":
        array = np.asarray(rgb, dtype=np.float32)
        noisy = np.clip(array + rng.normal(0.0, 5.0, size=array.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")
    if perturbation_name == "strong_gaussian_noise":
        array = np.asarray(rgb, dtype=np.float32)
        noisy = np.clip(array + rng.normal(0.0, 20.0, size=array.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")
    if perturbation_name == "mild_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=1.5))
    if perturbation_name == "strong_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=4.0))
    if perturbation_name == "brightness_decrease":
        return ImageEnhance.Brightness(rgb).enhance(0.75)
    if perturbation_name == "brightness_increase":
        return ImageEnhance.Brightness(rgb).enhance(1.25)
    if perturbation_name == "contrast_decrease":
        return ImageEnhance.Contrast(rgb).enhance(0.70)
    raise ValueError(f"Unsupported perturbation '{perturbation_name}'.")


def _select_perturbation_subset() -> list[str]:
    records = load_manifest_records(TEST_MANIFEST)
    rng = np.random.default_rng(PERTURBATION_SEED)
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.canonical_class].append(record.relative_path)

    selected_paths: list[str] = []
    for class_name in sorted(grouped):
        candidates = sorted(grouped[class_name])
        sample_size = min(PERTURBATION_SAMPLES_PER_CLASS, len(candidates))
        indices = rng.choice(len(candidates), size=sample_size, replace=False)
        selected_paths.extend(candidates[index] for index in sorted(indices))
    return selected_paths


def _evaluate_perturbations(
    pipeline: BrainScanInferencePipeline,
    test_rows: Sequence[dict[str, object]],
) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    selected_paths = _select_perturbation_subset()
    base_rows = {str(row["relative_path"]): row for row in test_rows}
    perturbation_rows: list[dict[str, object]] = []
    perturbations = (
        "mild_gaussian_noise",
        "strong_gaussian_noise",
        "mild_blur",
        "strong_blur",
        "brightness_decrease",
        "brightness_increase",
        "contrast_decrease",
    )
    rng = np.random.default_rng(PERTURBATION_SEED)

    with TemporaryDirectory(prefix="brainscan_perturb_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for relative_path in selected_paths:
            base_row = base_rows[relative_path]
            image_path = PROJECT_ROOT / relative_path
            with Image.open(image_path) as image:
                for perturbation_name in perturbations:
                    perturbed = _apply_perturbation(image, perturbation_name, rng)
                    target_path = tmp_dir_path / f"{Path(relative_path).stem}_{perturbation_name}.png"
                    perturbed.save(target_path)
                    result = pipeline.predict(target_path)
                    prediction = result.analysis_prediction
                    base_prediction = str(base_row["classifier_prediction"])
                    current_prediction = prediction.class_name if prediction is not None else ""
                    perturbation_rows.append(
                        {
                            "relative_path": relative_path,
                            "true_class": base_row["true_class"],
                            "perturbation_name": perturbation_name,
                            "base_prediction": base_prediction,
                            "perturbed_prediction": current_prediction,
                            "prediction_consistent": current_prediction == base_prediction,
                            "raw_confidence_change": (
                                float(prediction.raw_confidence) - float(base_row["raw_confidence"])
                                if prediction is not None
                                else 0.0
                            ),
                            "entropy_change": (
                                float(result.uncertainty.entropy) - float(base_row["entropy"])
                                if result.uncertainty is not None
                                else 0.0
                            ),
                            "perturbed_ood_score": float(result.ood.score) if result.ood is not None else 0.0,
                            "ood_score_change": (
                                float(result.ood.score) - float(base_row["ood_score"])
                                if result.ood is not None
                                else 0.0
                            ),
                            "safe_status": result.status,
                            "quality_flags": _serialize_flags(result.quality.flags if result.quality is not None else []),
                        }
                    )
    return len(selected_paths), perturbation_rows, summarize_perturbation_rows(perturbation_rows)


def _build_risk_coverage_artifact(
    validation_rows: Sequence[dict[str, object]],
    test_rows: Sequence[dict[str, object]],
) -> tuple[dict[str, object], Path]:
    validation_entropy = np.asarray([float(row["normalized_entropy"]) for row in validation_rows], dtype=np.float64)
    thresholds = sorted({float(np.quantile(validation_entropy, quantile)) for quantile in RISK_COVERAGE_QUANTILES})
    validation_points = build_risk_coverage_curve(validation_rows, score_key="normalized_entropy", thresholds=thresholds)
    test_points = build_risk_coverage_curve(test_rows, score_key="normalized_entropy", thresholds=thresholds)
    payload = {
        "score_kind": "normalized_entropy",
        "threshold_source": "validation quantiles only",
        "thresholds": thresholds,
        "validation_points": validation_points,
        "test_points": test_points,
    }
    save_json(payload, RISK_COVERAGE_PATH)
    plot_risk_coverage_curve(validation_points, test_points, RISK_COVERAGE_CURVE_PATH)
    return payload, resolve_project_path(RISK_COVERAGE_PATH)


def _build_conclusion(
    accept_metrics: dict[str, float | int],
    error_capture: dict[str, object],
    correctness: dict[str, object],
) -> str:
    coverage = float(accept_metrics["coverage"])
    risk = float(accept_metrics["risk"])
    captured = float(error_capture["error_capture_rate"])
    correct_non_acceptance = float(correctness["correct_non_acceptance_rate"])
    if coverage < 0.5 and correct_non_acceptance > 0.4 and captured > 0.7:
        return (
            "Current policy substantially reduces accepted error risk and captures most classifier errors, "
            "but it is quite conservative because more than half of correct predictions are not automatically accepted."
        )
    if risk < (13.0 / 702.0) and captured > 0.5:
        return "Current policy provides a useful risk reduction with moderate review burden."
    return "Current policy appears too conservative relative to the achieved accepted-error reduction."


def main() -> None:
    pipeline = BrainScanInferencePipeline()

    policy_payload, policy_path = build_abstention_policy_artifact(
        checkpoint_path=pipeline.checkpoint_path,
        checkpoint_epoch=int(pipeline.checkpoint["epoch"]),
        dataset_fingerprint=pipeline.dataset_fingerprint,
        ood_metadata=pipeline.ood_metadata,
        quality_payload=pipeline.quality_payload,
        uncertainty_metrics=pipeline.uncertainty_metrics,
        git_commit=get_git_commit(),
        output_path=ABSTENTION_POLICY_PATH,
    )

    validation_rows, validation_perf = _evaluate_manifest(pipeline, VALIDATION_MANIFEST, split_name="val")
    validation_summary = compute_status_summary(validation_rows)
    if validation_summary["total"] != 703:
        raise ValueError(f"Expected exactly 703 validation rows, found {validation_summary['total']}.")

    test_rows, test_perf = _evaluate_manifest(pipeline, TEST_MANIFEST, split_name="test")
    if len(test_rows) != 702:
        raise ValueError(f"Expected exactly 702 test rows, found {len(test_rows)}.")
    save_csv(test_rows, TEST_POLICY_RESULTS_PATH)

    test_summary = compute_status_summary(test_rows)
    correctness = compute_correctness_breakdown(test_rows)
    accept_metrics = compute_selective_metrics(test_rows, accepted_statuses={"ACCEPT"})
    accept_review_metrics = compute_selective_metrics(test_rows, accepted_statuses={"ACCEPT", "REVIEW"})
    error_capture = compute_error_capture(test_rows)
    if error_capture["wrong_total"] != 13:
        raise ValueError(f"Expected exactly 13 classifier errors on the held-out test set, found {error_capture['wrong_total']}.")

    error_rows = [
        {
            "relative_path": row["relative_path"],
            "true_class": row["true_class"],
            "classifier_prediction": row["classifier_prediction"],
            "raw_confidence": row["raw_confidence"],
            "uncertainty_level": row["uncertainty_level"],
            "ood_score": row["ood_score"],
            "ood_status": row["ood_status"],
            "quality_flags": row["quality_flags"],
            "safe_status": row["safe_status"],
            "decision_reasons": row["decision_reasons"],
        }
        for row in error_capture["error_rows"]
    ]
    save_csv(error_rows, ERROR_CAPTURE_PATH)

    synthetic_rows = _evaluate_synthetic_ood(pipeline)
    synthetic_summary = summarize_synthetic_ood(synthetic_rows)
    save_csv(synthetic_rows, SYNTHETIC_OOD_SUMMARY_PATH)

    perturbation_sample_size, perturbation_rows, perturbation_summary = _evaluate_perturbations(pipeline, test_rows)
    save_csv(perturbation_summary, PERTURBATION_SUMMARY_PATH)

    risk_coverage_payload, _risk_coverage_path = _build_risk_coverage_artifact(validation_rows, test_rows)

    quality_frequencies = compute_quality_flag_frequencies(test_rows)
    classwise_behavior = compute_classwise_status_counts(test_rows)
    entropy_ood_relationship = compute_entropy_ood_relationship(test_rows)

    ood_score_summary = {
        "validation_clean": summarize_numeric_distribution([float(row["ood_score"]) for row in validation_rows]),
        "test_clean": summarize_numeric_distribution([float(row["ood_score"]) for row in test_rows]),
        "synthetic_ood": summarize_numeric_distribution(
            [float(row["ood_score"]) for row in synthetic_rows if row["ood_score"] != ""]
        ),
        "perturbed": summarize_numeric_distribution(
            [float(row["perturbed_ood_score"]) for row in perturbation_rows]
        ),
    }

    baseline_accuracy = 1.0 - (13.0 / 702.0)
    baseline_risk = 13.0 / 702.0
    conclusion = _build_conclusion(accept_metrics, error_capture, correctness)
    performance_reference = json.loads(resolve_project_path("artifacts/evaluation/resnet18_baseline/performance.json").read_text(encoding="utf-8"))

    metrics_payload = {
        "policy_artifact": make_policy_relative(policy_path),
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "policy": policy_payload,
        "validation_summary": {
            "status": validation_summary,
            "performance": validation_perf,
        },
        "test_summary": {
            "status": test_summary,
            "performance": test_perf,
        },
        "baseline_metrics": {
            "accuracy": baseline_accuracy,
            "risk": baseline_risk,
        },
        "accept_only_metrics": accept_metrics,
        "accept_review_metrics": accept_review_metrics,
        "correctness_breakdown": correctness,
        "error_capture": {
            "wrong_total": error_capture["wrong_total"],
            "counts": error_capture["counts"],
            "unsafe_accepted_error_count": error_capture["unsafe_accepted_error_count"],
            "unsafe_accepted_error_rate_all_test": error_capture["unsafe_accepted_error_rate_all_test"],
            "error_capture_rate": error_capture["error_capture_rate"],
            "hard_error_capture_rate": error_capture["hard_error_capture_rate"],
            "unsafe_accepted_errors": [
                {
                    "relative_path": row["relative_path"],
                    "true_class": row["true_class"],
                    "classifier_prediction": row["classifier_prediction"],
                    "raw_confidence": row["raw_confidence"],
                    "safe_status": row["safe_status"],
                }
                for row in error_capture["unsafe_accepted_errors"]
            ],
        },
        "classwise_behavior": classwise_behavior,
        "quality_flag_frequencies": quality_frequencies,
        "synthetic_ood": {
            "summary": synthetic_summary,
            "score_distribution": summarize_numeric_distribution(
                [float(row["ood_score"]) for row in synthetic_rows if row["ood_score"] != ""]
            ),
        },
        "perturbation": {
            "sample_size": perturbation_sample_size,
            "summary_rows": perturbation_summary,
        },
        "ood_score_summary": ood_score_summary,
        "entropy_ood_relationship": entropy_ood_relationship,
        "risk_coverage": risk_coverage_payload,
        "latency": {
            "safe_inference_mean_latency_ms": test_perf["mean_latency_ms"],
            "safe_inference_throughput_images_per_second": test_perf["throughput_images_per_second"],
            "classifier_only_latency_ms_reference": performance_reference["mean_per_image_latency_ms"],
        },
        "artifacts": {
            "test_policy_results_csv": "artifacts/robustness/resnet18_baseline/test_policy_results.csv",
            "error_capture_csv": "artifacts/robustness/resnet18_baseline/error_capture.csv",
            "synthetic_ood_summary_csv": "artifacts/robustness/resnet18_baseline/synthetic_ood_summary.csv",
            "perturbation_summary_csv": "artifacts/robustness/resnet18_baseline/perturbation_summary.csv",
            "risk_coverage_json": "artifacts/robustness/resnet18_baseline/risk_coverage.json",
            "risk_coverage_curve_png": "artifacts/robustness/resnet18_baseline/risk_coverage_curve.png",
        },
        "conclusion": conclusion,
    }
    save_json(metrics_payload, ROBUSTNESS_METRICS_PATH)

    print("robustness_evaluation_summary=")
    print(f"  policy_artifact={policy_path}")
    print(f"  test_policy_results={resolve_project_path(TEST_POLICY_RESULTS_PATH)}")
    print(f"  test_accept={test_summary['counts']['ACCEPT']}")
    print(f"  test_review={test_summary['counts']['REVIEW']}")
    print(f"  test_abstain={test_summary['counts']['ABSTAIN']}")
    print(f"  unsafe_accepted_error_count={error_capture['unsafe_accepted_error_count']}")
    print(f"  synthetic_ood_abstention_rate={synthetic_summary['ood_abstention_rate']:.6f}")
    print(f"  safe_latency_ms={test_perf['mean_latency_ms']:.6f}")
    print(f"  risk_coverage_json={resolve_project_path(RISK_COVERAGE_PATH)}")
    print(f"  robustness_metrics_json={resolve_project_path(ROBUSTNESS_METRICS_PATH)}")


def make_policy_relative(path_value: Path) -> str:
    return path_value.relative_to(PROJECT_ROOT).as_posix()


if __name__ == "__main__":
    main()
