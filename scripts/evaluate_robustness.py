"""Formal robustness and abstention evaluation for frozen BrainScanAI pipelines."""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
from collections.abc import Sequence
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

from brainscan.core.config import make_project_relative_path, resolve_project_path
from brainscan.data import load_manifest_records
from brainscan.inference import BrainScanInferencePipeline
from brainscan.robustness import (
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


DEFAULT_CONFIG_PATH = Path("configs/densenet121_final.yaml")
DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/comparison/densenet121_seed42/best.pt")
DEFAULT_OOD_REFERENCE_NPZ_PATH = Path("artifacts/robustness/densenet121_final/ood_reference.npz")
DEFAULT_OOD_REFERENCE_JSON_PATH = Path("artifacts/robustness/densenet121_final/ood_reference.json")
DEFAULT_QUALITY_THRESHOLDS_PATH = Path("artifacts/robustness/densenet121_final/quality_thresholds.json")
DEFAULT_UNCERTAINTY_METRICS_PATH = Path("artifacts/calibration/densenet121_final/metrics.json")
DEFAULT_POLICY_PATH = Path("artifacts/robustness/densenet121_final/abstention_policy.json")
DEFAULT_FROZEN_VALIDATION_SUMMARY_PATH = Path("artifacts/robustness/densenet121_final/validation_policy_summary.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/robustness/densenet121_final")
DEFAULT_CLASSIFIER_PERFORMANCE_PATH = Path("artifacts/evaluation/densenet121_finalist/performance.json")
DEFAULT_BASELINE_ROBUSTNESS_METRICS_PATH = Path("artifacts/robustness/resnet18_baseline/robustness_metrics.json")
DEFAULT_VALIDATION_MANIFEST = Path("data/splits/val.csv")
DEFAULT_TEST_MANIFEST = Path("data/splits/test.csv")

PERTURBATION_SAMPLES_PER_CLASS = 20
PERTURBATION_SEED = 42
RISK_COVERAGE_QUANTILES = np.linspace(0.05, 1.0, 20)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run formal BrainScanAI robustness evaluation on a frozen policy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Inference config path.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Frozen checkpoint path.")
    parser.add_argument(
        "--ood-reference-npz",
        default=str(DEFAULT_OOD_REFERENCE_NPZ_PATH),
        help="Path to the frozen OOD reference NPZ.",
    )
    parser.add_argument(
        "--ood-reference-json",
        default=str(DEFAULT_OOD_REFERENCE_JSON_PATH),
        help="Path to the frozen OOD reference JSON.",
    )
    parser.add_argument(
        "--quality-thresholds-path",
        default=str(DEFAULT_QUALITY_THRESHOLDS_PATH),
        help="Path to the quality-threshold JSON.",
    )
    parser.add_argument(
        "--uncertainty-metrics-path",
        default=str(DEFAULT_UNCERTAINTY_METRICS_PATH),
        help="Path to the uncertainty metrics JSON.",
    )
    parser.add_argument(
        "--policy-path",
        default=str(DEFAULT_POLICY_PATH),
        help="Path to the already frozen abstention policy artifact.",
    )
    parser.add_argument(
        "--frozen-validation-summary-path",
        default=str(DEFAULT_FROZEN_VALIDATION_SUMMARY_PATH),
        help="Path to the validation-only summary captured when the policy was frozen.",
    )
    parser.add_argument(
        "--classifier-performance-path",
        default=str(DEFAULT_CLASSIFIER_PERFORMANCE_PATH),
        help="Classifier-only performance JSON used for latency comparison.",
    )
    parser.add_argument(
        "--baseline-robustness-metrics-path",
        default=str(DEFAULT_BASELINE_ROBUSTNESS_METRICS_PATH),
        help="Historical ResNet18 robustness metrics used for descriptive comparison.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for robustness artifacts.")
    parser.add_argument(
        "--validation-manifest",
        default=str(DEFAULT_VALIDATION_MANIFEST),
        help="Validation manifest used for pre-frozen threshold derivation.",
    )
    parser.add_argument("--test-manifest", default=str(DEFAULT_TEST_MANIFEST), help="Held-out test manifest.")
    parser.add_argument(
        "--prefer-cuda",
        action=BooleanOptionalAction,
        default=True,
        help="Prefer CUDA when available.",
    )
    return parser


def _serialize_flags(flags: list[str] | None) -> str:
    return "|".join(flags or [])


def _serialize_reasons(reasons: list[str] | None) -> str:
    return "|".join(reasons or [])


def _load_json(path_value: str | Path) -> dict[str, object]:
    with resolve_project_path(path_value).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}.")
    return payload


def _relative(path_value: str | Path) -> str:
    return make_project_relative_path(path_value)


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
                "classifier_correct": prediction.class_id == record.class_id if prediction is not None else False,
                "raw_confidence": prediction.raw_confidence if prediction is not None else "",
                "public_prediction": result.prediction.class_name if result.prediction is not None else "",
                "public_prediction_withheld": result.prediction is None,
                "entropy": result.uncertainty.entropy if result.uncertainty is not None else "",
                "normalized_entropy": result.uncertainty.normalized_entropy if result.uncertainty is not None else "",
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

    checker = ((np.indices((224, 224)).sum(axis=0) % 2) * 255).astype(np.uint8)
    Image.fromarray(np.stack([checker, checker, checker], axis=2), mode="RGB").save(base_dir / "checkerboard.png")
    probe_paths.append(("checkerboard", base_dir / "checkerboard.png"))

    rng = np.random.default_rng(PERTURBATION_SEED)
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
                    "normalized_entropy": result.uncertainty.normalized_entropy if result.uncertainty is not None else "",
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


def _select_perturbation_subset(test_manifest: Path) -> list[str]:
    records = load_manifest_records(test_manifest)
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
    test_manifest: Path,
) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    selected_paths = _select_perturbation_subset(test_manifest)
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
    *,
    output_json_path: Path,
    output_png_path: Path,
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
    save_json(payload, output_json_path)
    plot_risk_coverage_curve(validation_points, test_points, output_png_path)
    return payload, resolve_project_path(output_json_path)


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
            "but it remains conservative because many correct predictions still require review or abstention."
        )
    if risk < 0.01 and captured > 0.5:
        return "Current policy provides a useful risk reduction with moderate review burden."
    return "Current policy appears conservative relative to the accepted-error reduction achieved."


def _verify_frozen_policy(
    pipeline: BrainScanInferencePipeline,
    policy_payload: dict[str, object],
    frozen_validation_summary: dict[str, object],
    validation_summary: dict[str, object],
) -> None:
    expected_checkpoint = _relative(pipeline.checkpoint_path)
    if str(policy_payload["checkpoint"]) != expected_checkpoint:
        raise ValueError("Frozen policy checkpoint does not match the active inference checkpoint.")
    if str(policy_payload["architecture"]) != pipeline.active_architecture:
        raise ValueError("Frozen policy architecture does not match the active inference architecture.")
    if str(policy_payload["feature_layer"]) != pipeline.feature_extractor.feature_layer_name:
        raise ValueError("Frozen policy feature layer does not match the active feature extractor.")
    if int(policy_payload["feature_dimension"]) != pipeline.feature_extractor.feature_dimension:
        raise ValueError("Frozen policy feature dimension does not match the active feature extractor.")
    if str(policy_payload["dataset_fingerprint"]) != pipeline.dataset_fingerprint:
        raise ValueError("Frozen policy dataset fingerprint does not match the active checkpoint.")
    if dict(policy_payload["uncertainty_thresholds"]) != dict(pipeline.uncertainty_thresholds):
        raise ValueError("Frozen policy uncertainty thresholds do not match the active uncertainty artifact.")
    if dict(policy_payload["quality_thresholds"]) != dict(pipeline.quality_thresholds):
        raise ValueError("Frozen policy quality thresholds do not match the active quality artifact.")
    if str(policy_payload["probability_mode"]) != pipeline.default_probability_mode:
        raise ValueError("Frozen policy probability mode does not match the active uncertainty artifact.")
    frozen_counts = dict(frozen_validation_summary["counts"])
    current_counts = dict(validation_summary["counts"])
    if frozen_counts != current_counts:
        raise ValueError("Validation summary does not match the pre-frozen policy summary.")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_policy_results_path = output_dir / "test_policy_results.csv"
    error_capture_path = output_dir / "error_capture.csv"
    synthetic_ood_summary_path = output_dir / "synthetic_ood_summary.csv"
    perturbation_summary_path = output_dir / "perturbation_summary.csv"
    risk_coverage_path = output_dir / "risk_coverage.json"
    risk_coverage_curve_path = output_dir / "risk_coverage_curve.png"
    robustness_metrics_path = output_dir / "robustness_metrics.json"

    pipeline = BrainScanInferencePipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        ood_reference_npz_path=args.ood_reference_npz,
        ood_reference_json_path=args.ood_reference_json,
        quality_thresholds_path=args.quality_thresholds_path,
        uncertainty_metrics_path=args.uncertainty_metrics_path,
        prefer_cuda=args.prefer_cuda,
    )

    validation_manifest = Path(args.validation_manifest)
    test_manifest = Path(args.test_manifest)
    validation_rows, validation_perf = _evaluate_manifest(pipeline, validation_manifest, split_name="val")
    test_rows, test_perf = _evaluate_manifest(pipeline, test_manifest, split_name="test")
    save_csv(test_rows, test_policy_results_path)

    validation_summary = compute_status_summary(validation_rows)
    test_summary = compute_status_summary(test_rows)
    frozen_policy = _load_json(args.policy_path)
    frozen_validation_summary = _load_json(args.frozen_validation_summary_path)
    _verify_frozen_policy(
        pipeline,
        frozen_policy,
        frozen_validation_summary,
        validation_summary,
    )

    correctness = compute_correctness_breakdown(test_rows)
    accept_metrics = compute_selective_metrics(test_rows, accepted_statuses={"ACCEPT"})
    accept_review_metrics = compute_selective_metrics(test_rows, accepted_statuses={"ACCEPT", "REVIEW"})
    error_capture = compute_error_capture(test_rows)

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
    save_csv(error_rows, error_capture_path)

    synthetic_rows = _evaluate_synthetic_ood(pipeline)
    synthetic_summary = summarize_synthetic_ood(synthetic_rows)
    save_csv(synthetic_rows, synthetic_ood_summary_path)

    perturbation_sample_size, perturbation_rows, perturbation_summary = _evaluate_perturbations(
        pipeline,
        test_rows,
        test_manifest,
    )
    save_csv(perturbation_summary, perturbation_summary_path)

    risk_coverage_payload, _risk_coverage_path = _build_risk_coverage_artifact(
        validation_rows,
        test_rows,
        output_json_path=risk_coverage_path,
        output_png_path=risk_coverage_curve_path,
    )

    quality_frequencies = compute_quality_flag_frequencies(test_rows)
    classwise_behavior = compute_classwise_status_counts(test_rows)
    entropy_ood_relationship = compute_entropy_ood_relationship(test_rows)
    ood_score_summary = {
        "validation_clean": summarize_numeric_distribution([float(row["ood_score"]) for row in validation_rows]),
        "test_clean": summarize_numeric_distribution([float(row["ood_score"]) for row in test_rows]),
        "synthetic_ood": summarize_numeric_distribution(
            [float(row["ood_score"]) for row in synthetic_rows if row["ood_score"] != ""]
        ),
        "perturbed": summarize_numeric_distribution([float(row["perturbed_ood_score"]) for row in perturbation_rows]),
    }

    classifier_performance = _load_json(args.classifier_performance_path)
    baseline_robustness_metrics = _load_json(args.baseline_robustness_metrics_path)
    baseline_accuracy = 1.0 - (int(error_capture["wrong_total"]) / max(len(test_rows), 1))
    baseline_risk = float(error_capture["wrong_total"]) / max(len(test_rows), 1)
    conclusion = _build_conclusion(accept_metrics, error_capture, correctness)
    dense_error_status_counts = dict(error_capture["counts"])

    metrics_payload = {
        "policy_artifact": _relative(args.policy_path),
        "generated_timestamp_utc": datetime.now(UTC).isoformat(),
        "policy": frozen_policy,
        "validation_summary": {
            "status": validation_summary,
            "performance": validation_perf,
            "frozen_reference": _relative(args.frozen_validation_summary_path),
        },
        "test_summary": {
            "status": test_summary,
            "performance": test_perf,
        },
        "baseline_classifier_metrics": {
            "accuracy": baseline_accuracy,
            "risk": baseline_risk,
        },
        "accept_only_metrics": accept_metrics,
        "accept_review_metrics": accept_review_metrics,
        "correctness_breakdown": correctness,
        "error_capture": {
            "wrong_total": error_capture["wrong_total"],
            "counts": dense_error_status_counts,
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
        "dense_error_status_counts": dense_error_status_counts,
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
            "classifier_only_latency_ms_reference": classifier_performance["mean_per_image_latency_ms"],
            "classifier_only_throughput_reference": classifier_performance["throughput_images_per_second"],
        },
        "baseline_comparison": {
            "resnet18_safe_accept_count": baseline_robustness_metrics["test_summary"]["status"]["counts"]["ACCEPT"],
            "resnet18_safe_accept_rate": baseline_robustness_metrics["test_summary"]["status"]["rates"]["ACCEPT"],
            "resnet18_unsafe_accepted_error_count": baseline_robustness_metrics["error_capture"]["unsafe_accepted_error_count"],
            "resnet18_error_capture_rate": baseline_robustness_metrics["error_capture"]["error_capture_rate"],
            "resnet18_safe_latency_ms": baseline_robustness_metrics["latency"]["safe_inference_mean_latency_ms"],
            "resnet18_classifier_error_count": baseline_robustness_metrics["error_capture"]["wrong_total"],
        },
        "artifacts": {
            "test_policy_results_csv": _relative(test_policy_results_path),
            "error_capture_csv": _relative(error_capture_path),
            "synthetic_ood_summary_csv": _relative(synthetic_ood_summary_path),
            "perturbation_summary_csv": _relative(perturbation_summary_path),
            "risk_coverage_json": _relative(risk_coverage_path),
            "risk_coverage_curve_png": _relative(risk_coverage_curve_path),
        },
        "conclusion": conclusion,
    }
    save_json(metrics_payload, robustness_metrics_path)

    print("robustness_evaluation_summary=")
    print(f"  policy_artifact={resolve_project_path(args.policy_path)}")
    print(f"  output_dir={output_dir}")
    print(f"  architecture={pipeline.active_architecture}")
    print(f"  test_policy_results={test_policy_results_path}")
    print(f"  test_accept={test_summary['counts']['ACCEPT']}")
    print(f"  test_review={test_summary['counts']['REVIEW']}")
    print(f"  test_abstain={test_summary['counts']['ABSTAIN']}")
    print(f"  classifier_errors={error_capture['wrong_total']}")
    print(f"  unsafe_accepted_error_count={error_capture['unsafe_accepted_error_count']}")
    print(f"  synthetic_ood_abstention_rate={synthetic_summary['ood_abstention_rate']:.6f}")
    print(f"  safe_latency_ms={test_perf['mean_latency_ms']:.6f}")
    print(f"  risk_coverage_json={risk_coverage_path}")
    print(f"  robustness_metrics_json={robustness_metrics_path}")


if __name__ == "__main__":
    main()
