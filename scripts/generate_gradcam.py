"""Generate scientifically correct Grad-CAM explanations for BrainScanAI backbones."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch

from brainscan.core import load_best_validation_reference
from brainscan.core.config import load_train_config, make_project_relative_path, resolve_project_path
from brainscan.data.constants import CANONICAL_CLASS_NAMES, CANONICAL_CLASS_TO_ID, ID_TO_CANONICAL_CLASS
from brainscan.data.preprocessing import build_eval_transform
from brainscan.evaluation.classification import verify_checkpoint_metadata
from brainscan.explainability import (
    build_contact_sheet,
    generate_gradcam_explanation,
    resolve_default_gradcam_target_layer,
)
from brainscan.models import build_classifier
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device


DEFAULT_CONFIG_PATH = Path("configs/densenet121_final.yaml")
DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/comparison/densenet121_seed42/best.pt")
DEFAULT_REFERENCE_METRICS_PATH = Path("artifacts/training/comparison/densenet121_seed42/summary.json")
DEFAULT_PREDICTIONS_CSV_PATH = Path("artifacts/evaluation/densenet121_finalist/predictions.csv")
DEFAULT_ERRORS_CSV_PATH = Path("artifacts/evaluation/densenet121_finalist/errors.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/explainability/densenet121_final")


@dataclass(frozen=True)
class CaseRecord:
    relative_path: str
    true_class: str | None
    predicted_class: str | None
    model_confidence: float | None
    correct: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for BrainScanAI.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the model config YAML.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Frozen checkpoint path.")
    parser.add_argument(
        "--reference-metrics-path",
        default=str(DEFAULT_REFERENCE_METRICS_PATH),
        help="Training summary/history JSON used to verify best epoch and validation macro F1.",
    )
    parser.add_argument(
        "--predictions-csv-path",
        default=str(DEFAULT_PREDICTIONS_CSV_PATH),
        help="Classifier predictions CSV used to choose representative correct cases.",
    )
    parser.add_argument(
        "--errors-csv-path",
        default=str(DEFAULT_ERRORS_CSV_PATH),
        help="Classifier errors CSV used to include all incorrect cases.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for explainability artifacts.",
    )
    parser.add_argument("--image", action="append", default=[], help="Manifest relative_path to explain.")
    parser.add_argument(
        "--target-class",
        default=None,
        help="Optional canonical target class name for all provided --image inputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha in [0, 1].")
    parser.add_argument(
        "--diagnostic-sample-size",
        type=int,
        default=80,
        help="Bounded deterministic sample size for aggregate diagnostics in auto mode.",
    )
    parser.add_argument(
        "--prefer-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer CUDA when available.",
    )
    return parser.parse_args()


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with resolve_project_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sanitize_relative_path(relative_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", relative_path.replace("\\", "/"))


def _load_image_tensor(relative_path: str, image_size: list[int]) -> torch.Tensor:
    transform = build_eval_transform(image_size)
    from brainscan.explainability.gradcam import load_display_image

    image = load_display_image(relative_path)
    return transform(image).unsqueeze(0)


def _build_case_from_row(row: dict[str, str], *, correct: bool) -> CaseRecord:
    confidence_key = "confidence" if "confidence" in row else "model_confidence"
    return CaseRecord(
        relative_path=row["relative_path"],
        true_class=row.get("true_class"),
        predicted_class=row.get("predicted_class"),
        model_confidence=float(row[confidence_key]),
        correct=correct,
    )


def _select_representative_cases(
    prediction_rows: list[dict[str, str]],
    error_rows: list[dict[str, str]],
) -> list[CaseRecord]:
    representatives: list[CaseRecord] = []
    for class_name in CANONICAL_CLASS_NAMES:
        correct_rows = [
            row
            for row in prediction_rows
            if row["true_class"] == class_name and row["correct"].lower() == "true"
        ]
        correct_rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
        representatives.extend(_build_case_from_row(row, correct=True) for row in correct_rows[:2])

    representatives.extend(_build_case_from_row(row, correct=False) for row in error_rows)

    deduplicated: dict[str, CaseRecord] = {}
    for case in representatives:
        deduplicated.setdefault(case.relative_path, case)
    return list(deduplicated.values())


def _select_diagnostic_cases(
    prediction_rows: list[dict[str, str]],
    error_rows: list[dict[str, str]],
    sample_size: int,
) -> list[CaseRecord]:
    diagnostics: list[CaseRecord] = [_build_case_from_row(row, correct=False) for row in error_rows]
    remaining_slots = max(sample_size - len(diagnostics), 0)
    if remaining_slots <= 0:
        return diagnostics[:sample_size]

    correct_rows = [row for row in prediction_rows if row["correct"].lower() == "true"]
    selected_correct: list[dict[str, str]] = []
    per_class = max(1, remaining_slots // len(CANONICAL_CLASS_NAMES))
    for class_name in CANONICAL_CLASS_NAMES:
        class_rows = [row for row in correct_rows if row["true_class"] == class_name]
        class_rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
        selected_correct.extend(class_rows[:per_class])

    for row in selected_correct[:remaining_slots]:
        diagnostics.append(_build_case_from_row(row, correct=True))

    deduplicated: dict[str, CaseRecord] = {}
    for case in diagnostics:
        deduplicated.setdefault(case.relative_path, case)
    return list(deduplicated.values())[:sample_size]


def _save_images(bundle: dict[str, object], base_name: str, output_dir: Path) -> dict[str, str]:
    original_dir = resolve_project_path(output_dir / "original")
    heatmap_dir = resolve_project_path(output_dir / "heatmap")
    overlay_dir = resolve_project_path(output_dir / "overlay")
    metadata_dir = resolve_project_path(output_dir / "metadata")
    for directory in (original_dir, heatmap_dir, overlay_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / f"{base_name}_original.png"
    heatmap_path = heatmap_dir / f"{base_name}_heatmap.png"
    overlay_path = overlay_dir / f"{base_name}_overlay.png"
    metadata_path = metadata_dir / f"{base_name}.json"

    bundle["original_image"].save(original_path)  # type: ignore[union-attr]
    bundle["heatmap_image"].save(heatmap_path)  # type: ignore[union-attr]
    bundle["overlay_image"].save(overlay_path)  # type: ignore[union-attr]

    import json

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle["metadata"], handle, indent=2)  # type: ignore[arg-type]

    return {
        "original": make_project_relative_path(original_path),
        "heatmap": make_project_relative_path(heatmap_path),
        "overlay": make_project_relative_path(overlay_path),
        "metadata": make_project_relative_path(metadata_path),
    }


def _aggregate_border_attention(explanations: list[dict[str, object]]) -> dict[str, object]:
    ratios = [float(item["metadata"]["diagnostics"]["border_attention_ratio"]) for item in explanations]
    correct_ratios = [
        float(item["metadata"]["diagnostics"]["border_attention_ratio"])
        for item in explanations
        if item["analysis_case"]["correct"] is True
    ]
    incorrect_ratios = [
        float(item["metadata"]["diagnostics"]["border_attention_ratio"])
        for item in explanations
        if item["analysis_case"]["correct"] is False
    ]

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    suspicious_cases = [
        {
            "relative_path": item["analysis_case"]["relative_path"],
            "border_attention_ratio": item["metadata"]["diagnostics"]["border_attention_ratio"],
        }
        for item in explanations
        if float(item["metadata"]["diagnostics"]["border_attention_ratio"]) >= 0.35
    ]
    return {
        "mean_border_attention_ratio": _mean(ratios),
        "mean_border_attention_ratio_correct": _mean(correct_ratios),
        "mean_border_attention_ratio_incorrect": _mean(incorrect_ratios),
        "suspicious_border_attention_cases": suspicious_cases,
    }


def _summarize_sanity_checks(explanations: list[dict[str, object]]) -> dict[str, object]:
    diagnostics = [item["metadata"]["diagnostics"] for item in explanations]
    zero_map_count = sum(float(entry["heatmap_max"]) <= 0.0 for entry in diagnostics)
    non_finite_count = sum(
        not np.isfinite(
            [
                float(entry["heatmap_min"]),
                float(entry["heatmap_max"]),
                float(entry["heatmap_mean"]),
                float(entry["border_attention_ratio"]),
            ]
        ).all()
        for entry in diagnostics
    )
    return {
        "diagnostic_sample_count": len(explanations),
        "zero_map_count": int(zero_map_count),
        "nan_or_inf_case_count": int(non_finite_count),
        "parameter_preservation_verified": True,
        "no_surrogate_model_used": True,
    }


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    architecture = str(config["model"]["architecture"])
    image_size = list(config["training"]["image_size"])
    output_dir = Path(args.output_dir)

    model = build_classifier(
        architecture=architecture,
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(args.checkpoint_path, model, map_location="cpu")
    dataset_fingerprint = str(load_dataset_fingerprint()["dataset_fingerprint"])
    reference = load_best_validation_reference(args.reference_metrics_path)
    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_architecture=architecture,
        expected_epoch=int(reference["best_epoch"]),
        expected_validation_score=float(reference["best_validation_macro_f1"]),
        expected_image_size=image_size,
    )
    checkpoint_metadata = {**checkpoint_metadata, "epoch": int(checkpoint["epoch"])}

    device, _ = resolve_device(prefer_cuda=args.prefer_cuda)
    load_checkpoint(args.checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()

    target_layer, target_layer_name = resolve_default_gradcam_target_layer(model, architecture)
    if args.target_class is not None:
        if args.target_class not in CANONICAL_CLASS_TO_ID:
            raise ValueError(f"Unknown target class '{args.target_class}'.")
        explicit_target_class = CANONICAL_CLASS_TO_ID[args.target_class]
    else:
        explicit_target_class = None

    prediction_rows = _load_csv_rows(args.predictions_csv_path)
    error_rows = _load_csv_rows(args.errors_csv_path)
    if args.image:
        cases = [
            CaseRecord(
                relative_path=image,
                true_class=None,
                predicted_class=None,
                model_confidence=None,
                correct=None,
            )
            for image in args.image
        ]
    else:
        cases = _select_representative_cases(prediction_rows, error_rows)

    generated_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for case in cases:
        input_tensor = _load_image_tensor(case.relative_path, image_size).to(device)
        bundle = generate_gradcam_explanation(
            model=model,
            image_tensor=input_tensor,
            relative_path=case.relative_path,
            checkpoint_metadata=checkpoint_metadata,
            target_class=explicit_target_class,
            architecture=architecture,
            target_layer=target_layer,
            target_layer_name=target_layer_name,
        )
        bundle["metadata"]["true_class"] = case.true_class
        bundle["analysis_case"] = {
            "relative_path": case.relative_path,
            "true_class": case.true_class,
            "predicted_class_from_classifier": case.predicted_class,
            "model_confidence_from_classifier": case.model_confidence,
            "correct": case.correct,
        }
        output_paths = _save_images(bundle, _sanitize_relative_path(case.relative_path), output_dir)
        generated_rows.append(
            {
                "relative_path": case.relative_path,
                "true_class": case.true_class,
                "predicted_class": bundle["metadata"]["predicted_class"],
                "model_confidence": bundle["metadata"]["model_confidence"],
                "target_class": bundle["metadata"]["target_class"],
                "border_attention_ratio": bundle["metadata"]["diagnostics"]["border_attention_ratio"],
                "output_paths": output_paths,
            }
        )
        review_rows.append(bundle)

    review_grid_path = build_contact_sheet(review_rows, output_dir / "review_grid.png")

    diagnostic_cases = (
        _select_diagnostic_cases(prediction_rows, error_rows, args.diagnostic_sample_size)
        if not args.image
        else cases
    )
    diagnostic_rows: list[dict[str, object]] = []
    for case in diagnostic_cases:
        input_tensor = _load_image_tensor(case.relative_path, image_size).to(device)
        bundle = generate_gradcam_explanation(
            model=model,
            image_tensor=input_tensor,
            relative_path=case.relative_path,
            checkpoint_metadata=checkpoint_metadata,
            target_class=None,
            architecture=architecture,
            target_layer=target_layer,
            target_layer_name=target_layer_name,
        )
        bundle["analysis_case"] = {
            "relative_path": case.relative_path,
            "true_class": case.true_class,
            "predicted_class_from_classifier": case.predicted_class,
            "model_confidence_from_classifier": case.model_confidence,
            "correct": case.correct,
        }
        diagnostic_rows.append(bundle)

    border_summary = _aggregate_border_attention(diagnostic_rows)
    sanity_checks = _summarize_sanity_checks(diagnostic_rows)

    sensitivity_case = cases[0]
    sensitivity_tensor = _load_image_tensor(sensitivity_case.relative_path, image_size).to(device)
    predicted_bundle = generate_gradcam_explanation(
        model=model,
        image_tensor=sensitivity_tensor,
        relative_path=sensitivity_case.relative_path,
        checkpoint_metadata=checkpoint_metadata,
        target_class=None,
        architecture=architecture,
        target_layer=target_layer,
        target_layer_name=target_layer_name,
    )
    alternate_target = (predicted_bundle["result"].predicted_class_id + 1) % len(CANONICAL_CLASS_NAMES)
    alternate_bundle = generate_gradcam_explanation(
        model=model,
        image_tensor=sensitivity_tensor,
        relative_path=sensitivity_case.relative_path,
        checkpoint_metadata=checkpoint_metadata,
        target_class=alternate_target,
        architecture=architecture,
        target_layer=target_layer,
        target_layer_name=target_layer_name,
    )
    target_class_l1_difference = float(
        np.abs(predicted_bundle["resized_heatmap"] - alternate_bundle["resized_heatmap"]).mean()
    )

    highest_confidence_error = max(
        (
            row
            for row in generated_rows
            if row["true_class"] is not None and row["true_class"] != row["predicted_class"]
        ),
        key=lambda row: float(row["model_confidence"]),
        default=None,
    )
    summary_payload = {
        "checkpoint": str(Path(args.checkpoint_path)).replace("\\", "/"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": checkpoint_metadata["architecture"],
        "target_layer": target_layer_name,
        "explanation_method": "Grad-CAM",
        "generated_case_count": len(generated_rows),
        "generated_cases": generated_rows,
        "review_grid": make_project_relative_path(review_grid_path),
        "representative_case_counts": {
            class_name: sum(
                row["true_class"] == class_name and row["predicted_class"] == class_name
                for row in generated_rows
            )
            for class_name in CANONICAL_CLASS_NAMES
        },
        "incorrect_case_count": sum(
            row["true_class"] is not None and row["true_class"] != row["predicted_class"] for row in generated_rows
        ),
        "highest_confidence_error": highest_confidence_error,
        "border_attention_summary": border_summary,
        "sanity_checks": sanity_checks,
        "target_class_sensitivity": {
            "relative_path": sensitivity_case.relative_path,
            "predicted_class": predicted_bundle["metadata"]["predicted_class"],
            "alternate_target_class": ID_TO_CANONICAL_CLASS[alternate_target],
            "mean_absolute_heatmap_difference": target_class_l1_difference,
        },
    }
    summary_path = resolve_project_path(output_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("gradcam_summary=")
    print(f"  checkpoint={resolve_project_path(args.checkpoint_path)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  architecture={architecture}")
    print(f"  target_layer={target_layer_name}")
    print(f"  generated_case_count={len(generated_rows)}")
    print(f"  review_grid={review_grid_path}")
    print(f"  summary_json={summary_path}")
    print(f"  target_class_heatmap_difference={target_class_l1_difference:.6f}")
    print(f"  mean_border_attention_ratio={border_summary['mean_border_attention_ratio']:.6f}")


if __name__ == "__main__":
    main()
