"""Generate scientifically correct Grad-CAM explanations for the BrainScanAI baseline."""

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

from brainscan.core.config import load_train_config, make_project_relative_path, resolve_project_path
from brainscan.data.constants import CANONICAL_CLASS_NAMES, CANONICAL_CLASS_TO_ID, ID_TO_CANONICAL_CLASS
from brainscan.data.preprocessing import build_eval_transform
from brainscan.explainability import (
    build_contact_sheet,
    generate_gradcam_explanation,
    resolve_default_gradcam_target_layer,
)
from brainscan.models import build_classifier
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device
from brainscan.evaluation.classification import verify_checkpoint_metadata


CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
TRAINING_HISTORY_PATH = Path("artifacts/training/resnet18_baseline_history.json")
PREDICTIONS_CSV_PATH = Path("artifacts/evaluation/resnet18_baseline/predictions.csv")
ERRORS_CSV_PATH = Path("artifacts/evaluation/resnet18_baseline/errors.csv")
OUTPUT_DIR = Path("artifacts/explainability/resnet18_baseline")


@dataclass(frozen=True)
class CaseRecord:
    relative_path: str
    true_class: str | None
    predicted_class: str | None
    model_confidence: float | None
    correct: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for BrainScanAI.")
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
    return parser.parse_args()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with resolve_project_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sanitize_relative_path(relative_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", relative_path.replace("\\", "/"))


def _load_image_tensor(relative_path: str, image_size: list[int]) -> torch.Tensor:
    transform = build_eval_transform(image_size)
    from brainscan.explainability.gradcam import load_display_image

    image = load_display_image(relative_path)
    tensor = transform(image).unsqueeze(0)
    return tensor


def _select_representative_cases() -> list[CaseRecord]:
    prediction_rows = _load_csv_rows(PREDICTIONS_CSV_PATH)
    error_rows = _load_csv_rows(ERRORS_CSV_PATH)

    representatives: list[CaseRecord] = []
    for class_name in CANONICAL_CLASS_NAMES:
        correct_rows = [
            row
            for row in prediction_rows
            if row["true_class"] == class_name and row["correct"].lower() == "true"
        ]
        correct_rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
        for row in correct_rows[:2]:
            representatives.append(
                CaseRecord(
                    relative_path=row["relative_path"],
                    true_class=row["true_class"],
                    predicted_class=row["predicted_class"],
                    model_confidence=float(row["confidence"]),
                    correct=True,
                )
            )

    chosen_error_paths: set[str] = set()
    for row in error_rows:
        confidence = float(row["model_confidence"])
        pair = (row["true_class"], row["predicted_class"])
        if confidence >= 0.90 or pair in {("glioma", "meningioma"), ("meningioma", "glioma")}:
            if row["relative_path"] not in chosen_error_paths:
                representatives.append(
                    CaseRecord(
                        relative_path=row["relative_path"],
                        true_class=row["true_class"],
                        predicted_class=row["predicted_class"],
                        model_confidence=confidence,
                        correct=False,
                    )
                )
                chosen_error_paths.add(row["relative_path"])

    for row in error_rows:
        if len(chosen_error_paths) >= 3:
            break
        if row["relative_path"] in chosen_error_paths:
            continue
        representatives.append(
            CaseRecord(
                relative_path=row["relative_path"],
                true_class=row["true_class"],
                predicted_class=row["predicted_class"],
                model_confidence=float(row["model_confidence"]),
                correct=False,
            )
        )
        chosen_error_paths.add(row["relative_path"])

    deduplicated: dict[str, CaseRecord] = {}
    for case in representatives:
        deduplicated.setdefault(case.relative_path, case)
    return list(deduplicated.values())


def _select_diagnostic_cases(sample_size: int) -> list[CaseRecord]:
    prediction_rows = _load_csv_rows(PREDICTIONS_CSV_PATH)
    error_rows = _load_csv_rows(ERRORS_CSV_PATH)
    diagnostics: list[CaseRecord] = []

    for row in error_rows:
        diagnostics.append(
            CaseRecord(
                relative_path=row["relative_path"],
                true_class=row["true_class"],
                predicted_class=row["predicted_class"],
                model_confidence=float(row["model_confidence"]),
                correct=False,
            )
        )

    remaining_slots = max(sample_size - len(diagnostics), 0)
    correct_rows = [row for row in prediction_rows if row["correct"].lower() == "true"]
    selected_correct: list[dict[str, str]] = []
    for class_name in CANONICAL_CLASS_NAMES:
        class_rows = [row for row in correct_rows if row["true_class"] == class_name]
        class_rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
        selected_correct.extend(class_rows[: max(1, remaining_slots // len(CANONICAL_CLASS_NAMES))])

    for row in selected_correct[:remaining_slots]:
        diagnostics.append(
            CaseRecord(
                relative_path=row["relative_path"],
                true_class=row["true_class"],
                predicted_class=row["predicted_class"],
                model_confidence=float(row["confidence"]),
                correct=True,
            )
        )

    deduplicated: dict[str, CaseRecord] = {}
    for case in diagnostics:
        deduplicated.setdefault(case.relative_path, case)
    return list(deduplicated.values())[:sample_size]


def _save_images(bundle: dict[str, object], base_name: str) -> dict[str, str]:
    original_dir = resolve_project_path(OUTPUT_DIR / "original")
    heatmap_dir = resolve_project_path(OUTPUT_DIR / "heatmap")
    overlay_dir = resolve_project_path(OUTPUT_DIR / "overlay")
    metadata_dir = resolve_project_path(OUTPUT_DIR / "metadata")
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


def main() -> None:
    args = parse_args()
    config = load_train_config("configs/train.yaml")
    checkpoint = load_checkpoint(CHECKPOINT_PATH, build_classifier("resnet18", 4, pretrained=False), map_location="cpu")
    dataset_fingerprint = load_dataset_fingerprint()["dataset_fingerprint"]

    import json

    training_history = json.loads(resolve_project_path(TRAINING_HISTORY_PATH).read_text(encoding="utf-8"))
    best_history_entry = next(entry for entry in training_history if int(entry["epoch"]) == 9)
    checkpoint_metadata = verify_checkpoint_metadata(
        checkpoint,
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_validation_score=float(best_history_entry["val_macro_f1"]),
    )
    checkpoint_metadata = {**checkpoint_metadata, "epoch": int(checkpoint["epoch"])}

    device, _ = resolve_device(prefer_cuda=True)
    model = build_classifier("resnet18", 4, pretrained=False)
    load_checkpoint(CHECKPOINT_PATH, model, map_location=device)
    model.to(device)
    model.eval()

    target_layer, target_layer_name = resolve_default_gradcam_target_layer(model, "resnet18")
    if args.target_class is not None:
        if args.target_class not in CANONICAL_CLASS_TO_ID:
            raise ValueError(f"Unknown target class '{args.target_class}'.")
        explicit_target_class = CANONICAL_CLASS_TO_ID[args.target_class]
    else:
        explicit_target_class = None

    if args.image:
        cases = [CaseRecord(relative_path=image, true_class=None, predicted_class=None, model_confidence=None, correct=None) for image in args.image]
    else:
        cases = _select_representative_cases()

    generated_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for case in cases:
        input_tensor = _load_image_tensor(case.relative_path, config["training"]["image_size"]).to(device)
        bundle = generate_gradcam_explanation(
            model=model,
            image_tensor=input_tensor,
            relative_path=case.relative_path,
            checkpoint_metadata=checkpoint_metadata,
            target_class=explicit_target_class,
            architecture="resnet18",
            target_layer=target_layer,
            target_layer_name=target_layer_name,
        )
        bundle["metadata"]["true_class"] = case.true_class
        bundle["analysis_case"] = {
            "relative_path": case.relative_path,
            "true_class": case.true_class,
            "predicted_class_from_phase1e": case.predicted_class,
            "model_confidence_from_phase1e": case.model_confidence,
            "correct": case.correct,
        }
        base_name = _sanitize_relative_path(case.relative_path)
        output_paths = _save_images(bundle, base_name)
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

    review_grid_path = build_contact_sheet(review_rows, OUTPUT_DIR / "review_grid.png")

    diagnostic_cases = _select_diagnostic_cases(args.diagnostic_sample_size) if not args.image else cases
    diagnostic_rows: list[dict[str, object]] = []
    for case in diagnostic_cases:
        input_tensor = _load_image_tensor(case.relative_path, config["training"]["image_size"]).to(device)
        bundle = generate_gradcam_explanation(
            model=model,
            image_tensor=input_tensor,
            relative_path=case.relative_path,
            checkpoint_metadata=checkpoint_metadata,
            target_class=None,
            architecture="resnet18",
            target_layer=target_layer,
            target_layer_name=target_layer_name,
        )
        bundle["analysis_case"] = {
            "relative_path": case.relative_path,
            "true_class": case.true_class,
            "predicted_class_from_phase1e": case.predicted_class,
            "model_confidence_from_phase1e": case.model_confidence,
            "correct": case.correct,
        }
        diagnostic_rows.append(bundle)

    border_summary = _aggregate_border_attention(diagnostic_rows)

    sensitivity_case = cases[0]
    sensitivity_tensor = _load_image_tensor(sensitivity_case.relative_path, config["training"]["image_size"]).to(device)
    predicted_bundle = generate_gradcam_explanation(
        model=model,
        image_tensor=sensitivity_tensor,
        relative_path=sensitivity_case.relative_path,
        checkpoint_metadata=checkpoint_metadata,
        target_class=None,
        architecture="resnet18",
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
        architecture="resnet18",
        target_layer=target_layer,
        target_layer_name=target_layer_name,
    )
    target_class_l1_difference = float(
        np.abs(predicted_bundle["resized_heatmap"] - alternate_bundle["resized_heatmap"]).mean()
    )

    summary_payload = {
        "checkpoint": str(CHECKPOINT_PATH).replace("\\", "/"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": checkpoint_metadata["architecture"],
        "target_layer": target_layer_name,
        "explanation_method": "Grad-CAM",
        "generated_case_count": len(generated_rows),
        "generated_cases": generated_rows,
        "review_grid": make_project_relative_path(review_grid_path),
        "border_attention_summary": border_summary,
        "target_class_sensitivity": {
            "relative_path": sensitivity_case.relative_path,
            "predicted_class": predicted_bundle["metadata"]["predicted_class"],
            "alternate_target_class": ID_TO_CANONICAL_CLASS[alternate_target],
            "mean_absolute_heatmap_difference": target_class_l1_difference,
        },
    }
    summary_path = resolve_project_path(OUTPUT_DIR / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("gradcam_summary=")
    print(f"  checkpoint={resolve_project_path(CHECKPOINT_PATH)}")
    print(f"  checkpoint_epoch={checkpoint['epoch']}")
    print(f"  target_layer={target_layer_name}")
    print(f"  generated_case_count={len(generated_rows)}")
    print(f"  review_grid={review_grid_path}")
    print(f"  summary_json={summary_path}")
    print(f"  target_class_heatmap_difference={target_class_l1_difference:.6f}")
    print(f"  mean_border_attention_ratio={border_summary['mean_border_attention_ratio']:.6f}")


if __name__ == "__main__":
    main()
