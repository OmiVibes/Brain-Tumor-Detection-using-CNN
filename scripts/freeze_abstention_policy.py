"""Freeze a validation-derived abstention policy for a specific BrainScanAI backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import resolve_project_path
from brainscan.inference import BrainScanInferencePipeline
from brainscan.robustness import build_abstention_policy_artifact, save_json
from brainscan.training.train_classifier import get_git_commit


DEFAULT_CONFIG_PATH = Path("configs/train.yaml")
DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
DEFAULT_OOD_REFERENCE_NPZ_PATH = Path("artifacts/robustness/resnet18_baseline/ood_reference.npz")
DEFAULT_OOD_REFERENCE_JSON_PATH = Path("artifacts/robustness/resnet18_baseline/ood_reference.json")
DEFAULT_QUALITY_THRESHOLDS_PATH = Path("artifacts/robustness/resnet18_baseline/quality_thresholds.json")
DEFAULT_UNCERTAINTY_METRICS_PATH = Path("artifacts/calibration/resnet18_baseline/metrics.json")
DEFAULT_POLICY_PATH = Path("artifacts/robustness/resnet18_baseline/abstention_policy.json")
DEFAULT_VALIDATION_SUMMARY_PATH = Path("artifacts/robustness/resnet18_baseline/validation_policy_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a BrainScanAI abstention policy from validation-only artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Training-style config path.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Frozen checkpoint path.")
    parser.add_argument("--ood-reference-npz", default=str(DEFAULT_OOD_REFERENCE_NPZ_PATH), help="OOD reference npz path.")
    parser.add_argument("--ood-reference-json", default=str(DEFAULT_OOD_REFERENCE_JSON_PATH), help="OOD reference metadata json path.")
    parser.add_argument("--quality-thresholds-path", default=str(DEFAULT_QUALITY_THRESHOLDS_PATH), help="Quality thresholds json path.")
    parser.add_argument("--uncertainty-metrics-path", default=str(DEFAULT_UNCERTAINTY_METRICS_PATH), help="Calibration/uncertainty metrics json path.")
    parser.add_argument("--policy-output-path", default=str(DEFAULT_POLICY_PATH), help="Policy artifact output path.")
    parser.add_argument(
        "--validation-summary-output-path",
        default=str(DEFAULT_VALIDATION_SUMMARY_PATH),
        help="Validation policy-summary output path.",
    )
    parser.add_argument("--prefer-cuda", action="store_true", help="Prefer CUDA when available.")
    parser.add_argument("--policy-version", default="phase2b_resnet18_v1", help="Policy artifact version label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BrainScanInferencePipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        ood_reference_npz_path=args.ood_reference_npz,
        ood_reference_json_path=args.ood_reference_json,
        quality_thresholds_path=args.quality_thresholds_path,
        uncertainty_metrics_path=args.uncertainty_metrics_path,
        prefer_cuda=bool(args.prefer_cuda),
    )

    policy_payload, policy_path = build_abstention_policy_artifact(
        checkpoint_path=args.checkpoint_path,
        checkpoint_epoch=int(pipeline.checkpoint["epoch"]),
        dataset_fingerprint=pipeline.dataset_fingerprint,
        ood_metadata=pipeline.ood_metadata,
        quality_payload=pipeline.quality_payload,
        uncertainty_metrics=pipeline.uncertainty_metrics,
        git_commit=get_git_commit(),
        output_path=args.policy_output_path,
        policy_version=args.policy_version,
    )
    validation_summary = pipeline.summarize_manifest(Path(pipeline.config["dataset"]["split_manifest_dir"]) / "val.csv")
    validation_summary["architecture"] = pipeline.active_architecture
    validation_summary["checkpoint"] = str(args.checkpoint_path).replace("\\", "/")
    validation_summary["policy_artifact"] = str(Path(args.policy_output_path)).replace("\\", "/")
    validation_summary_path = save_json(validation_summary, args.validation_summary_output_path)

    print("abstention_policy_summary=")
    print(f"  checkpoint={resolve_project_path(args.checkpoint_path)}")
    print(f"  architecture={pipeline.active_architecture}")
    print(f"  policy_version={policy_payload['policy_version']}")
    print(f"  feature_layer={policy_payload['feature_layer']}")
    print(f"  feature_dimension={policy_payload['feature_dimension']}")
    print(f"  validation_accept={validation_summary['counts']['ACCEPT']}")
    print(f"  validation_review={validation_summary['counts']['REVIEW']}")
    print(f"  validation_abstain={validation_summary['counts']['ABSTAIN']}")
    print(f"  policy_json={resolve_project_path(policy_path)}")
    print(f"  validation_summary_json={resolve_project_path(validation_summary_path)}")


if __name__ == "__main__":
    main()
