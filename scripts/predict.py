"""Run safe BrainScanAI inference on one image."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.inference import BrainScanInferencePipeline


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run BrainScanAI safe inference on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = BrainScanInferencePipeline()
    result = pipeline.predict(args.image)

    if args.json:
        print(result.to_json())
        return

    print(f"Status: {result.status}")
    if result.prediction is None:
        print("Prediction: WITHHELD")
    else:
        print(f"Prediction: {result.prediction.class_name}")
        print(f"Raw confidence: {result.prediction.raw_confidence * 100:.1f}%")

    if result.uncertainty is not None:
        print(f"Uncertainty: {result.uncertainty.level}")
    if result.ood is not None:
        print(f"OOD: {result.ood.status}")
    if result.quality is not None:
        flags = ", ".join(result.quality.flags) if result.quality.flags else "none"
        print(f"Quality flags: {flags}")
    if result.decision.reasons:
        print("Reasons:")
        for reason in result.decision.reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()
