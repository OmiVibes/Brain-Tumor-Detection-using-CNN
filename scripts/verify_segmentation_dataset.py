"""Verify whether the authorized segmentation dataset is present locally."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, resolve_project_path


def main() -> int:
    config = load_segmentation_config("configs/segmentation.yaml")
    dataset_root = resolve_project_path(config["dataset"]["root"])
    if dataset_root.exists():
        print("Segmentation dataset detected.")
        print(f"dataset_root={dataset_root}")
        return 0

    print("Dataset not found.")
    print("")
    print("Please place the authorized dataset under:")
    print(f"{dataset_root}")
    print("")
    print("Expected source for Phase 3A:")
    print("- BraTS adult glioma dataset from the official Synapse challenge page")
    print("- access requires a registered Synapse account and acceptance of the dataset terms")
    print("- raw medical volumes must not be committed to Git")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
