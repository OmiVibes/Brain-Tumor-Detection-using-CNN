"""Inspect real manifest-backed MRI datasets and dataloaders."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from brainscan.core.config import load_train_config
from brainscan.core.reproducibility import set_global_seed
from brainscan.data.loaders import count_dataset_classes, create_dataloaders, create_datasets


def main() -> None:
    config = load_train_config()
    set_global_seed(int(config["training"]["seed"]))

    datasets = create_datasets()
    loaders = create_dataloaders()

    print("[datasets]")
    for split_name, dataset in datasets.items():
        print(f"{split_name}_size={len(dataset)}")
        print(f"{split_name}_class_counts={count_dataset_classes(dataset)}")
        sample_image, sample_label = dataset[0]
        print(
            f"{split_name}_sample_shape={tuple(sample_image.shape)} "
            f"{split_name}_sample_label={sample_label}"
        )

    train_loader = loaders["train"]
    batch_images, batch_labels = next(iter(train_loader))

    print("[batch]")
    print(f"image_shape={tuple(batch_images.shape)}")
    print(f"label_shape={tuple(batch_labels.shape)}")
    print(f"image_dtype={batch_images.dtype}")
    print(f"label_dtype={batch_labels.dtype}")
    print(f"label_ids={batch_labels.tolist()}")
    print(f"has_nan={torch.isnan(batch_images).any().item()}")
    print(f"has_inf={torch.isinf(batch_images).any().item()}")


if __name__ == "__main__":
    main()
