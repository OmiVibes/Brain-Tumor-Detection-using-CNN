"""PyTorch dataset and dataloader construction helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.core.reproducibility import make_torch_generator, seed_worker
from brainscan.data.dataset import BrainMRIDataset
from brainscan.data.preprocessing import build_eval_transform, build_train_transform


def create_datasets(
    config_path: str | Path = "configs/train.yaml",
    include_test: bool = True,
) -> dict[str, BrainMRIDataset]:
    config = load_train_config(config_path)
    image_size = config["training"]["image_size"]
    split_manifest_dir = resolve_project_path(config["dataset"]["split_manifest_dir"])

    datasets = {
        "train": BrainMRIDataset(
            manifest_path=split_manifest_dir / "train.csv",
            transform=build_train_transform(image_size),
        ),
        "val": BrainMRIDataset(
            manifest_path=split_manifest_dir / "val.csv",
            transform=build_eval_transform(image_size),
        ),
    }

    if include_test:
        datasets["test"] = BrainMRIDataset(
            manifest_path=split_manifest_dir / "test.csv",
            transform=build_eval_transform(image_size),
        )

    return datasets


def create_dataloaders(
    config_path: str | Path = "configs/train.yaml",
    include_test: bool = True,
) -> dict[str, DataLoader]:
    config = load_train_config(config_path)
    datasets = create_datasets(config_path, include_test=include_test)
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    seed = int(config["training"]["seed"])

    train_generator = make_torch_generator(seed)

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=train_generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        ),
    }

    if include_test:
        loaders["test"] = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    return loaders


def create_train_val_dataloaders(config_path: str | Path = "configs/train.yaml") -> dict[str, DataLoader]:
    return create_dataloaders(config_path, include_test=False)


def count_dataset_classes(dataset: BrainMRIDataset) -> dict[str, int]:
    counts = Counter(record.canonical_class for record in dataset.records)
    return dict(sorted(counts.items()))
