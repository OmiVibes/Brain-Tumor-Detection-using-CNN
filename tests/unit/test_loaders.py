import csv
from pathlib import Path

from PIL import Image
import yaml

from brainscan.core.config import get_project_root
from brainscan.data.dataset import ManifestRecord
from brainscan.data.loaders import count_dataset_classes, create_dataloaders
from brainscan.data.dataset import BrainMRIDataset
from brainscan.data.preprocessing import build_eval_transform


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "physical_class",
                "canonical_class",
                "class_id",
                "split",
                "sha256",
                "perceptual_hash",
                "subject_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_temp_loader_setup(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    split_dir = tmp_path / "data" / "splits"
    rows_by_split = {"train": [], "val": [], "test": []}
    colors = {
        "glioma": (255, 0, 0),
        "meningioma": (0, 255, 0),
        "notumor": (0, 0, 255),
        "pituitary": (255, 255, 0),
    }
    class_meta = {
        "glioma": ("glioma", 0),
        "meningioma": ("meningioma", 1),
        "pituitary": ("pituitary", 2),
        "notumor": ("no_tumor", 3),
    }

    for split_name in rows_by_split:
        for physical_class, color in colors.items():
            for index in range(2):
                file_name = f"{split_name}_{physical_class}_{index}.jpg"
                image_path = dataset_root / split_name / physical_class / file_name
                _make_image(image_path, color)
                canonical_class, class_id = class_meta[physical_class]
                rows_by_split[split_name].append(
                    {
                        "relative_path": image_path.relative_to(tmp_path).as_posix(),
                        "physical_class": physical_class,
                        "canonical_class": canonical_class,
                        "class_id": str(class_id),
                        "split": split_name,
                        "sha256": f"{split_name}-{physical_class}-{index}",
                        "perceptual_hash": f"{index:016x}",
                        "subject_id": "",
                    }
                )

    for split_name, rows in rows_by_split.items():
        _write_manifest(split_dir / f"{split_name}.csv", rows)

    config = {
        "training": {
            "seed": 123,
            "image_size": [64, 64],
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "num_workers": 0,
        },
        "dataset": {
            "root": "dataset",
            "train_dir": "dataset/train",
            "test_dir": "dataset/test",
            "split_manifest_dir": "data/splits",
            "split_seed": 42,
            "physical_class_folders": ["glioma", "meningioma", "notumor", "pituitary"],
            "canonical_class_names": ["glioma", "meningioma", "pituitary", "no_tumor"],
        },
        "model": {
            "architecture": "legacy_cnn_baseline",
            "num_classes": 4,
            "pretrained": False,
        },
    }
    config_path = tmp_path / "temp_train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_count_dataset_classes_uses_manifest(tmp_path: Path):
    image_path = tmp_path / "dataset" / "train" / "glioma" / "sample.jpg"
    _make_image(image_path, (10, 20, 30))
    dataset = BrainMRIDataset(
        records=[
            ManifestRecord(
                relative_path="dataset/train/glioma/sample.jpg",
                physical_class="glioma",
                canonical_class="glioma",
                class_id=0,
                split="train",
            )
        ],
        transform=build_eval_transform((224, 224)),
        project_root=tmp_path,
    )

    assert count_dataset_classes(dataset) == {"glioma": 1}


def test_no_exact_duplicate_hash_crosses_final_split_manifests():
    project_root = get_project_root()
    split_hashes: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        with (project_root / "data" / "splits" / f"{split_name}.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            split_hashes[split_name] = {row["sha256"] for row in reader}

    assert split_hashes["train"].isdisjoint(split_hashes["val"])
    assert split_hashes["train"].isdisjoint(split_hashes["test"])
    assert split_hashes["val"].isdisjoint(split_hashes["test"])


def test_real_manifest_sizes_match_phase_1b():
    project_root = get_project_root()
    expected = {"train": 5618, "val": 703, "test": 702}
    actual = {}
    for split_name in expected:
        with (project_root / "data" / "splits" / f"{split_name}.csv").open("r", encoding="utf-8", newline="") as handle:
            actual[split_name] = sum(1 for _ in csv.DictReader(handle))
    assert actual == expected


def test_create_dataloaders_respects_config_image_size_and_batch_size(tmp_path: Path):
    config_path = _build_temp_loader_setup(tmp_path)
    loaders = create_dataloaders(config_path)

    images, labels = next(iter(loaders["train"]))

    assert images.shape == (4, 3, 64, 64)
    assert labels.shape == (4,)


def test_train_shuffling_is_seed_reproducible_and_eval_order_is_deterministic(tmp_path: Path):
    config_path = _build_temp_loader_setup(tmp_path)

    first_loaders = create_dataloaders(config_path)
    second_loaders = create_dataloaders(config_path)

    first_train_batch = next(iter(first_loaders["train"]))[1].tolist()
    second_train_batch = next(iter(second_loaders["train"]))[1].tolist()
    first_val_batch = next(iter(first_loaders["val"]))[1].tolist()
    second_val_batch = next(iter(second_loaders["val"]))[1].tolist()
    first_test_batch = next(iter(first_loaders["test"]))[1].tolist()
    second_test_batch = next(iter(second_loaders["test"]))[1].tolist()

    assert first_train_batch == second_train_batch
    assert first_val_batch == second_val_batch
    assert first_test_batch == second_test_batch
