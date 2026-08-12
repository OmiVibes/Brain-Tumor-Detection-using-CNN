from pathlib import Path

from PIL import Image
import pytest
import torch

from brainscan.data.constants import CANONICAL_CLASS_TO_ID
from brainscan.data.dataset import BrainMRIDataset, ManifestRecord
from brainscan.data.preprocessing import build_eval_transform


def _make_rgb_image(path: Path, color: tuple[int, int, int] = (120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path)


def test_dataset_length_matches_manifest_length(tmp_path: Path):
    image_path = tmp_path / "dataset" / "train" / "glioma" / "sample.jpg"
    _make_rgb_image(image_path)
    records = [
        ManifestRecord(
            relative_path="dataset/train/glioma/sample.jpg",
            physical_class="glioma",
            canonical_class="glioma",
            class_id=0,
            split="train",
        )
    ]
    dataset = BrainMRIDataset(records=records, transform=build_eval_transform((224, 224)), project_root=tmp_path)

    assert len(dataset) == 1


def test_dataset_uses_manifest_class_id_not_folder_name(tmp_path: Path):
    image_path = tmp_path / "dataset" / "train" / "glioma" / "sample.jpg"
    _make_rgb_image(image_path)
    records = [
        ManifestRecord(
            relative_path="dataset/train/glioma/sample.jpg",
            physical_class="glioma",
            canonical_class="glioma",
            class_id=CANONICAL_CLASS_TO_ID["glioma"],
            split="train",
        )
    ]
    dataset = BrainMRIDataset(records=records, transform=build_eval_transform((224, 224)), project_root=tmp_path)

    _, label = dataset[0]
    assert label == CANONICAL_CLASS_TO_ID["glioma"]


def test_dataset_returns_three_channel_tensor(tmp_path: Path):
    gray_path = tmp_path / "dataset" / "train" / "notumor" / "sample.png"
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 40), color=180).save(gray_path)
    records = [
        ManifestRecord(
            relative_path="dataset/train/notumor/sample.png",
            physical_class="notumor",
            canonical_class="no_tumor",
            class_id=CANONICAL_CLASS_TO_ID["no_tumor"],
            split="train",
        )
    ]
    dataset = BrainMRIDataset(records=records, transform=build_eval_transform((224, 224)), project_root=tmp_path)

    image_tensor, label = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 224, 224)
    assert label == CANONICAL_CLASS_TO_ID["no_tumor"]


def test_missing_image_raises_useful_exception(tmp_path: Path):
    records = [
        ManifestRecord(
            relative_path="dataset/train/glioma/missing.jpg",
            physical_class="glioma",
            canonical_class="glioma",
            class_id=0,
            split="train",
        )
    ]
    dataset = BrainMRIDataset(records=records, transform=build_eval_transform((224, 224)), project_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        dataset[0]


def test_corrupt_image_raises_useful_exception(tmp_path: Path):
    bad_path = tmp_path / "dataset" / "train" / "glioma" / "broken.jpg"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not-an-image")
    records = [
        ManifestRecord(
            relative_path="dataset/train/glioma/broken.jpg",
            physical_class="glioma",
            canonical_class="glioma",
            class_id=0,
            split="train",
        )
    ]
    dataset = BrainMRIDataset(records=records, transform=build_eval_transform((224, 224)), project_root=tmp_path)

    with pytest.raises(ValueError, match="broken.jpg"):
        dataset[0]
