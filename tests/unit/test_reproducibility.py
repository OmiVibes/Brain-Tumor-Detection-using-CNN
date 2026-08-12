from pathlib import Path
import csv

from PIL import Image
import torch
import yaml

from brainscan.core.config import load_train_config
from brainscan.core.reproducibility import make_torch_generator, set_global_seed
from brainscan.data.loaders import create_dataloaders


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


def test_make_torch_generator_is_seed_reproducible():
    generator_a = make_torch_generator(42)
    generator_b = make_torch_generator(42)

    values_a = torch.rand(5, generator=generator_a)
    values_b = torch.rand(5, generator=generator_b)

    assert torch.equal(values_a, values_b)


def test_global_seed_controls_torch_randomness():
    set_global_seed(123)
    values_a = torch.rand(4)
    set_global_seed(123)
    values_b = torch.rand(4)

    assert torch.equal(values_a, values_b)
