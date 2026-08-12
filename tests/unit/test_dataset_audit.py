from pathlib import Path

from PIL import Image

from brainscan.data.audit import (
    compute_dhash,
    compute_sha256,
    find_exact_duplicate_rows,
    inspect_image,
    scan_dataset,
)


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_image_hashing_is_deterministic(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    _make_image(image_path, (10, 20, 30))

    assert compute_sha256(image_path) == compute_sha256(image_path)
    assert compute_dhash(image_path) == compute_dhash(image_path)


def test_corrupt_image_handling_does_not_crash(tmp_path: Path):
    bad_path = tmp_path / "broken.jpg"
    bad_path.write_bytes(b"not-a-real-image")

    readable, metadata = inspect_image(bad_path)

    assert readable is False
    assert metadata["readable"] is False
    assert metadata["error"] is not None


def test_exact_duplicate_rows_detect_duplicates(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image_a = dataset_root / "train" / "glioma" / "a.jpg"
    image_b = dataset_root / "test" / "glioma" / "b.jpg"
    _make_image(image_a, (200, 10, 10))
    image_b.parent.mkdir(parents=True, exist_ok=True)
    image_b.write_bytes(image_a.read_bytes())

    samples = scan_dataset(dataset_root)
    duplicates = find_exact_duplicate_rows(samples)

    assert len(duplicates) == 2
    assert all(row["cross_split_duplicate"] for row in duplicates)
