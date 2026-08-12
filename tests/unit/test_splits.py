from pathlib import Path

from PIL import Image

from brainscan.data.splits import (
    load_metadata_rows,
    manifests_are_mutually_exclusive,
    prepare_split_manifests,
    summarize_manifest_counts,
    write_manifest_csv,
)
from brainscan.data.audit import scan_dataset, summarize_samples, write_metadata_csv, write_json


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _build_temporary_dataset(tmp_path: Path) -> tuple[Path, list[dict[str, str]], dict]:
    dataset_root = tmp_path / "dataset"
    colors = {
        "glioma": (255, 0, 0),
        "meningioma": (0, 255, 0),
        "notumor": (0, 0, 255),
        "pituitary": (255, 255, 0),
    }
    for physical_class, color in colors.items():
        for index in range(4):
            _make_image(dataset_root / "train" / physical_class / f"Tr-{physical_class[:2]}_{index:04d}.jpg", color)
        for index in range(2):
            _make_image(dataset_root / "test" / physical_class / f"Te-{physical_class[:2]}_{index:04d}.jpg", color)

    samples = scan_dataset(dataset_root)
    summary = summarize_samples(samples)
    metadata_path = tmp_path / "image_metadata.csv"
    summary_path = tmp_path / "summary.json"
    write_metadata_csv(metadata_path, samples)
    write_json(summary_path, summary)
    return dataset_root, load_metadata_rows(metadata_path), summary


def test_manifest_paths_are_relative_and_class_ids_come_from_mapping(tmp_path: Path):
    _, metadata_rows, summary = _build_temporary_dataset(tmp_path)
    manifests, _ = prepare_split_manifests(metadata_rows, summary, split_seed=42)

    for split_rows in manifests.values():
        for row in split_rows:
            assert not Path(row.relative_path).is_absolute()
            assert row.class_id in {0, 1, 2, 3}


def test_splits_are_mutually_exclusive_and_cover_all_valid_samples(tmp_path: Path):
    _, metadata_rows, summary = _build_temporary_dataset(tmp_path)
    manifests, _ = prepare_split_manifests(metadata_rows, summary, split_seed=42)

    assert manifests_are_mutually_exclusive(manifests) is True
    total_manifest_rows = sum(len(rows) for rows in manifests.values())
    assert total_manifest_rows == len([row for row in metadata_rows if row["readable"] == "True"])


def test_same_seed_produces_identical_manifests(tmp_path: Path):
    _, metadata_rows, summary = _build_temporary_dataset(tmp_path)
    manifests_a, _ = prepare_split_manifests(metadata_rows, summary, split_seed=42)
    manifests_b, _ = prepare_split_manifests(metadata_rows, summary, split_seed=42)

    assert manifests_a == manifests_b


def test_manifest_writing_is_deterministic(tmp_path: Path):
    _, metadata_rows, summary = _build_temporary_dataset(tmp_path)
    manifests, _ = prepare_split_manifests(metadata_rows, summary, split_seed=42)

    out_a = tmp_path / "train_a.csv"
    out_b = tmp_path / "train_b.csv"
    write_manifest_csv(out_a, manifests["train"])
    write_manifest_csv(out_b, manifests["train"])

    assert out_a.read_bytes() == out_b.read_bytes()
    manifest_summary = summarize_manifest_counts(manifests)
    assert manifest_summary["split_counts"]["test"] > 0
