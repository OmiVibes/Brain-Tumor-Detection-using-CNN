"""Precompute per-subject modality normalization stats for Phase 3C training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, make_project_relative_path, resolve_project_path
from brainscan.segmentation.data.dataset import load_nifti_image
from brainscan.segmentation.data.preprocessing import compute_nonzero_stats
from brainscan.segmentation.data.slice_index import load_slice_index_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build segmentation normalization stats without touching test data.")
    parser.add_argument("--config", default="configs/segmentation_2p5d.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/segmentation/normalization_stats/subject_modality_stats.json",
    )
    return parser.parse_args()


def _subject_lookup(records: list, modalities: list[str]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for record in records:
        entry = lookup.setdefault(record.subject_id, {"mask_path": record.mask_path})
        for modality in modalities:
            path_value = record.modality_paths.get(modality, "")
            if path_value:
                entry[modality] = path_value
    return lookup


def _resolve_dataset_file(dataset_root: Path, relative_path: str) -> Path:
    return (dataset_root / relative_path).resolve()


def _load_existing_subjects(output_path: Path) -> dict[str, dict[str, dict[str, float | int | str]]]:
    if not output_path.exists():
        return {}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    subjects_payload = payload.get("subjects", {})
    if not isinstance(subjects_payload, dict):
        raise TypeError("Existing normalization stats artifact is malformed.")
    return subjects_payload


def _save_payload(
    output_path: Path,
    *,
    config_path: str | Path,
    slice_dir: Path,
    modalities: list[str],
    subjects_payload: dict[str, dict[str, dict[str, float | int | str]]],
) -> None:
    payload = {
        "created_from_config": make_project_relative_path(config_path),
        "train_manifest": make_project_relative_path(slice_dir / "train.csv"),
        "val_manifest": make_project_relative_path(slice_dir / "val.csv"),
        "subject_count": len(subjects_payload),
        "modalities": modalities,
        "test_used": False,
        "subjects": subjects_payload,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_segmentation_config(args.config)
    modalities = list(config["dataset"]["modalities"])
    dataset_root = resolve_project_path(config["dataset"]["root"])
    slice_dir = resolve_project_path(config["dataset"]["slice_index_dir"])
    train_records = load_slice_index_records(slice_dir / "train.csv", modalities=modalities)
    val_records = load_slice_index_records(slice_dir / "val.csv", modalities=modalities)
    subjects = _subject_lookup(train_records + val_records, modalities)

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_subjects = _load_existing_subjects(output_path)
    started = time.perf_counter()

    for subject_index, subject_id in enumerate(sorted(subjects), start=1):
        if subject_id in payload_subjects:
            continue
        subject_paths = subjects[subject_id]
        modality_payload: dict[str, dict[str, float | int]] = {}
        for modality in modalities:
            image = load_nifti_image(_resolve_dataset_file(dataset_root, subject_paths[modality]))
            stats = compute_nonzero_stats(np.asarray(image.dataobj, dtype=np.float32))
            modality_payload[modality] = {
                "mean_nonzero": float(stats.mean_nonzero),
                "std_nonzero": float(stats.std_nonzero),
                "nonzero_count": int(stats.nonzero_count),
                "path": subject_paths[modality],
            }
        payload_subjects[subject_id] = modality_payload
        _save_payload(
            output_path,
            config_path=args.config,
            slice_dir=slice_dir,
            modalities=modalities,
            subjects_payload=payload_subjects,
        )
        if subject_index % 10 == 0:
            print(
                f"normalization_progress={len(payload_subjects)}/{len(subjects)} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}"
            )
    _save_payload(
        output_path,
        config_path=args.config,
        slice_dir=slice_dir,
        modalities=modalities,
        subjects_payload=payload_subjects,
    )
    print(f"normalization_stats={output_path}")
    print(f"subject_count={len(payload_subjects)}")
    print("test_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
