"""Audit an authorized BraTS-style segmentation dataset without modifying it."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainscan.core.config import load_segmentation_config, resolve_project_path
from brainscan.segmentation.data import (
    audit_brats_subjects,
    build_sample_overlay_grid,
    build_segmentation_dataset_fingerprint,
    create_subject_level_manifests,
    discover_brats_subjects,
    summarize_segmentation_audit,
    write_segmentation_manifest_csv,
)
from brainscan.segmentation.data.audit import write_mask_statistics_csv, write_segmentation_json


def main() -> int:
    config = load_segmentation_config("configs/segmentation.yaml")
    dataset_cfg = config["dataset"]
    seed = int(config["reproducibility"]["seed"])
    dataset_root = resolve_project_path(dataset_cfg["root"])
    if not dataset_root.exists():
        print("Dataset not found.")
        print("")
        print("Please place the authorized dataset under:")
        print(dataset_root)
        return 1

    modalities = list(dataset_cfg["modalities"])
    mask_name = str(dataset_cfg["mask_name"])
    source_dataset = str(dataset_cfg["name"])
    source_version = str(dataset_cfg["version"])

    subjects = discover_brats_subjects(
        dataset_root,
        modalities,
        mask_name,
        source_dataset=source_dataset,
        source_version=source_version,
    )
    audited_rows = audit_brats_subjects(subjects, modalities, binary_mode=str(config["segmentation"]["mode"]))
    manifests = create_subject_level_manifests(audited_rows, modalities=modalities, seed=seed)

    split_dir = resolve_project_path(dataset_cfg["split_manifest_dir"])
    for split_name, rows in manifests.items():
        write_segmentation_manifest_csv(split_dir / f"{split_name}.csv", rows, modalities=modalities)

    summary = summarize_segmentation_audit(
        audited_rows,
        modalities,
        dataset_name=source_dataset,
        dataset_version=source_version,
        license_access_status="manual Synapse access with accepted terms",
        manual_download_required=True,
    )
    fingerprint = build_segmentation_dataset_fingerprint(
        audited_rows,
        label_mapping={"background": 0, "ncr_net": 1, "edema": 2, "enhancing_tumor": 3},
        dataset_name=source_dataset,
        dataset_version=source_version,
    )

    artifacts_root = resolve_project_path("artifacts/segmentation/data_audit")
    write_segmentation_json(artifacts_root / "summary.json", summary)
    write_segmentation_json(artifacts_root / "dataset_fingerprint.json", fingerprint)
    write_mask_statistics_csv(artifacts_root / "mask_statistics.csv", audited_rows)
    build_sample_overlay_grid(
        subjects,
        audited_rows,
        modality=modalities[0],
        output_path=artifacts_root / "sample_overlays.png",
    )

    print("segmentation_audit_summary=")
    print(f"  dataset_root={dataset_root}")
    print(f"  subject_count={summary['subject_count']}")
    print(f"  tumor_positive_slice_ratio={summary['tumor_positive_slice_ratio']:.6f}")
    print(f"  fingerprint={fingerprint['dataset_fingerprint']}")
    print(f"  summary_json={artifacts_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
