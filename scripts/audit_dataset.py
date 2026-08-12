"""Run the Phase 1B dataset audit and write reproducible audit artifacts."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.data.audit import (
    PHASH_THRESHOLD,
    PHASH_VERSION,
    build_dataset_fingerprint,
    find_exact_duplicate_rows,
    find_near_duplicate_rows,
    scan_dataset,
    search_dataset_provenance,
    summarize_samples,
    write_csv_rows,
    write_json,
    write_metadata_csv,
)


def main() -> None:
    config = load_train_config()
    dataset_root = resolve_project_path(config["dataset"]["root"])
    artifact_dir = PROJECT_ROOT / "artifacts" / "data_audit"

    samples = scan_dataset(dataset_root)
    summary = summarize_samples(samples)
    provenance = search_dataset_provenance(PROJECT_ROOT)

    write_metadata_csv(artifact_dir / "image_metadata.csv", samples)
    write_csv_rows(artifact_dir / "exact_duplicates.csv", find_exact_duplicate_rows([sample for sample in samples if sample.readable]))
    write_csv_rows(artifact_dir / "near_duplicates.csv", find_near_duplicate_rows([sample for sample in samples if sample.readable], threshold=PHASH_THRESHOLD))

    fingerprint = build_dataset_fingerprint(samples, summary["split_counts"])
    fingerprint["audit_timestamp"] = datetime.now(UTC).isoformat()
    write_json(artifact_dir / "dataset_fingerprint.json", fingerprint)

    summary_payload = {
        **summary,
        "dataset_root": dataset_root.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_provenance": provenance,
        "audit_algorithm": {
            "content_hash": "sha256",
            "perceptual_hash": PHASH_VERSION,
            "perceptual_hash_threshold": PHASH_THRESHOLD,
        },
        "audit_timestamp": datetime.now(UTC).isoformat(),
    }
    write_json(artifact_dir / "summary.json", summary_payload)

    print(f"Audited {summary['total_images']} files from {dataset_root.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Unreadable/corrupt images: {summary['corrupt_or_unreadable_images']}")
    print(f"Exact duplicate groups: {summary['exact_duplicate_summary']['duplicate_groups']}")
    print(f"Near-duplicate candidate pairs: {summary['near_duplicate_summary']['candidate_pairs']}")


if __name__ == "__main__":
    main()
