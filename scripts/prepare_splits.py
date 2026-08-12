"""Prepare deterministic train/val/test manifests for BrainScanAI Phase 1B."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.data.audit import build_dataset_fingerprint
from brainscan.data.splits import (
    load_metadata_rows,
    manifests_are_mutually_exclusive,
    prepare_split_manifests,
    summarize_manifest_counts,
    write_dataset_audit_markdown,
    write_manifest_csv,
    write_summary_json,
)


def main() -> None:
    config = load_train_config()
    artifact_dir = PROJECT_ROOT / "artifacts" / "data_audit"
    metadata_path = artifact_dir / "image_metadata.csv"
    summary_path = artifact_dir / "summary.json"
    if not metadata_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Dataset audit artifacts are missing. Run scripts/audit_dataset.py first.")

    metadata_rows = load_metadata_rows(metadata_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_seed = int(config["dataset"]["split_seed"])

    manifests, strategy = prepare_split_manifests(metadata_rows, summary, split_seed=split_seed)
    if not manifests_are_mutually_exclusive(manifests):
        raise ValueError("Split manifests are not mutually exclusive.")

    split_manifest_dir = resolve_project_path(config["dataset"]["split_manifest_dir"])
    for split_name, rows in manifests.items():
        write_manifest_csv(split_manifest_dir / f"{split_name}.csv", rows)

    manifest_summary = summarize_manifest_counts(manifests)
    fingerprint = build_dataset_fingerprint(
        [
            type("Sample", (), {
                "relative_path": row.relative_path,
                "sha256": row.sha256,
                "canonical_class": row.canonical_class,
            })()
            for split_rows in manifests.values()
            for row in split_rows
        ],
        manifest_summary["split_counts"],
    )
    fingerprint["audit_timestamp"] = datetime.now(UTC).isoformat()

    write_summary_json(artifact_dir / "dataset_fingerprint.json", fingerprint)
    write_summary_json(
        artifact_dir / "summary.json",
        {
            **summary,
            "chosen_split_methodology": strategy,
            "final_manifest_counts": manifest_summary,
            "audit_timestamp": datetime.now(UTC).isoformat(),
        },
    )
    write_dataset_audit_markdown(
        PROJECT_ROOT / "docs" / "dataset_audit.md",
        summary=summary,
        strategy=strategy,
        manifest_summary=manifest_summary,
        provenance=summary["dataset_provenance"],
        fingerprint=fingerprint,
    )

    print(f"Prepared manifests in {split_manifest_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Train/Val/Test counts: {manifest_summary['split_counts']}")
    print(f"Strategy: {strategy['strategy']}")


if __name__ == "__main__":
    main()
