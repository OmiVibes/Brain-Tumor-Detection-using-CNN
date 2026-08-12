"""Deterministic split-manifest preparation for BrainScanAI Phase 1B."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import random

from brainscan.data.constants import CANONICAL_CLASS_TO_ID


@dataclass(frozen=True)
class ManifestRow:
    relative_path: str
    physical_class: str
    canonical_class: str
    class_id: int
    split: str
    sha256: str
    perceptual_hash: str
    subject_id: str


def load_metadata_rows(metadata_csv_path: Path) -> list[dict[str, str]]:
    with metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def choose_split_strategy(summary: dict[str, Any]) -> dict[str, Any]:
    exact_summary = summary["exact_duplicate_summary"]
    if exact_summary["cross_split_groups"] == 0 and exact_summary["cross_class_groups"] == 0:
        return {
            "strategy": "preserve_existing_test_split_with_deterministic_validation_split",
            "justification": (
                "The legacy dataset already contains explicit train/test folders and the Phase 1B audit found "
                "no exact duplicate leakage across the existing split boundaries. The existing test split is "
                "retained for continuity, while a deterministic validation split is carved from the legacy train "
                "partition using duplicate-aware grouped stratification."
            ),
            "validation_fraction_of_legacy_train": 0.2,
        }
    return {
        "strategy": "rebuild_full_dataset_stratified_80_10_10",
        "justification": (
            "Cross-split or cross-class exact duplicate leakage was detected, so the legacy split boundaries "
            "cannot be trusted safely. The full valid dataset is re-split deterministically using duplicate-aware "
            "image-level grouping."
        ),
        "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    }


def _build_group_key(row: dict[str, str]) -> str:
    subject_id = row.get("subject_id") or ""
    if subject_id:
        return f"subject::{subject_id}"
    return f"sha256::{row['sha256']}"


def _group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_build_group_key(row)].append(row)
    return grouped


def _validate_group_labels(grouped_rows: dict[str, list[dict[str, str]]]) -> None:
    for group_key, rows in grouped_rows.items():
        classes = {row["canonical_class"] for row in rows}
        if len(classes) != 1:
            raise ValueError(f"Group {group_key} contains mixed canonical classes: {sorted(classes)}")


def _deterministic_group_split(
    grouped_rows: dict[str, list[dict[str, str]]],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    _validate_group_labels(grouped_rows)
    per_class_groups: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_key, rows in grouped_rows.items():
        per_class_groups[rows[0]["canonical_class"]].append((group_key, rows))

    split_rows: dict[str, list[dict[str, str]]] = {split_name: [] for split_name in fractions}
    split_order = tuple(fractions.keys())

    for canonical_class in sorted(per_class_groups):
        groups = per_class_groups[canonical_class]
        groups.sort(key=lambda item: item[0])
        random.Random(seed + CANONICAL_CLASS_TO_ID[canonical_class]).shuffle(groups)

        total_rows = sum(len(rows) for _, rows in groups)
        target_counts = {
            split_name: int(round(total_rows * fraction))
            for split_name, fraction in fractions.items()
        }
        delta = total_rows - sum(target_counts.values())
        target_counts[split_order[0]] += delta

        current_counts = Counter()
        for group_key, rows in groups:
            del group_key
            candidate_split = min(
                split_order,
                key=lambda split_name: (
                    current_counts[split_name] / max(target_counts[split_name], 1),
                    current_counts[split_name],
                    split_name,
                ),
            )
            split_rows[candidate_split].extend(rows)
            current_counts[candidate_split] += len(rows)

    for split_name in split_rows:
        split_rows[split_name] = sorted(split_rows[split_name], key=lambda row: row["relative_path"])
    return split_rows


def _manifest_rows_from_dicts(rows: list[dict[str, str]], split_name: str) -> list[ManifestRow]:
    manifest_rows: list[ManifestRow] = []
    for row in rows:
        manifest_rows.append(
            ManifestRow(
                relative_path=row["relative_path"],
                physical_class=row["physical_class"],
                canonical_class=row["canonical_class"],
                class_id=int(row["class_id"]),
                split=split_name,
                sha256=row["sha256"],
                perceptual_hash=row.get("perceptual_hash") or "",
                subject_id=row.get("subject_id") or "",
            )
        )
    return manifest_rows


def prepare_split_manifests(
    metadata_rows: list[dict[str, str]],
    summary: dict[str, Any],
    split_seed: int,
) -> tuple[dict[str, list[ManifestRow]], dict[str, Any]]:
    valid_rows = [row for row in metadata_rows if row["readable"] == "True"]
    strategy = choose_split_strategy(summary)

    if strategy["strategy"] == "preserve_existing_test_split_with_deterministic_validation_split":
        legacy_train_rows = [row for row in valid_rows if row["split"] == "train"]
        legacy_test_rows = [row for row in valid_rows if row["split"] == "test"]
        grouped_train = _group_rows(legacy_train_rows)
        train_val_rows = _deterministic_group_split(
            grouped_train,
            fractions={"train": 0.8, "val": 0.2},
            seed=split_seed,
        )
        manifests = {
            "train": _manifest_rows_from_dicts(train_val_rows["train"], "train"),
            "val": _manifest_rows_from_dicts(train_val_rows["val"], "val"),
            "test": _manifest_rows_from_dicts(sorted(legacy_test_rows, key=lambda row: row["relative_path"]), "test"),
        }
    else:
        grouped_rows = _group_rows(valid_rows)
        rebuilt_rows = _deterministic_group_split(
            grouped_rows,
            fractions={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=split_seed,
        )
        manifests = {
            split_name: _manifest_rows_from_dicts(rows, split_name)
            for split_name, rows in rebuilt_rows.items()
        }

    return manifests, strategy


def write_manifest_csv(output_path: Path, rows: list[ManifestRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "physical_class",
        "canonical_class",
        "class_id",
        "split",
        "sha256",
        "perceptual_hash",
        "subject_id",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "relative_path": row.relative_path,
                    "physical_class": row.physical_class,
                    "canonical_class": row.canonical_class,
                    "class_id": row.class_id,
                    "split": row.split,
                    "sha256": row.sha256,
                    "perceptual_hash": row.perceptual_hash,
                    "subject_id": row.subject_id,
                }
            )


def manifests_are_mutually_exclusive(manifests: dict[str, list[ManifestRow]]) -> bool:
    seen_paths: set[str] = set()
    for rows in manifests.values():
        for row in rows:
            if row.relative_path in seen_paths:
                return False
            seen_paths.add(row.relative_path)
    return True


def summarize_manifest_counts(manifests: dict[str, list[ManifestRow]]) -> dict[str, Any]:
    split_counts = {split_name: len(rows) for split_name, rows in manifests.items()}
    per_class_counts: dict[str, dict[str, int]] = {}
    for split_name, rows in manifests.items():
        counter = Counter(row.canonical_class for row in rows)
        per_class_counts[split_name] = dict(sorted(counter.items()))
    return {"split_counts": split_counts, "per_class_counts": per_class_counts}


def write_dataset_audit_markdown(
    output_path: Path,
    summary: dict[str, Any],
    strategy: dict[str, Any],
    manifest_summary: dict[str, Any],
    provenance: dict[str, Any],
    fingerprint: dict[str, Any],
) -> None:
    lines = [
        "# Dataset Audit",
        "",
        "## Dataset Size",
        f"- Total images discovered: {summary['total_images']}",
        f"- Valid readable images: {summary['valid_images']}",
        f"- Corrupt/unreadable images: {summary['corrupt_or_unreadable_images']}",
        f"- Zero-byte files: {summary['zero_byte_files']}",
        "",
        "## Current Train/Test Distribution",
        f"- Train: {summary['split_counts'].get('train', 0)}",
        f"- Test: {summary['split_counts'].get('test', 0)}",
        "",
        "## Per-Class Distribution",
    ]
    for class_name, count in summary["canonical_class_counts"].items():
        lines.append(f"- {class_name}: {count}")

    lines.extend(
        [
            "",
            "## Corruption Findings",
            f"- Unreadable/corrupt images: {summary['corrupt_or_unreadable_images']}",
            f"- Non-image extension files: {summary['non_image_extension_files']}",
            "",
            "## Exact Duplicate Findings",
            f"- Exact duplicate groups: {summary['exact_duplicate_summary']['duplicate_groups']}",
            f"- Cross-split exact duplicate groups: {summary['exact_duplicate_summary']['cross_split_groups']}",
            f"- Cross-class exact duplicate groups: {summary['exact_duplicate_summary']['cross_class_groups']}",
            "",
            "## Near-Duplicate Findings",
            f"- Candidate near-duplicate pairs: {summary['near_duplicate_summary']['candidate_pairs']}",
            f"- Cross-split near-duplicate pairs: {summary['near_duplicate_summary']['cross_split_pairs']}",
            f"- Cross-class near-duplicate pairs: {summary['near_duplicate_summary']['cross_class_pairs']}",
            f"- Algorithm: {summary['near_duplicate_summary']['algorithm']}",
            f"- Distance threshold: {summary['near_duplicate_summary']['distance_threshold']}",
            "",
            "## Subject-Level Leakage",
            "- Reliable subject IDs were not detected from the current legacy filenames.",
            "- Subject-level leakage cannot be ruled out for this dataset.",
            "",
            "## Dataset Provenance",
            f"- dataset_provenance: {provenance['dataset_provenance']}",
            f"- Notes: {provenance['notes']}",
            "",
            "## Chosen Split Methodology",
            f"- Strategy: {strategy['strategy']}",
            f"- Justification: {strategy['justification']}",
            "",
            "## Final Train/Val/Test Counts",
            f"- Train: {manifest_summary['split_counts']['train']}",
            f"- Val: {manifest_summary['split_counts']['val']}",
            f"- Test: {manifest_summary['split_counts']['test']}",
            "",
            "## Final Per-Class Split Counts",
        ]
    )
    for split_name, counts in manifest_summary["per_class_counts"].items():
        lines.append(f"- {split_name}: {counts}")

    lines.extend(
        [
            "",
            "## Known Limitations",
            "- Subject-level leakage cannot be excluded because reliable subject identifiers were not available.",
            "- Near-duplicate detection uses perceptual hashing and should be treated as a review aid, not ground truth.",
            "- Provenance, license, and citation information were not established from the current repository contents.",
            "",
            "## Dataset Fingerprint",
            f"- dataset_fingerprint: `{fingerprint['dataset_fingerprint']}`",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
