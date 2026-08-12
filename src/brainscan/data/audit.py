"""Dataset audit utilities for BrainScanAI Phase 1B."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import csv
import json
import re

from PIL import Image, UnidentifiedImageError

from brainscan.data.constants import DATASET_FOLDER_TO_CANONICAL, DATASET_FOLDER_TO_ID


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PHASH_VERSION = "dhash-8"
PHASH_THRESHOLD = 4
PROVENANCE_STATUS_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DatasetSample:
    relative_path: str
    split: str
    physical_class: str
    canonical_class: str
    class_id: int
    file_name: str
    extension: str
    file_size_bytes: int
    sha256: str
    perceptual_hash: str | None
    width: int | None
    height: int | None
    aspect_ratio: float | None
    mode: str | None
    channels: int | None
    readable: bool
    is_image_extension: bool
    is_zero_byte: bool
    subject_id: str | None
    error: str | None


def _count_channels(mode: str | None) -> int | None:
    if mode is None:
        return None
    if mode == "L":
        return 1
    if mode == "RGB":
        return 3
    if mode == "RGBA":
        return 4
    bands = Image.getmodebands(mode)
    return bands or None


def compute_sha256(file_path: Path) -> str:
    hasher = sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(file_path: Path, hash_size: int = 8) -> str:
    with Image.open(file_path) as image:
        grayscale = image.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(grayscale.getdata())

    bits: list[str] = []
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bits.append("1" if left > right else "0")

    return f"{int(''.join(bits), 2):016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def infer_subject_id(file_name: str) -> str | None:
    """Infer subject/group ID only when a reliable signal exists.

    The legacy filenames inspected so far mostly look like split/class-prefixed
    serial numbers such as ``Tr-gl_0010.jpg`` or ``Te-pi_0277.jpg``. Those are
    not reliable patient identifiers, so this function returns ``None`` unless a
    less ambiguous subject-style token is present.
    """

    subject_patterns = (
        r"(patient[_-]?\d+)",
        r"(subject[_-]?\d+)",
        r"(case[_-]?\d+)",
        r"(study[_-]?\d+)",
        r"(id[_-]?\d{3,})",
    )
    lower_name = file_name.lower()
    for pattern in subject_patterns:
        match = re.search(pattern, lower_name)
        if match:
            return match.group(1)
    return None


def inspect_image(file_path: Path) -> tuple[bool, dict[str, Any]]:
    size_bytes = file_path.stat().st_size
    is_zero_byte = size_bytes == 0
    extension = file_path.suffix.lower()
    is_image_extension = extension in VALID_IMAGE_EXTENSIONS
    digest = compute_sha256(file_path)

    if is_zero_byte:
        return False, {
            "extension": extension,
            "file_size_bytes": size_bytes,
            "sha256": digest,
            "perceptual_hash": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "mode": None,
            "channels": None,
            "readable": False,
            "is_image_extension": is_image_extension,
            "is_zero_byte": True,
            "error": "zero-byte file",
        }

    try:
        with Image.open(file_path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
        return True, {
            "extension": extension,
            "file_size_bytes": size_bytes,
            "sha256": digest,
            "perceptual_hash": compute_dhash(file_path),
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 6) if height else None,
            "mode": mode,
            "channels": _count_channels(mode),
            "readable": True,
            "is_image_extension": is_image_extension,
            "is_zero_byte": False,
            "error": None,
        }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, {
            "extension": extension,
            "file_size_bytes": size_bytes,
            "sha256": digest,
            "perceptual_hash": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "mode": None,
            "channels": None,
            "readable": False,
            "is_image_extension": is_image_extension,
            "is_zero_byte": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def scan_dataset(dataset_root: Path) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    relative_base = dataset_root.parent.resolve()
    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        split = split_dir.name
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            physical_class = class_dir.name
            canonical_class = DATASET_FOLDER_TO_CANONICAL[physical_class]
            class_id = DATASET_FOLDER_TO_ID[physical_class]
            for file_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                readable, metadata = inspect_image(file_path)
                relative_path = file_path.resolve().relative_to(relative_base).as_posix()
                samples.append(
                    DatasetSample(
                        relative_path=relative_path,
                        split=split,
                        physical_class=physical_class,
                        canonical_class=canonical_class,
                        class_id=class_id,
                        file_name=file_path.name,
                        extension=metadata["extension"],
                        file_size_bytes=metadata["file_size_bytes"],
                        sha256=metadata["sha256"],
                        perceptual_hash=metadata["perceptual_hash"],
                        width=metadata["width"],
                        height=metadata["height"],
                        aspect_ratio=metadata["aspect_ratio"],
                        mode=metadata["mode"],
                        channels=metadata["channels"],
                        readable=metadata["readable"],
                        is_image_extension=metadata["is_image_extension"],
                        is_zero_byte=metadata["is_zero_byte"],
                        subject_id=infer_subject_id(file_path.name),
                        error=metadata["error"],
                    )
                )
    return samples


def find_exact_duplicate_rows(samples: list[DatasetSample]) -> list[dict[str, Any]]:
    grouped: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sha256].append(sample)

    rows: list[dict[str, Any]] = []
    for digest, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        splits = {sample.split for sample in group}
        classes = {sample.canonical_class for sample in group}
        for sample in sorted(group, key=lambda item: item.relative_path):
            rows.append(
                {
                    "hash": digest,
                    "file_path": sample.relative_path,
                    "split": sample.split,
                    "physical_class": sample.physical_class,
                    "canonical_class": sample.canonical_class,
                    "duplicate_group_size": len(group),
                    "cross_split_duplicate": len(splits) > 1,
                    "cross_class_duplicate": len(classes) > 1,
                }
            )
    return rows


class _BKNode:
    def __init__(self, value: str, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, _BKNode] = {}


class BKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: str, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, index)
            return

        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, index)
                return
            node = child

    def query(self, value: str, threshold: int) -> list[int]:
        if self.root is None:
            return []

        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= threshold:
                matches.extend(node.indices)
            lower = distance - threshold
            upper = distance + threshold
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)
        return matches


def find_near_duplicate_rows(samples: list[DatasetSample], threshold: int = PHASH_THRESHOLD) -> list[dict[str, Any]]:
    readable_samples = [sample for sample in samples if sample.readable and sample.perceptual_hash]
    tree = BKTree()
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for index, sample in enumerate(readable_samples):
        for candidate_index in tree.query(sample.perceptual_hash or "", threshold):
            candidate = readable_samples[candidate_index]
            if candidate.relative_path == sample.relative_path:
                continue
            pair = tuple(sorted((candidate.relative_path, sample.relative_path)))
            if pair in seen_pairs:
                continue
            distance = hamming_distance(candidate.perceptual_hash or "", sample.perceptual_hash or "")
            if distance > threshold:
                continue
            seen_pairs.add(pair)
            rows.append(
                {
                    "image_a": candidate.relative_path,
                    "image_b": sample.relative_path,
                    "split_a": candidate.split,
                    "split_b": sample.split,
                    "class_a": candidate.canonical_class,
                    "class_b": sample.canonical_class,
                    "hash_distance": distance,
                }
            )
        tree.add(sample.perceptual_hash or "", index)

    rows.sort(key=lambda row: (row["hash_distance"], row["image_a"], row["image_b"]))
    return rows


def summarize_samples(samples: list[DatasetSample]) -> dict[str, Any]:
    total_images = len(samples)
    valid_samples = [sample for sample in samples if sample.readable]
    corrupt_samples = [sample for sample in samples if not sample.readable]
    zero_byte_samples = [sample for sample in samples if sample.is_zero_byte]
    non_image_extension_samples = [sample for sample in samples if not sample.is_image_extension]

    split_counts = Counter(sample.split for sample in samples)
    physical_counts = Counter(sample.physical_class for sample in samples)
    canonical_counts = Counter(sample.canonical_class for sample in samples)
    split_class_counts: dict[str, dict[str, int]] = {}
    for split in sorted(split_counts):
        split_class_counts[split] = {
            canonical_class: sum(
                1
                for sample in samples
                if sample.split == split and sample.canonical_class == canonical_class
            )
            for canonical_class in sorted(canonical_counts)
        }

    width_values = [sample.width for sample in valid_samples if sample.width is not None]
    height_values = [sample.height for sample in valid_samples if sample.height is not None]
    aspect_values = [sample.aspect_ratio for sample in valid_samples if sample.aspect_ratio is not None]
    size_values = [sample.file_size_bytes for sample in valid_samples]
    mode_counts = Counter(sample.mode for sample in valid_samples if sample.mode)
    extension_counts = Counter(sample.extension for sample in samples)

    exact_duplicate_rows = find_exact_duplicate_rows(valid_samples)
    near_duplicate_rows = find_near_duplicate_rows(valid_samples)

    exact_grouped = defaultdict(list)
    for row in exact_duplicate_rows:
        exact_grouped[row["hash"]].append(row)

    cross_split_exact_groups = sum(
        1 for rows in exact_grouped.values() if any(item["cross_split_duplicate"] for item in rows)
    )
    cross_class_exact_groups = sum(
        1 for rows in exact_grouped.values() if any(item["cross_class_duplicate"] for item in rows)
    )

    cross_split_near_pairs = sum(1 for row in near_duplicate_rows if row["split_a"] != row["split_b"])
    cross_class_near_pairs = sum(1 for row in near_duplicate_rows if row["class_a"] != row["class_b"])

    unusual_dimension_samples = [
        sample.relative_path
        for sample in valid_samples
        if (sample.width and sample.width < 64)
        or (sample.height and sample.height < 64)
        or (
            sample.aspect_ratio is not None
            and (sample.aspect_ratio < 0.5 or sample.aspect_ratio > 2.0)
        )
    ]

    return {
        "total_images": total_images,
        "valid_images": len(valid_samples),
        "corrupt_or_unreadable_images": len(corrupt_samples),
        "zero_byte_files": len(zero_byte_samples),
        "non_image_extension_files": len(non_image_extension_samples),
        "split_counts": dict(sorted(split_counts.items())),
        "physical_class_counts": dict(sorted(physical_counts.items())),
        "canonical_class_counts": dict(sorted(canonical_counts.items())),
        "split_class_counts": split_class_counts,
        "extension_counts": dict(sorted(extension_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "width_summary": _numeric_summary(width_values),
        "height_summary": _numeric_summary(height_values),
        "aspect_ratio_summary": _numeric_summary(aspect_values),
        "file_size_summary_bytes": _numeric_summary(size_values),
        "exact_duplicate_summary": {
            "duplicate_rows": len(exact_duplicate_rows),
            "duplicate_groups": len(exact_grouped),
            "cross_split_groups": cross_split_exact_groups,
            "cross_class_groups": cross_class_exact_groups,
        },
        "near_duplicate_summary": {
            "candidate_pairs": len(near_duplicate_rows),
            "cross_split_pairs": cross_split_near_pairs,
            "cross_class_pairs": cross_class_near_pairs,
            "algorithm": PHASH_VERSION,
            "distance_threshold": PHASH_THRESHOLD,
        },
        "subject_id_inference": {
            "inference_rule": "No reliable subject identifier detected in current legacy filenames",
            "subject_ids_detected": len([sample for sample in samples if sample.subject_id]),
            "cross_split_subject_leakage": None,
            "status": "CANNOT_RULE_OUT",
        },
        "unusual_dimension_samples": unusual_dimension_samples,
        "unreadable_files": [sample.relative_path for sample in corrupt_samples],
    }


def _numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 6),
    }


def search_dataset_provenance(project_root: Path) -> dict[str, Any]:
    candidate_files = [
        project_root / "README.md",
        project_root / "UPGRADE_PLAN.md",
        project_root / "docs" / "legacy_system.md",
        project_root / "main.py",
        project_root / "train_model.py",
        project_root / "test.py",
        project_root / "multi_stage_classification.py",
        project_root / "templates" / "index.html",
    ]
    keywords = ("kaggle", "license", "citation", "source", "dataset", "author", "authors", "url")
    evidence: list[dict[str, str]] = []

    for file_path in candidate_files:
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_number, line in enumerate(lines, start=1):
            lower = line.lower()
            if any(keyword in lower for keyword in keywords):
                evidence.append(
                    {
                        "file": file_path.relative_to(project_root).as_posix(),
                        "line": str(line_number),
                        "text": line.strip(),
                    }
                )

    return {
        "dataset_provenance": PROVENANCE_STATUS_UNKNOWN,
        "evidence_inspected": evidence,
        "notes": "No explicit source URL, license, author, citation, or dataset README establishing provenance was found in the repository files inspected.",
    }


def write_csv_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_csv(output_path: Path, samples: list[DatasetSample]) -> None:
    rows = [asdict(sample) for sample in samples]
    write_csv_rows(output_path, rows)


def build_dataset_fingerprint(samples: list[DatasetSample], split_counts: dict[str, int]) -> dict[str, Any]:
    fingerprint_input = "\n".join(
        f"{sample.relative_path}|{sample.sha256}|{sample.canonical_class}"
        for sample in sorted(samples, key=lambda item: item.relative_path)
    )
    dataset_fingerprint = sha256(fingerprint_input.encode("utf-8")).hexdigest()
    class_counts = Counter(sample.canonical_class for sample in samples)
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "total_images": len(samples),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": split_counts,
        "algorithm": {
            "content_hash": "sha256",
            "perceptual_hash": PHASH_VERSION,
        },
    }


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
