"""NIfTI-based segmentation dataset audit helpers for BrainScanAI Phase 3A."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import csv
import json

import nibabel as nib
from nibabel.orientations import aff2axcodes
import numpy as np
from PIL import Image

from brainscan.core.config import make_project_relative_path, resolve_project_path


NIFTI_EXTENSIONS = (".nii", ".nii.gz")


@dataclass(frozen=True)
class BraTSSegmentationSubject:
    """One BraTS-style subject with multimodal MRI paths and one segmentation mask."""

    subject_id: str
    modality_paths: dict[str, str]
    mask_path: str
    source_dataset: str
    source_version: str


def compute_file_sha256(path_value: str | Path) -> str:
    """Compute a deterministic hash for an image or mask file."""
    resolved_path = resolve_project_path(path_value)
    hasher = sha256()
    with resolved_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_nifti_file(path: Path) -> bool:
    return any(str(path).lower().endswith(extension) for extension in NIFTI_EXTENSIONS)


def _split_nifti_stem(file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith(".nii.gz"):
        return file_name[:-7]
    if lower.endswith(".nii"):
        return file_name[:-4]
    return Path(file_name).stem


def _normalize_subject_id(subject_dir: Path) -> str:
    subject_id = subject_dir.name.strip()
    if not subject_id:
        raise ValueError(f"Encountered empty subject directory name at {subject_dir}.")
    return subject_id


def _looks_like_brats_subject_dir(path: Path, modalities: list[str], mask_name: str) -> bool:
    if not path.is_dir():
        return False
    subject_id = path.name.strip()
    if not subject_id:
        return False
    file_map = _subject_file_map(path)
    expected_suffixes = [f"{subject_id}-{modality}" for modality in modalities]
    expected_suffixes.append(f"{subject_id}-{mask_name}")
    return any(suffix in file_map for suffix in expected_suffixes)


def resolve_brats_subject_root(dataset_root: str | Path, modalities: list[str], mask_name: str) -> Path:
    """Resolve the directory that directly contains BraTS subject folders."""
    resolved_root = resolve_project_path(dataset_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Segmentation dataset root not found: {resolved_root}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Segmentation dataset root is not a directory: {resolved_root}")

    direct_children = sorted(path for path in resolved_root.iterdir() if path.is_dir())
    if any(_looks_like_brats_subject_dir(path, modalities, mask_name) for path in direct_children):
        return resolved_root

    if len(direct_children) == 1:
        nested_root = direct_children[0]
        nested_children = sorted(path for path in nested_root.iterdir() if path.is_dir())
        if any(_looks_like_brats_subject_dir(path, modalities, mask_name) for path in nested_children):
            return nested_root

    raise ValueError(
        "Could not find BraTS subject directories under the configured segmentation dataset root: "
        f"{resolved_root}"
    )


def _subject_file_map(subject_dir: Path) -> dict[str, Path]:
    file_map: dict[str, Path] = {}
    for file_path in sorted(subject_dir.iterdir()):
        if not file_path.is_file() or not _is_nifti_file(file_path):
            continue
        file_map[_split_nifti_stem(file_path.name)] = file_path
    return file_map


def discover_brats_subjects(
    dataset_root: str | Path,
    modalities: list[str],
    mask_name: str,
    *,
    source_dataset: str,
    source_version: str,
) -> list[BraTSSegmentationSubject]:
    """Discover BraTS-style subject folders without mutating any source data."""
    resolved_root = resolve_brats_subject_root(dataset_root, modalities, mask_name)

    subjects: list[BraTSSegmentationSubject] = []
    for subject_dir in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
        subject_id = _normalize_subject_id(subject_dir)
        file_map = _subject_file_map(subject_dir)
        modality_paths: dict[str, str] = {}
        for modality in modalities:
            candidate = file_map.get(f"{subject_id}-{modality}")
            if candidate is not None:
                modality_paths[modality] = make_project_relative_path(candidate)

        mask_candidate = file_map.get(f"{subject_id}-{mask_name}")
        mask_path = make_project_relative_path(mask_candidate) if mask_candidate is not None else ""
        subjects.append(
            BraTSSegmentationSubject(
                subject_id=subject_id,
                modality_paths=modality_paths,
                mask_path=mask_path,
                source_dataset=source_dataset,
                source_version=source_version,
            )
        )
    return subjects


def load_nifti_volume(path_value: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI volume for audit-time inspection only."""
    resolved_path = resolve_project_path(path_value)
    image = nib.load(str(resolved_path))
    array = np.asanyarray(image.dataobj)
    return array, image


def _float_list(values: tuple[float, ...] | list[float] | np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _shape_tuple(values: tuple[int, ...] | list[int] | np.ndarray) -> list[int]:
    return [int(value) for value in values]


def _mask_label_values(mask_array: np.ndarray) -> list[int]:
    unique_values = np.unique(mask_array)
    rounded = []
    for value in unique_values.tolist():
        if float(value).is_integer():
            rounded.append(int(value))
        else:
            rounded.append(int(round(float(value))))
    return sorted(set(rounded))


def _subject_hash_inputs(subject: BraTSSegmentationSubject) -> list[str]:
    values = [subject.subject_id, subject.mask_path, subject.source_dataset, subject.source_version]
    for modality_name in sorted(subject.modality_paths):
        values.append(f"{modality_name}:{subject.modality_paths[modality_name]}")
    return values


def _load_audit_checkpoint_rows(checkpoint_path: str | Path) -> dict[str, dict[str, object]]:
    resolved_path = resolve_project_path(checkpoint_path)
    if not resolved_path.exists():
        return {}

    cached_rows: dict[str, dict[str, object]] = {}
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            cached_rows[str(row["subject_id"])] = row
    return cached_rows


def _append_audit_checkpoint_row(checkpoint_path: str | Path, row: dict[str, object]) -> None:
    resolved_path = resolve_project_path(checkpoint_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _audit_single_subject(
    subject: BraTSSegmentationSubject,
    modalities: list[str],
    *,
    binary_mode: str,
) -> dict[str, object]:
    modality_shapes: dict[str, list[int] | None] = {}
    modality_spacings: dict[str, list[float] | None] = {}
    modality_orientations: dict[str, str | None] = {}
    unreadable_modalities: list[str] = []
    subject_arrays: dict[str, np.ndarray] = {}
    subject_images: dict[str, nib.Nifti1Image] = {}

    for modality in modalities:
        modality_path = subject.modality_paths.get(modality, "")
        if not modality_path:
            modality_shapes[modality] = None
            modality_spacings[modality] = None
            modality_orientations[modality] = None
            continue
        try:
            array, image = load_nifti_volume(modality_path)
        except Exception:
            modality_shapes[modality] = None
            modality_spacings[modality] = None
            modality_orientations[modality] = None
            unreadable_modalities.append(modality)
            continue

        subject_arrays[modality] = array
        subject_images[modality] = image
        modality_shapes[modality] = _shape_tuple(array.shape)
        modality_spacings[modality] = _float_list(image.header.get_zooms()[: array.ndim])
        modality_orientations[modality] = "".join(aff2axcodes(image.affine))

    mask_readable = False
    mask_shape: list[int] | None = None
    mask_spacing: list[float] | None = None
    mask_orientation: str | None = None
    mask_label_values: list[int] = []
    tumor_voxel_count = 0
    tumor_fraction = 0.0
    num_positive_slices = 0
    total_slices = 0
    empty_mask = True
    unreadable_mask = False
    shape_mismatch = False
    spacing_mismatch = False
    orientation_mismatch = False

    if subject.mask_path:
        try:
            mask_array, mask_image = load_nifti_volume(subject.mask_path)
            mask_readable = True
            mask_shape = _shape_tuple(mask_array.shape)
            mask_spacing = _float_list(mask_image.header.get_zooms()[: mask_array.ndim])
            mask_orientation = "".join(aff2axcodes(mask_image.affine))
            mask_label_values = _mask_label_values(mask_array)
            binary_mask = mask_array > 0
            tumor_voxel_count = int(binary_mask.sum())
            empty_mask = tumor_voxel_count == 0
            tumor_fraction = float(tumor_voxel_count / binary_mask.size) if binary_mask.size else 0.0
            if mask_array.ndim >= 3:
                total_slices = int(mask_array.shape[2])
                num_positive_slices = int(np.count_nonzero(binary_mask.reshape(-1, mask_array.shape[2]).any(axis=0)))
            else:
                total_slices = 1
                num_positive_slices = int(bool(tumor_voxel_count))

            reference_modality = next((mod for mod in modalities if mod in subject_arrays), None)
            if reference_modality is not None:
                reference_array = subject_arrays[reference_modality]
                reference_image = subject_images[reference_modality]
                shape_mismatch = tuple(reference_array.shape) != tuple(mask_array.shape)
                spacing_mismatch = tuple(reference_image.header.get_zooms()[: mask_array.ndim]) != tuple(
                    mask_image.header.get_zooms()[: mask_array.ndim]
                )
                orientation_mismatch = "".join(aff2axcodes(reference_image.affine)) != "".join(
                    aff2axcodes(mask_image.affine)
                )
        except Exception:
            unreadable_mask = True

    return {
        "subject_id": subject.subject_id,
        "source_dataset": subject.source_dataset,
        "source_version": subject.source_version,
        "mask_path": subject.mask_path,
        "modality_paths": dict(subject.modality_paths),
        "modality_count_present": sum(1 for modality in modalities if subject.modality_paths.get(modality)),
        "missing_modalities": [modality for modality in modalities if not subject.modality_paths.get(modality)],
        "unreadable_modalities": unreadable_modalities,
        "mask_present": bool(subject.mask_path),
        "mask_readable": mask_readable,
        "unreadable_mask": unreadable_mask,
        "modality_shapes": modality_shapes,
        "modality_spacings": modality_spacings,
        "modality_orientations": modality_orientations,
        "modality_file_hashes": {
            modality: compute_file_sha256(subject.modality_paths[modality])
            for modality in modalities
            if subject.modality_paths.get(modality)
        },
        "mask_shape": mask_shape,
        "mask_spacing": mask_spacing,
        "mask_orientation": mask_orientation,
        "mask_file_hash": compute_file_sha256(subject.mask_path) if subject.mask_path else "",
        "shape_mismatch": shape_mismatch,
        "spacing_mismatch": spacing_mismatch,
        "orientation_mismatch": orientation_mismatch,
        "mask_label_values": mask_label_values,
        "empty_mask": empty_mask,
        "tumor_voxel_count": tumor_voxel_count,
        "tumor_fraction": tumor_fraction,
        "total_slices": total_slices,
        "positive_slices": num_positive_slices,
        "binary_segmentation_mode": binary_mode,
        "subject_hash_basis": _subject_hash_inputs(subject),
    }


def audit_brats_subjects(
    subjects: list[BraTSSegmentationSubject],
    modalities: list[str],
    *,
    binary_mode: str = "binary_whole_tumor",
    checkpoint_path: str | Path | None = None,
    progress_interval: int = 25,
) -> list[dict[str, object]]:
    """Audit BraTS-style subjects without modifying source volumes."""
    if binary_mode != "binary_whole_tumor":
        raise ValueError(f"Unsupported segmentation mode for Phase 3A audit: {binary_mode}")

    cached_rows = _load_audit_checkpoint_rows(checkpoint_path) if checkpoint_path is not None else {}
    rows: list[dict[str, object]] = []
    total_subjects = len(subjects)
    for index, subject in enumerate(subjects, start=1):
        cached_row = cached_rows.get(subject.subject_id)
        expected_hash_basis = _subject_hash_inputs(subject)
        if cached_row is not None and cached_row.get("subject_hash_basis") == expected_hash_basis:
            row = cached_row
        else:
            row = _audit_single_subject(subject, modalities, binary_mode=binary_mode)
            if checkpoint_path is not None:
                _append_audit_checkpoint_row(checkpoint_path, row)
        rows.append(row)
        if progress_interval > 0 and (index == total_subjects or index % progress_interval == 0):
            print(f"[segmentation-audit] processed {index}/{total_subjects} subjects")
    return rows


def build_segmentation_dataset_fingerprint(
    audited_rows: list[dict[str, object]],
    *,
    label_mapping: dict[str, int],
    dataset_name: str,
    dataset_version: str,
) -> dict[str, object]:
    """Build a deterministic fingerprint for the segmentation dataset foundation."""
    normalized_lines: list[str] = []
    for row in sorted(audited_rows, key=lambda item: str(item["subject_id"])):
        modality_paths = row["modality_paths"]
        assert isinstance(modality_paths, dict)
        modality_hashes = row["modality_file_hashes"]
        assert isinstance(modality_hashes, dict)
        modality_bits = "|".join(
            f"{name}:{modality_paths[name]}:{modality_hashes.get(name, '')}"
            for name in sorted(modality_paths)
        )
        label_values = ",".join(str(value) for value in row["mask_label_values"])
        normalized_lines.append(
            "||".join(
                [
                    str(row["subject_id"]),
                    modality_bits,
                    str(row["mask_path"]),
                    str(row["mask_file_hash"]),
                    label_values,
                    dataset_name,
                    dataset_version,
                ]
            )
        )

    digest = sha256("\n".join(normalized_lines).encode("utf-8")).hexdigest()
    return {
        "dataset_fingerprint": digest,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "subject_count": len(audited_rows),
        "label_mapping": label_mapping,
        "fingerprint_algorithm": "sha256(subject_ids|relative_paths|file_hashes|label_values|dataset_version)",
    }


def summarize_segmentation_audit(
    audited_rows: list[dict[str, object]],
    modalities: list[str],
    *,
    dataset_name: str,
    dataset_version: str,
    license_access_status: str,
    manual_download_required: bool,
) -> dict[str, object]:
    """Summarize an audited segmentation dataset."""
    subject_count = len(audited_rows)
    modality_coverage = {
        modality: sum(1 for row in audited_rows if row["modality_paths"].get(modality))
        for modality in modalities
    }
    missing_modalities = sum(1 for row in audited_rows if row["missing_modalities"])
    missing_masks = sum(1 for row in audited_rows if not bool(row["mask_present"]))
    corrupt_volumes = sum(
        1
        for row in audited_rows
        if row["unreadable_mask"] or bool(row["unreadable_modalities"])
    )
    empty_masks = sum(1 for row in audited_rows if bool(row["empty_mask"]))
    shape_mismatches = sum(1 for row in audited_rows if bool(row["shape_mismatch"]))
    spacing_mismatches = sum(1 for row in audited_rows if bool(row["spacing_mismatch"]))
    orientation_mismatches = sum(1 for row in audited_rows if bool(row["orientation_mismatch"]))
    label_counter = Counter()
    file_hash_counter = Counter()
    total_slices = sum(int(row["total_slices"]) for row in audited_rows)
    positive_slices = sum(int(row["positive_slices"]) for row in audited_rows)
    tumor_voxel_counts = [int(row["tumor_voxel_count"]) for row in audited_rows]
    tumor_fractions = [float(row["tumor_fraction"]) for row in audited_rows]

    volume_shapes = Counter(
        tuple(row["mask_shape"]) for row in audited_rows if row["mask_shape"] is not None
    )
    voxel_spacings = Counter(
        tuple(round(float(value), 6) for value in row["mask_spacing"])
        for row in audited_rows
        if row["mask_spacing"] is not None
    )
    orientations = Counter(
        str(row["mask_orientation"])
        for row in audited_rows
        if row["mask_orientation"] is not None
    )

    for row in audited_rows:
        for value in row["mask_label_values"]:
            label_counter[int(value)] += 1
        modality_hashes = row["modality_file_hashes"]
        assert isinstance(modality_hashes, dict)
        for file_hash in modality_hashes.values():
            file_hash_counter[str(file_hash)] += 1
        if row["mask_file_hash"]:
            file_hash_counter[str(row["mask_file_hash"])] += 1

    def _numeric(values: list[float] | list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
        }

    return {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "license_access_status": license_access_status,
        "manual_download_required": manual_download_required,
        "subject_count": subject_count,
        "modalities": modalities,
        "modality_coverage": modality_coverage,
        "subjects_with_missing_modalities": missing_modalities,
        "subjects_with_missing_masks": missing_masks,
        "subjects_with_corrupt_or_unreadable_volumes": corrupt_volumes,
        "subjects_with_empty_masks": empty_masks,
        "duplicate_subject_ids": subject_count - len({str(row['subject_id']) for row in audited_rows}),
        "duplicate_file_hash_groups": sum(1 for count in file_hash_counter.values() if count > 1),
        "shape_mismatch_subjects": shape_mismatches,
        "spacing_mismatch_subjects": spacing_mismatches,
        "orientation_mismatch_subjects": orientation_mismatches,
        "mask_label_values": sorted(label_counter),
        "volume_shape_distribution": {str(key): value for key, value in sorted(volume_shapes.items())},
        "voxel_spacing_distribution": {str(key): value for key, value in sorted(voxel_spacings.items())},
        "mask_orientation_distribution": dict(sorted(orientations.items())),
        "tumor_voxel_count_summary": _numeric(tumor_voxel_counts),
        "tumor_fraction_summary": _numeric(tumor_fractions),
        "total_slices": total_slices,
        "tumor_positive_slices": positive_slices,
        "tumor_positive_slice_ratio": float(positive_slices / total_slices) if total_slices else 0.0,
    }


def _normalize_slice(array: np.ndarray) -> np.ndarray:
    slice_array = array.astype(np.float32)
    minimum = float(slice_array.min())
    maximum = float(slice_array.max())
    if maximum <= minimum:
        return np.zeros_like(slice_array, dtype=np.uint8)
    normalized = (slice_array - minimum) / (maximum - minimum)
    return np.uint8(np.clip(normalized, 0.0, 1.0) * 255.0)


def build_sample_overlay_grid(
    subjects: list[BraTSSegmentationSubject],
    audited_rows: list[dict[str, object]],
    *,
    modality: str,
    output_path: str | Path,
    max_subjects: int = 6,
) -> Path:
    """Create a compact MRI / mask / overlay sanity-check grid."""
    selected_subjects = []
    rows_by_subject = {str(row["subject_id"]): row for row in audited_rows}
    for subject in subjects:
        row = rows_by_subject.get(subject.subject_id)
        if row is None or row["mask_readable"] is not True:
            continue
        if modality not in subject.modality_paths:
            continue
        if int(row["positive_slices"]) <= 0:
            continue
        selected_subjects.append(subject)
        if len(selected_subjects) >= max_subjects:
            break

    tile_width = 160
    tile_height = 160
    canvas = Image.new("RGB", (tile_width * 3, tile_height * max(len(selected_subjects), 1)), color="white")

    for row_index, subject in enumerate(selected_subjects):
        row = rows_by_subject[subject.subject_id]
        image_array, _ = load_nifti_volume(subject.modality_paths[modality])
        mask_array, _ = load_nifti_volume(subject.mask_path)
        positive_by_slice = (mask_array > 0).reshape(-1, mask_array.shape[2]).any(axis=0)
        slice_index = int(np.argmax(positive_by_slice)) if positive_by_slice.any() else image_array.shape[2] // 2
        image_slice = _normalize_slice(image_array[:, :, slice_index])
        mask_slice = np.uint8((mask_array[:, :, slice_index] > 0) * 255)

        base_rgb = np.stack([image_slice, image_slice, image_slice], axis=2)
        overlay_rgb = base_rgb.copy()
        overlay_rgb[:, :, 0] = np.maximum(overlay_rgb[:, :, 0], mask_slice)
        overlay_rgb[:, :, 1] = np.where(mask_slice > 0, overlay_rgb[:, :, 1] // 2, overlay_rgb[:, :, 1])
        overlay_rgb[:, :, 2] = np.where(mask_slice > 0, overlay_rgb[:, :, 2] // 2, overlay_rgb[:, :, 2])

        tiles = [
            Image.fromarray(base_rgb).resize((tile_width, tile_height)),
            Image.fromarray(np.stack([mask_slice, mask_slice, mask_slice], axis=2)).resize((tile_width, tile_height)),
            Image.fromarray(overlay_rgb).resize((tile_width, tile_height)),
        ]
        for column_index, tile in enumerate(tiles):
            canvas.paste(tile, (column_index * tile_width, row_index * tile_height))

    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_path)
    return resolved_path


def write_segmentation_json(output_path: str | Path, payload: dict[str, Any]) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved_path


def write_mask_statistics_csv(output_path: str | Path, audited_rows: list[dict[str, object]]) -> Path:
    resolved_path = resolve_project_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "subject_id": row["subject_id"],
            "tumor_voxel_count": row["tumor_voxel_count"],
            "tumor_fraction": row["tumor_fraction"],
            "num_positive_slices": row["positive_slices"],
            "total_slices": row["total_slices"],
            "volume_shape": json.dumps(row["mask_shape"]),
            "spacing": json.dumps(row["mask_spacing"]),
        }
        for row in audited_rows
    ]
    if not rows:
        resolved_path.write_text("", encoding="utf-8")
        return resolved_path
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved_path


def audited_rows_to_serializable(audited_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return JSON-safe copies of the audited rows."""
    return [json.loads(json.dumps(row)) for row in audited_rows]
