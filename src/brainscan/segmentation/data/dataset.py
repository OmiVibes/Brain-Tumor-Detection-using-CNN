"""Lazy multimodal BraTS slice dataset for 2D and 2.5D segmentation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from brainscan.core.config import get_project_root, resolve_project_path
from brainscan.segmentation.data.preprocessing import (
    SegmentationAugmenter,
    VolumeNormalizationStats,
    accumulate_nonzero_stats,
    finalize_nonzero_stats,
    normalize_nonzero_slice,
)
from brainscan.segmentation.data.slice_index import SliceIndexRecord, load_slice_index_records


@dataclass(frozen=True)
class CachedSubjectHandles:
    subject_id: str
    modality_images: dict[str, nib.Nifti1Image]
    mask_image: nib.Nifti1Image
    depth: int
    normalization_stats: dict[str, VolumeNormalizationStats]


class SubjectVolumeLRUCache:
    """A tiny LRU cache for lightweight nibabel proxies and stats."""

    def __init__(self, max_size: int = 0) -> None:
        if max_size < 0:
            raise ValueError("max_size must be non-negative.")
        self.max_size = int(max_size)
        self._entries: OrderedDict[str, CachedSubjectHandles] = OrderedDict()

    def get(self, key: str) -> CachedSubjectHandles | None:
        if self.max_size == 0:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key: str, value: CachedSubjectHandles) -> None:
        if self.max_size == 0:
            return
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)


def load_nifti_image(path_value: str | Path) -> nib.Nifti1Image:
    """Open a NIfTI file and keep proxy-backed access for lazy slicing."""
    resolved_path = resolve_project_path(path_value)
    return nib.load(str(resolved_path))


def read_axial_slice(image: nib.Nifti1Image, slice_index: int, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Read one axial slice lazily from a proxy-backed NIfTI image."""
    dataobj = image.dataobj
    return np.asarray(dataobj[:, :, int(slice_index)], dtype=dtype)


def compute_subject_modality_stats(image: nib.Nifti1Image) -> VolumeNormalizationStats:
    """Compute non-zero normalization stats lazily by scanning one slice at a time."""
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI image, got shape {tuple(image.shape)}")
    count = 0
    total = 0.0
    total_squares = 0.0
    for slice_index in range(int(image.shape[2])):
        slice_array = read_axial_slice(image, slice_index, dtype=np.float32)
        count, total, total_squares = accumulate_nonzero_stats(
            count=count,
            total=total,
            total_squares=total_squares,
            values=slice_array,
        )
    return finalize_nonzero_stats(count=count, total=total, total_squares=total_squares)


def load_normalization_stats_map(
    path_value: str | Path,
) -> dict[str, dict[str, VolumeNormalizationStats]]:
    """Load per-subject, per-modality normalization stats from a JSON artifact."""
    resolved_path = resolve_project_path(path_value)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    subjects_payload = payload.get("subjects", payload)
    if not isinstance(subjects_payload, dict):
        raise TypeError("Normalization stats artifact must contain a 'subjects' mapping.")

    stats_map: dict[str, dict[str, VolumeNormalizationStats]] = {}
    for subject_id, modalities_payload in subjects_payload.items():
        if not isinstance(modalities_payload, dict):
            raise TypeError(f"Normalization stats for subject '{subject_id}' must be a mapping.")
        per_subject: dict[str, VolumeNormalizationStats] = {}
        for modality, raw_stats in modalities_payload.items():
            if not isinstance(raw_stats, dict):
                raise TypeError(f"Normalization stats for '{subject_id}/{modality}' must be a mapping.")
            per_subject[modality] = VolumeNormalizationStats(
                mean_nonzero=float(raw_stats["mean_nonzero"]),
                std_nonzero=float(raw_stats["std_nonzero"]),
                nonzero_count=int(raw_stats["nonzero_count"]),
            )
        stats_map[str(subject_id)] = per_subject
    return stats_map


def resolve_neighbor_indices(
    *,
    center_index: int,
    depth: int,
    slice_offsets: list[int],
    neighbor_policy: str,
) -> list[int]:
    """Resolve slice offsets into in-volume indices without crossing subjects."""
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if not slice_offsets:
        raise ValueError("slice_offsets must contain at least one offset.")

    resolved_indices: list[int] = []
    for offset in slice_offsets:
        candidate = int(center_index) + int(offset)
        if neighbor_policy == "edge_replicate":
            candidate = min(max(candidate, 0), depth - 1)
        else:
            raise ValueError(f"Unsupported neighbor_policy '{neighbor_policy}'.")
        resolved_indices.append(candidate)
    return resolved_indices


def extract_slice_context(
    modality_images: dict[str, nib.Nifti1Image],
    *,
    modalities: list[str],
    normalization_stats: dict[str, VolumeNormalizationStats],
    center_index: int,
    depth: int,
    slice_offsets: list[int],
    neighbor_policy: str,
) -> np.ndarray:
    """Return [C,H,W] context with modality-major, offset-minor channel ordering."""
    neighbor_indices = resolve_neighbor_indices(
        center_index=int(center_index),
        depth=depth,
        slice_offsets=slice_offsets,
        neighbor_policy=neighbor_policy,
    )
    channels: list[np.ndarray] = []
    for modality in modalities:
        image = modality_images[modality]
        stats = normalization_stats[modality]
        for slice_index in neighbor_indices:
            channels.append(normalize_nonzero_slice(read_axial_slice(image, slice_index, dtype=np.float32), stats))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def select_training_slice_records(
    records: list[SliceIndexRecord],
    *,
    positive_negative_ratio: float,
    seed: int,
) -> list[SliceIndexRecord]:
    """Include all positives plus a deterministic negative subset for training."""
    if positive_negative_ratio <= 0:
        raise ValueError("positive_negative_ratio must be positive.")
    positives = [record for record in records if record.contains_tumor]
    negatives = [record for record in records if not record.contains_tumor]
    if not positives:
        raise ValueError("Training slice index contains no positive slices.")

    target_negative_count = min(len(negatives), int(round(len(positives) * positive_negative_ratio)))
    generator = np.random.default_rng(seed)
    chosen_negative_indices = sorted(generator.choice(len(negatives), size=target_negative_count, replace=False).tolist())
    selected_negatives = [negatives[index] for index in chosen_negative_indices]
    selected = positives + selected_negatives

    subject_ids = sorted({record.subject_id for record in selected})
    shuffled_subject_ids = subject_ids[:]
    generator.shuffle(shuffled_subject_ids)
    order_lookup = {subject_id: index for index, subject_id in enumerate(shuffled_subject_ids)}
    return sorted(selected, key=lambda record: (order_lookup[record.subject_id], record.slice_index))


class BraTSSliceDataset(Dataset[tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Any]]]):
    """Read multimodal axial slices directly from NIfTI volumes."""

    def __init__(
        self,
        *,
        index_path: str | Path | None = None,
        records: list[SliceIndexRecord] | None = None,
        modalities: list[str],
        augmenter: SegmentationAugmenter | None = None,
        return_metadata: bool = False,
        max_subject_cache_size: int = 1,
        dataset_root: str | Path | None = None,
        slice_offsets: list[int] | None = None,
        neighbor_policy: str = "edge_replicate",
        normalization_stats_path: str | Path | None = None,
        normalization_stats: dict[str, dict[str, VolumeNormalizationStats]] | None = None,
    ) -> None:
        if index_path is None and records is None:
            raise ValueError("Either index_path or records must be provided.")
        self.modalities = list(modalities)
        self.records = records or load_slice_index_records(index_path, modalities=self.modalities)
        self.augmenter = augmenter
        self.return_metadata = return_metadata
        self.cache = SubjectVolumeLRUCache(max_size=max_subject_cache_size)
        self.dataset_root = resolve_project_path(dataset_root) if dataset_root is not None else get_project_root()
        self.slice_offsets = list(slice_offsets) if slice_offsets is not None else [0]
        self.neighbor_policy = str(neighbor_policy)
        self.normalization_stats = (
            normalization_stats
            if normalization_stats is not None
            else load_normalization_stats_map(normalization_stats_path)
            if normalization_stats_path is not None
            else {}
        )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_data_path(self, path_value: str) -> str:
        candidate = Path(path_value)
        if candidate.is_absolute():
            raise ValueError(f"Slice dataset path must be relative, got absolute path: {path_value}")
        return str((self.dataset_root / candidate).resolve())

    def _subject_stats(
        self,
        *,
        subject_id: str,
        modality_images: dict[str, nib.Nifti1Image],
    ) -> dict[str, VolumeNormalizationStats]:
        cached_stats = self.normalization_stats.get(subject_id)
        if cached_stats is not None:
            return cached_stats
        computed = {
            modality: compute_subject_modality_stats(modality_images[modality])
            for modality in self.modalities
        }
        self.normalization_stats[subject_id] = computed
        return computed

    def _load_subject(self, record: SliceIndexRecord) -> CachedSubjectHandles:
        cached = self.cache.get(record.subject_id)
        if cached is not None:
            return cached

        modality_images: dict[str, nib.Nifti1Image] = {}
        for modality in self.modalities:
            modality_path = record.modality_paths[modality]
            if not modality_path:
                raise ValueError(f"Missing modality '{modality}' for subject '{record.subject_id}'.")
            modality_images[modality] = load_nifti_image(self._resolve_data_path(modality_path))

        mask_image = load_nifti_image(self._resolve_data_path(record.mask_path))
        first_image = modality_images[self.modalities[0]]
        cached = CachedSubjectHandles(
            subject_id=record.subject_id,
            modality_images=modality_images,
            mask_image=mask_image,
            depth=int(first_image.shape[2]),
            normalization_stats=self._subject_stats(subject_id=record.subject_id, modality_images=modality_images),
        )
        self.cache.put(record.subject_id, cached)
        return cached

    def __getitem__(self, index: int):
        record = self.records[index]
        subject = self._load_subject(record)
        slice_index = int(record.slice_index)
        image_array = extract_slice_context(
            subject.modality_images,
            modalities=self.modalities,
            normalization_stats=subject.normalization_stats,
            center_index=slice_index,
            depth=subject.depth,
            slice_offsets=self.slice_offsets,
            neighbor_policy=self.neighbor_policy,
        )
        image = torch.from_numpy(np.ascontiguousarray(image_array)).to(dtype=torch.float32)
        mask_slice = (read_axial_slice(subject.mask_image, slice_index, dtype=np.uint8) > 0).astype(
            np.float32,
            copy=False,
        )
        mask = torch.from_numpy(np.ascontiguousarray(mask_slice)).unsqueeze(0).to(dtype=torch.float32)

        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        if not self.return_metadata:
            return image, mask

        metadata = {
            "subject_id": record.subject_id,
            "split": record.split,
            "slice_index": slice_index,
            "contains_tumor": record.contains_tumor,
            "tumor_pixel_count": int(record.tumor_pixel_count),
            "mask_path": record.mask_path,
            "modality_paths": dict(record.modality_paths),
            "slice_offsets": list(self.slice_offsets),
            "neighbor_policy": self.neighbor_policy,
        }
        return image, mask, metadata
