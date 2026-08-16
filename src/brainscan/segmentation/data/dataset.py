"""Cached multimodal BraTS slice dataset for 2D segmentation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from brainscan.core.config import get_project_root, resolve_project_path
from brainscan.segmentation.data.audit import load_nifti_volume
from brainscan.segmentation.data.preprocessing import (
    SegmentationAugmenter,
    build_binary_mask,
    stack_modalities,
    zscore_normalize_nonzero,
)
from brainscan.segmentation.data.slice_index import SliceIndexRecord, load_slice_index_records


@dataclass(frozen=True)
class CachedSubjectVolumes:
    subject_id: str
    modalities: np.ndarray
    binary_mask: np.ndarray


class SubjectVolumeLRUCache:
    """A tiny LRU cache to avoid repeated `.nii.gz` decode for adjacent slices."""

    def __init__(self, max_size: int = 1) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = int(max_size)
        self._entries: OrderedDict[str, CachedSubjectVolumes] = OrderedDict()

    def get(self, key: str) -> CachedSubjectVolumes | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key: str, value: CachedSubjectVolumes) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)


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
    ) -> None:
        if index_path is None and records is None:
            raise ValueError("Either index_path or records must be provided.")
        self.modalities = list(modalities)
        self.records = records or load_slice_index_records(index_path, modalities=self.modalities)
        self.augmenter = augmenter
        self.return_metadata = return_metadata
        self.cache = SubjectVolumeLRUCache(max_size=max_subject_cache_size)
        self.dataset_root = resolve_project_path(dataset_root) if dataset_root is not None else get_project_root()

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_data_path(self, path_value: str) -> str:
        candidate = Path(path_value)
        if candidate.is_absolute():
            raise ValueError(f"Slice dataset path must be relative, got absolute path: {path_value}")
        return str((self.dataset_root / candidate).resolve())

    def _load_subject(self, record: SliceIndexRecord) -> CachedSubjectVolumes:
        cached = self.cache.get(record.subject_id)
        if cached is not None:
            return cached

        modality_volumes: list[np.ndarray] = []
        for modality in self.modalities:
            modality_path = record.modality_paths[modality]
            if not modality_path:
                raise ValueError(f"Missing modality '{modality}' for subject '{record.subject_id}'.")
            volume, _ = load_nifti_volume(self._resolve_data_path(modality_path))
            modality_volumes.append(zscore_normalize_nonzero(volume))

        mask_volume, _ = load_nifti_volume(self._resolve_data_path(record.mask_path))
        cached = CachedSubjectVolumes(
            subject_id=record.subject_id,
            modalities=stack_modalities(modality_volumes),
            binary_mask=build_binary_mask(mask_volume),
        )
        self.cache.put(record.subject_id, cached)
        return cached

    def __getitem__(self, index: int):
        record = self.records[index]
        subject = self._load_subject(record)
        slice_index = int(record.slice_index)
        image = torch.from_numpy(subject.modalities[:, :, :, slice_index].copy()).to(dtype=torch.float32)
        mask = torch.from_numpy(subject.binary_mask[:, :, slice_index].copy()).unsqueeze(0).to(dtype=torch.float32)

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
        }
        return image, mask, metadata
