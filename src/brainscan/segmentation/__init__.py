"""Segmentation data and training foundation for BrainScanAI."""

from brainscan.segmentation.data import (
    BraTSSegmentationSubject,
    BraTSSliceDataset,
    SegmentationManifestRow,
    SliceIndexRecord,
    audit_brats_subjects,
    build_segmentation_dataset_fingerprint,
    choose_subject_split_strategy,
    discover_brats_subjects,
    load_slice_index_records,
    write_slice_index_csv,
    write_segmentation_manifest_csv,
)
from brainscan.segmentation.models import UNet2D
from brainscan.segmentation.training import BCEDiceLoss

__all__ = [
    "BraTSSegmentationSubject",
    "BraTSSliceDataset",
    "BCEDiceLoss",
    "SegmentationManifestRow",
    "SliceIndexRecord",
    "UNet2D",
    "audit_brats_subjects",
    "build_segmentation_dataset_fingerprint",
    "choose_subject_split_strategy",
    "discover_brats_subjects",
    "load_slice_index_records",
    "write_slice_index_csv",
    "write_segmentation_manifest_csv",
]
