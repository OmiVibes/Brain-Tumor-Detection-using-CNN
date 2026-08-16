# Segmentation Dataset Audit

## Dataset Identity

- dataset: `brats_adult_glioma`
- configured version: `2023`
- local package directory:
  `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
- authoritative source family: official BraTS Synapse distribution
- access status: manual Synapse access with accepted terms
- manual download required: **yes**

The exact challenge-side revision metadata was not exposed in the local extracted files, so this audit records the actual package directory identity rather than guessing a stronger revision claim.

## Separation From Classification Data

BrainScanAI now uses two independent supervised datasets:

- classification dataset for tumor-class prediction
- BraTS segmentation dataset for voxel-level glioma masks

These splits and fingerprints are intentionally independent.

## Actual Directory Structure

Observed local structure:

```text
data/segmentation/raw/brats2023_glioma/
  ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
    BraTS-GLI-xxxxx-xxx/
      BraTS-GLI-xxxxx-xxx-t1c.nii.gz
      BraTS-GLI-xxxxx-xxx-t1n.nii.gz
      BraTS-GLI-xxxxx-xxx-t2f.nii.gz
      BraTS-GLI-xxxxx-xxx-t2w.nii.gz
      BraTS-GLI-xxxxx-xxx-seg.nii.gz
```

## Cohort Summary

- total subjects: `1251`
- total MRI volumes: `5004`
- total segmentation masks: `1251`
- missing modalities: `0`
- missing masks: `0`
- corrupt or unreadable volumes: `0`
- empty-mask subjects: `0`
- duplicate subject IDs: `0`
- duplicate file-hash groups: `0`

## Modalities And Masks

Confirmed modalities in every subject:

- `t1c`
- `t1n`
- `t2f`
- `t2w`

Confirmed segmentation mask name:

- `seg`

Mask labels observed in the real dataset:

- `0`
- `1`
- `2`
- `3`

Documented interpretation used by BrainScanAI:

- `0`: background
- `1`: NCR / NET
- `2`: edema
- `3`: enhancing tumor

## Geometry And Alignment

- volume shape distribution:
  - `(240, 240, 155)`: `1251`
- voxel spacing distribution:
  - `(1.0, 1.0, 1.0)`: `1251`
- mask orientation distribution:
  - `LPS`: `1251`
- image/mask shape mismatches: `0`
- affine/spacing mismatches: `0`
- orientation mismatches: `0`

The audited BraTS package is geometrically consistent and did not require resizing, registration, or relabeling.

## Tumor Statistics

- tumor voxel count summary:
  - min: `2808`
  - median: `89335`
  - mean: `95967.62`
  - max: `361783`
- tumor fraction summary:
  - min: `0.0003145161`
  - median: `0.0100061604`
  - mean: `0.0107490617`
  - max: `0.0405222894`

## 2D Axial Slice Statistics

- total axial slices: `193905`
- tumor-positive slices: `81437`
- tumor-negative slices: `112468`
- tumor-positive slice ratio: `0.419984`
- tumor-negative slice ratio: `0.580016`
- positive slices per subject:
  - min: `11`
  - median: `66`
  - mean: `65.10`
  - max: `109`

These numbers support a 2D baseline while still showing meaningful class imbalance at slice level.

## Subject-Level Split

Strategy used:

- deterministic subject-level `70 / 15 / 15`
- seed `42`

Split counts:

- train subjects: `875`
- validation subjects: `188`
- test subjects: `188`

Leakage verification:

- `train ∩ val = 0`
- `train ∩ test = 0`
- `val ∩ test = 0`
- `train ∪ val ∪ test = 1251`

No subject leakage was detected.

## Dataset Fingerprint

- segmentation dataset fingerprint:
  `5bc6c1fe8868265cc8df6042ba353891aff6707f9b1acb65c4816c03920314db`

Fingerprint source:

- subject IDs
- relative modality paths
- relative mask paths
- file hashes
- observed label values
- dataset version

## Visual Sanity Check

Artifact:

- `artifacts/segmentation/data_audit/sample_overlays.png`

Visual review result:

- masks overlap plausible intracranial tumor regions
- no obvious left-right flip was observed
- no gross spatial shift was observed
- overlays appear aligned with the MRI slices selected for inspection

## Generated Artifacts

- `data/segmentation/splits/train.csv`
- `data/segmentation/splits/val.csv`
- `data/segmentation/splits/test.csv`
- `artifacts/segmentation/data_audit/summary.json`
- `artifacts/segmentation/data_audit/dataset_fingerprint.json`
- `artifacts/segmentation/data_audit/mask_statistics.csv`
- `artifacts/segmentation/data_audit/sample_overlays.png`

## Known Limitations

- raw medical volumes are intentionally excluded from Git
- the local package does not expose richer challenge-revision metadata
- this audit validates the dataset foundation only
- no segmentation model training was started in Phase 3A
