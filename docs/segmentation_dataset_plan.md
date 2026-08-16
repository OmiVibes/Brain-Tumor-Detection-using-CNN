# Segmentation Dataset Plan

## Chosen Task

Initial segmentation scope: **adult glioma MRI tumor segmentation**

This remains intentionally narrower than the existing classifier.

Current BrainScanAI classifier supports:

- `glioma`
- `meningioma`
- `pituitary`
- `no_tumor`

Phase 3A segmentation support is established only for:

- **adult glioma segmentation**

We do not claim meningioma or pituitary localization, because no corresponding segmentation dataset has been integrated.

## Dataset Selection

- dataset: **BraTS 2023 Adult Glioma training package**
- authoritative source: official **BraTS 2023 Challenge** distribution on **Synapse**
- local extracted package identity:
  `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`

The classification dataset and segmentation dataset are independent supervised resources. They are not forced to share subject IDs, labels, or splits.

## Why This Dataset

BraTS remains the correct first segmentation foundation because it provides:

- real multimodal MRI volumes
- real voxel-level tumor masks
- subject-level organization
- benchmark-standard adult glioma segmentation conventions
- strong scientific and interview credibility

## Authoritative Source

Primary source family:

- official BraTS 2023 Synapse workspace: `syn51156910`

Phase 3A was built around the official Synapse distribution rather than a repackaged mirror.

## Access Method And Status

Acquisition method:

1. Use a registered Synapse account
2. Accept the dataset terms
3. Download the authorized package manually
4. Extract under:
   `data/segmentation/raw/brats2023_glioma/`

Current status:

- manual download required: **yes**
- raw medical volumes committed to Git: **no**
- exact challenge-side revision metadata exposed locally: **not available**

The local extracted package proves the dataset was obtained, but it does not itself expose a richer revision identifier beyond the package directory name above.

## Actual Extracted Structure

The real local layout contains one outer container directory under the configured root:

```text
data/segmentation/raw/brats2023_glioma/
  ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
    BraTS-GLI-00000-000/
      BraTS-GLI-00000-000-t1c.nii.gz
      BraTS-GLI-00000-000-t1n.nii.gz
      BraTS-GLI-00000-000-t2f.nii.gz
      BraTS-GLI-00000-000-t2w.nii.gz
      BraTS-GLI-00000-000-seg.nii.gz
```

Phase 3A now resolves this extracted-package parent directory automatically during discovery and audit.

## Actual Modalities And Mask Naming

Confirmed from the real files:

- `t1c`
- `t1n`
- `t2f`
- `t2w`
- `seg`

Source data remains in NIfTI format:

- images: `.nii.gz`
- mask: `.nii.gz`

## Ground-Truth Labels

Observed in the real masks:

- `0`
- `1`
- `2`
- `3`

BraTS label semantics used for documentation and future processing:

- background: `0`
- NCR / NET: `1`
- edema: `2`
- enhancing tumor: `3`

For the planned Phase 3B baseline, binary whole-tumor masks should be derived at runtime as:

- background: `mask == 0`
- tumor: `mask > 0`

Original multi-region masks must remain untouched.

## Split Methodology

Required rule:

- no subject may appear in more than one split

Because the cohort is large, the active strategy is:

- `70 / 15 / 15`
- deterministic subject-level split
- fixed seed `42`

Manifest outputs:

- `data/segmentation/splits/train.csv`
- `data/segmentation/splits/val.csv`
- `data/segmentation/splits/test.csv`

All manifest paths are repository-relative and portable.

## Hardware Constraints

Current GPU:

- NVIDIA GeForce GTX 1650
- about `4 GB` VRAM

This still supports the same staged plan:

1. Phase 3B: 2D baseline
2. Later: 2.5D or patch-based segmentation
3. Optional 3D work only after resource validation

We are not starting full-resolution 3D segmentation first.

## Known Limitations

- the raw BraTS dataset is protected, manually acquired data and cannot be pushed to GitHub
- the local extracted package does not expose richer revision metadata than the package directory identity
- BrainScanAI segmentation currently covers glioma only
- classification and segmentation remain separate dataset pipelines by design
