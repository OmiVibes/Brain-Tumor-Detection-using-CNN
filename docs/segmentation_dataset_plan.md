# Segmentation Dataset Plan

## Chosen Task

Initial segmentation scope: **adult glioma MRI tumor segmentation**

This is intentionally narrower than the existing classifier.

Current BrainScanAI classifier supports:

- `glioma`
- `meningioma`
- `pituitary`
- `no_tumor`

Phase 3A segmentation support is planned only for:

- **adult glioma segmentation**

We will not claim meningioma or pituitary localization until corresponding mask-labeled datasets are actually integrated.

## Chosen Dataset

- dataset: **BraTS 2023 Adult Glioma**
- authoritative source: official **BraTS 2023 Challenge** distribution on **Synapse**
- challenge family: BraTS adult glioma benchmark used continuously across BraTS 2021, 2022, and 2023

## Why This Dataset

BraTS is the preferred first segmentation foundation because it provides:

- real MRI volumes
- real expert-derived segmentation masks
- benchmark-standard adult glioma task definitions
- established evaluation conventions
- multimodal MRI structure with subject-level organization

## Authoritative Source

Primary access point:

- Synapse BraTS 2023 Challenge page: `syn51156910`

Relevant source facts verified from the official pages:

- BraTS datasets are distributed through Synapse
- access requires a registered Synapse user account
- users must accept the post-challenge terms and conditions
- the data may be used for non-commercial research under those access conditions

Notes:

- BraTS 2021 is archived and now points users to BraTS 2023 Adult Glioma
- this plan therefore uses BraTS 2023 Adult Glioma as the concrete acquisition target

## Access Method

Expected acquisition path:

1. Create or use a registered Synapse account
2. Open the official BraTS 2023 challenge workspace
3. Accept the required dataset terms / conditions
4. Download the authorized adult glioma training data manually
5. Place the raw dataset under:
   `data/segmentation/raw/brats2023_glioma/`

Phase 3A will not bypass login, terms, or access controls.

## License / Access Conditions

Current status:

- access-controlled research dataset
- manual authorization required
- raw data must not be committed to Git

Because the dataset is distributed behind Synapse registration and terms acceptance, BrainScanAI should treat the dataset as **manually acquired protected research data**, not as a redistributable repo asset.

## MRI Modalities

BraTS Adult Glioma subject folders are documented with these modality names:

- `t1c`
- `t1n`
- `t2f`
- `t2w`

These are the source naming conventions expected by the Phase 3A config and verification scripts.

## Ground-Truth Labels

BraTS Adult Glioma includes real segmentation labels.

Documented label semantics from the official BraTS Adult Glioma description:

- background: `0`
- NCR / NET: `1`
- edema: `2`
- enhancing tumor: `3`

Phase 3A baseline mode:

- **binary whole-tumor segmentation**

That means later training can use:

- binary mask = `mask > 0`

while still preserving the original multi-region label information for future extensions.

## Subject Structure

Expected BraTS-style layout per subject:

```text
BraTS-GLI-xxxxx-xxx/
  BraTS-GLI-xxxxx-xxx-t1c.nii.gz
  BraTS-GLI-xxxxx-xxx-t1n.nii.gz
  BraTS-GLI-xxxxx-xxx-t2f.nii.gz
  BraTS-GLI-xxxxx-xxx-t2w.nii.gz
  BraTS-GLI-xxxxx-xxx-seg.nii.gz
```

Phase 3A assumes subject-level directories and will split only by subject.

## Expected Download Size

The official crawled source pages did not expose a precise archive size directly.

Planning estimate:

- treat the raw BraTS dataset as a **multi-gigabyte download**
- reserve at least **20-30 GB** of free local disk for:
  - raw archives
  - extracted NIfTI volumes
  - audit artifacts

This size estimate is a planning inference, not an official published number from the source page.

## Planned Subset Strategy

Phase 3A does not train yet, but the intended staged development path is:

1. Keep the authoritative raw NIfTI volumes untouched
2. Audit all authorized subjects and masks
3. Create deterministic subject-level manifests
4. Start with a **2D whole-tumor baseline** in Phase 3B
5. Later consider:
   - positive-slice-focused sampling
   - 2.5D slice stacks
   - patch-based 3D extensions

## Train / Val / Test Methodology

Required rule:

- no subject may appear in more than one split

Planned deterministic subject-level split options:

- `70 / 15 / 15` when subject count is large enough
- `80 / 10 / 10` for smaller cohorts

Phase 3A code uses fixed seed `42` and writes:

- `data/segmentation/splits/train.csv`
- `data/segmentation/splits/val.csv`
- `data/segmentation/splits/test.csv`

## Hardware Constraints

Current GPU:

- NVIDIA GeForce GTX 1650
- about `4 GB` VRAM

This makes a first-pass full-resolution 3D segmentation experiment risky.

Planned progression:

1. Phase 3B: 2D baseline
2. Later: 2.5D or patch-based segmentation
3. Only after resource validation: optional 3D work

## Known Limitations

- the segmentation dataset is not currently present in this repository
- manual dataset download is required
- Phase 3A must stop at the acquisition boundary until authorized data is placed locally
- classification and segmentation datasets will remain separate supervised resources
- BrainScanAI segmentation will initially support glioma only, not all classifier tumor classes
