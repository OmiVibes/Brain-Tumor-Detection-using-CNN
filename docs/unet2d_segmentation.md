# U-Net Whole-Tumor Segmentation Baseline

## Scope

Phase 3B adds the first BrainScanAI tumor-segmentation baseline:

- model: `2D U-Net`
- task: binary whole-tumor segmentation
- input: 4-channel axial slice `t1c + t1n + t2f + t2w`
- target: runtime binary mask where `seg > 0`

This baseline is independent from the earlier classification stack.

Important distinction:

- Grad-CAM is classifier attention over an image
- U-Net segmentation is a pixel-level tumor mask prediction

They are not equivalent and should not be presented as interchangeable outputs.

## Dataset And Split

- dataset: BraTS 2023 Adult Glioma
- audited dataset fingerprint: `5bc6c1fe8868265cc8df6042ba353891aff6707f9b1acb65c4816c03920314db`
- subject split:
  - train: `875`
  - val: `188`
  - test: `188`

BraTS source masks remain unchanged on disk. The source labels are still:

- `0`: background
- `1`: NCR / NET
- `2`: edema
- `3`: enhancing tumor

The training/evaluation target is derived at runtime as:

```text
whole_tumor = seg > 0
```

## Slice Index And Sampling

The segmentation pipeline reads NIfTI volumes directly and uses derived slice manifests instead of exporting images.

Slice-manifest counts:

- train slices: `135625`
- val slices: `29140`
- test slices: `29140`

Positive tumor slices:

- train positives available: `56987`
- val positives: `12168`
- test positives: `12282`

Training sampler policy:

- include all train positive slices
- include a deterministic sampled negative subset
- configured ratio: `1.0` positive : negative

Effective sampled train slices used during training:

- positives used: `56987`
- negatives used: `56987`
- total selected train slices: `113974`

Validation and test keep the full natural slice distribution.

## Preprocessing

- native slice size: `240 x 240`
- no RGB conversion
- no ImageNet normalization
- normalization: per-modality z-score on non-zero voxels only
- zero-valued background preserved
- zero-standard-deviation case handled safely

Training augmentation is conservative and applied identically to all 4 modalities and the mask:

- horizontal flip probability: `0.5`
- max rotation: `5` degrees
- max translation: `4` pixels

Mask interpolation remains nearest-neighbor.

## Model

Architecture:

- `UNet2D`
- input channels: `4`
- output channels: `1`
- base channels: `16`
- output: logits with shape `[B, 1, H, W]`

Parameter count:

- `1,942,721`

The baseline uses base channels `16` rather than `32` because the target hardware is a GTX 1650 with about 4 GB VRAM.

## Training Setup

- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`
- loss: `0.5 * BCEWithLogitsLoss + 0.5 * soft Dice loss`
- threshold for metrics/inference: `0.5`
- max epochs configured: `30`
- early stopping patience: `6`
- batch size: `32`
- AMP: disabled after CUDA instability on this GPU
- max subject cache size: `1`
- seed: `42`

Verified environment:

- Python virtual environment: project-local `.venv`
- device used: CUDA
- GPU: NVIDIA GeForce GTX 1650
- peak VRAM during full training: `3178372096` bytes

## Validation Selection

Checkpoint selection uses validation subject-level whole-tumor Dice only. Test subjects were not used for:

- architecture choice
- threshold tuning
- checkpoint selection
- training decisions

Frozen selection artifact:

- `artifacts/segmentation/model_selection.json`

Frozen best validation checkpoint:

- checkpoint: `artifacts/models/segmentation/unet2d_whole_tumor/best.pt`
- best epoch: `9`
- best validation subject Dice: `0.884722074469403`
- best validation IoU: `0.8037340541902028`
- validation slice Dice at best epoch: `0.5072312764368236`

Training did not stop because early stopping triggered. The full baseline run terminated after epoch `9` because repeated execution/runtime limits interrupted the longer run, and epoch `9` was then frozen as the best validation checkpoint among the completed epochs.

Training history artifacts:

- `artifacts/segmentation/training/unet2d_whole_tumor/history.json`
- `artifacts/segmentation/training/unet2d_whole_tumor/history.csv`
- `artifacts/segmentation/training/unet2d_whole_tumor/training_state.json`

## Formal Held-Out Test Evaluation

The frozen epoch-9 checkpoint was then evaluated once on all `188` test subjects.

Subject-level whole-volume test metrics:

- mean Dice: `0.876161137303281`
- median Dice: `0.917591166708129`
- Dice std: `0.1081122043939062`
- mean IoU: `0.7933622793794609`
- mean precision: `0.856860813650186`
- mean recall: `0.9120344830452767`
- mean specificity: `0.9988519307862386`

Dice distribution:

- min: `0.32631238597512285`
- q25: `0.8673601265539533`
- median: `0.917591166708129`
- q75: `0.9403784057123776`
- max: `0.976069775174627`

Evaluation artifacts:

- `artifacts/segmentation/evaluation/unet2d_whole_tumor/summary.json`
- `artifacts/segmentation/evaluation/unet2d_whole_tumor/failure_analysis.json`
- `artifacts/segmentation/evaluation/unet2d_whole_tumor/subject_metrics.csv`
- `artifacts/segmentation/evaluation/unet2d_whole_tumor/review_grid.png`

## Example Outcomes

Representative review cases are exported in:

- `artifacts/segmentation/evaluation/unet2d_whole_tumor/review_grid.png`

The grid contains:

- good case
- median case
- poor case

For each case the grid shows:

- `t2f` MRI slice
- ground-truth binary mask
- predicted binary mask
- overlay

## Failure Analysis

Best subject:

- subject: `BraTS-GLI-01537-000`
- Dice: `0.976069775174627`

Median subject:

- subject: `BraTS-GLI-01046-000`
- Dice: `0.9179500004813442`

Worst subject:

- subject: `BraTS-GLI-01530-000`
- Dice: `0.32631238597512285`

Observed failure pattern in the worst case:

- recall remained relatively high: `0.8271279709737075`
- precision dropped sharply: `0.20324817309391727`
- predicted tumor voxels: `69516`
- true tumor voxels: `17082`
- false positives: `55387`

This indicates the weakest case is dominated by over-segmentation rather than full tumor miss.

## Verification

Phase 3B verification includes:

- unit tests for dataset, preprocessing, U-Net, losses, training, and evaluation
- compile check for `src` and `scripts`
- smoke training on CUDA before full training
- frozen validation-only checkpoint selection before any test evaluation

## Limitations

- this is a first 2D baseline, not a 3D segmentation system
- training stopped under repeated execution-time limits rather than classic clean early stopping
- AMP was unstable on the GTX 1650 for this workload and was disabled
- the worst cases still show substantial false-positive over-segmentation
- this phase handles only binary whole-tumor masks, not BraTS sub-region segmentation
- this remains a research prototype and is not clinically validated
