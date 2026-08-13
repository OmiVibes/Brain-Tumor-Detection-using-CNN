# ResNet18 Baseline Evaluation

## Scope

This document summarizes the first formal held-out evaluation for the frozen BrainScanAI Phase 1D ResNet18 baseline.

- model: `resnet18`
- checkpoint: `artifacts/models/resnet18_baseline_best.pt`
- checkpoint epoch: `9`
- training selection rule: best validation macro F1
- dataset fingerprint: `b15363ff85ecf29d11c67a2c38fad723a7bb339cb4489e53dd637ab16f8c24b1`
- formal test size: `702`
- evaluation device: `cuda` on `NVIDIA GeForce GTX 1650`

The model was **not retrained** for this phase. The held-out test set was evaluated exactly once for the formal baseline result and was not used for checkpoint selection or tuning.

## Overall Results

Held-out test metrics:

- accuracy: `0.9815`
- balanced accuracy: `0.9801`
- macro precision: `0.9810`
- macro recall: `0.9801`
- macro F1: `0.9805`
- weighted precision: `0.9814`
- weighted recall: `0.9815`
- weighted F1: `0.9814`
- multiclass ROC-AUC (macro OvR): `0.9996`

Validation-to-test comparison:

- best validation macro F1 at training time: `0.9858`
- held-out test macro F1: `0.9805`

## Per-Class Results

Class order follows `src/brainscan/data/constants.py`.

| Class | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Support |
|---|---:|---:|---:|---:|---:|---:|---:|
| Glioma | 0.9813 | 0.9691 | 0.9752 | 0.9944 | 0.9996 | 0.9986 | 162 |
| Meningioma | 0.9689 | 0.9512 | 0.9600 | 0.9907 | 0.9989 | 0.9964 | 164 |
| Pituitary | 0.9888 | 1.0000 | 0.9944 | 0.9962 | 1.0000 | 0.9999 | 176 |
| No Tumor | 0.9852 | 1.0000 | 0.9926 | 0.9940 | 1.0000 | 1.0000 | 200 |

## Confusion Matrix

Rows are true classes and columns are predicted classes, in canonical order:

```text
[
  [157, 5, 0, 0],
  [3, 156, 2, 3],
  [0, 0, 176, 0],
  [0, 0, 0, 200]
]
```

Major confusion pairs:

- `glioma -> meningioma`: `5`
- `meningioma -> glioma`: `3`
- `meningioma -> no_tumor`: `3`
- `meningioma -> pituitary`: `2`

## High-Confidence Mistakes

Total incorrect predictions: `13`

Highest-confidence incorrect examples:

1. `dataset/train/glioma/Tr-gl_1181.jpg` predicted as `meningioma` with model confidence `0.9563`
2. `dataset/test/glioma/Te-gl_0232.jpg` predicted as `meningioma` with model confidence `0.9359`
3. `dataset/train/meningioma/Tr-me_1268.jpg` predicted as `pituitary` with model confidence `0.9209`
4. `dataset/test/meningioma/Te-me_0178.jpg` predicted as `glioma` with model confidence `0.8381`
5. `dataset/test/meningioma/Te-me_0226.jpg` predicted as `glioma` with model confidence `0.7972`

Confidence summary:

- mean confidence for correct predictions: `0.9899`
- mean confidence for incorrect predictions: `0.6799`
- median confidence for correct predictions: `0.9999`
- median confidence for incorrect predictions: `0.6233`
- incorrect predictions with confidence `>= 0.90`: `3`
- incorrect predictions with confidence `>= 0.95`: `1`
- predictions with top-1 / top-2 margin `< 0.10`: `3`

These values are raw softmax-based model confidence. They are **not** calibrated medical certainty.

## Near-Duplicate Sensitivity Findings

Phase 1B previously found `18,037` near-duplicate candidate pairs. Phase 1E does not alter the official test split. Instead it annotates each formal test prediction using the existing near-duplicate report.

Results:

- flagged test images with cross-split near-duplicate candidates: `74`
- near-duplicate-filtered sensitivity subset size: `628`
- subset accuracy: `0.9825`
- subset balanced accuracy: `0.9820`
- subset macro F1: `0.9821`

Interpretation:

- the filtered subset is slightly better than the full test set
- this is only a sensitivity check
- it is **not** a perfectly independent subject-level evaluation set

## Inference Performance

Measured on the GTX 1650 with CUDA synchronization for pure model timing:

- batch size: `16`
- warm-up iterations: `10`
- timed iterations: `50`
- mean batch latency: `118.06 ms`
- mean per-image latency: `7.38 ms`
- pure-model throughput: `135.52 images/sec`
- checkpoint load time: `1254.12 ms`
- end-to-end evaluation throughput across the 702-image run: `109.74 images/sec`

## Artifacts

Generated evaluation outputs live under `artifacts/evaluation/resnet18_baseline/`.

Key files:

- `metrics.json`
- `classification_report.json`
- `confusion_matrix.json`
- `confusion_pairs.csv`
- `performance.json`
- `predictions.csv`
- `errors.csv`

## Limitations

- subject-level leakage cannot currently be ruled out because reliable subject IDs are unavailable
- near-duplicate candidates remain in the dataset
- the underlying dataset provenance/license is still unclear from the repository contents
- the model output is for research/educational demonstration only
- this result is **not** clinical validation and should not be presented as medical-grade performance
