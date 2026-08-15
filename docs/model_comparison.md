# BrainScanAI Architecture Comparison

## Scope

This document records the Phase 2C architecture comparison completed on Saturday, August 15, 2026.

Important rule:

- the held-out test split was not used to rank architectures
- challenger selection was frozen from validation-only evidence before any DenseNet121 test evaluation
- only the frozen DenseNet121 finalist was newly tested in Phase 2C

## Candidates

The comparison benchmark evaluated these ImageNet-pretrained backbones against the same canonical 4-class MRI classification task:

| Architecture | Seeds | Batch Size | Mixed Precision |
| --- | --- | ---: | --- |
| `resnet18` | `42` | 16 | enabled |
| `densenet121` | `42`, `1337` | 8 | disabled |
| `convnext_tiny` | `42` | 4 | enabled |
| `efficientnet_v2_s` | `42` | 8 | enabled |

Shared protocol:

- dataset fingerprint: `b15363ff85ecf29d11c67a2c38fad723a7bb339cb4489e53dd637ab16f8c24b1`
- image size: `224x224`
- optimizer: `AdamW`
- learning rate: `1e-4`
- scheduler: `ReduceLROnPlateau`
- early stopping monitor: validation macro F1
- evaluation device for formal testing: `NVIDIA GeForce GTX 1650` (`4.0 GB`)

## Frozen Selection Procedure

Validation-only comparison artifacts:

- `artifacts/model_comparison/model_runs.json`
- `artifacts/model_comparison/model_comparison.json`
- `artifacts/model_comparison/selection_decision.json`

Selection rule:

- rank by validation-only evidence
- treat architectures within `0.002` macro F1 and `0.002` accuracy of the best mean validation result as near-tied
- break near-ties using latency, checkpoint size, and parameter count

Frozen challenger selected before test:

- architecture: `densenet121`
- finalist experiment: `densenet121_seed42`
- seed: `42`
- checkpoint: `artifacts/models/comparison/densenet121_seed42/best.pt`
- best epoch: `13`
- validation macro F1: `0.9944200759418182`

`selection_decision.json` still records:

- `selected_architecture = densenet121`
- `test_evaluated_before_freeze = false`

## Validation Comparison

| Architecture | Runs | Mean Val Macro F1 | Best Val Macro F1 | Val Macro F1 Std | Mean Val Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `densenet121` | 2 | 0.9930 | 0.9944 | 0.0020 | 0.9929 |
| `convnext_tiny` | 1 | 0.9927 | 0.9927 | n/a | 0.9929 |
| `resnet18` | 1 | 0.9858 | 0.9858 | n/a | 0.9858 |
| `efficientnet_v2_s` | 1 | 0.9856 | 0.9856 | n/a | 0.9858 |

Interpretation:

- DenseNet121 led the comparison on validation performance
- DenseNet121 also showed good seed stability across two runs
- ConvNeXt-Tiny stayed close in validation macro F1, but was much larger and slower

## Parameter, Size, and Latency Comparison

| Architecture | Params | Pure Model Size | Checkpoint Size | Comparison Latency | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 11.18M | 42.64 MB | 128.04 MB | 1.80 ms/image | 555.84 img/s |
| `densenet121` | 6.96M | 26.54 MB | 80.42 MB | 7.15 ms/image | 139.95 img/s |
| `convnext_tiny` | 27.82M | 106.14 MB | 318.59 MB | 9.24 ms/image | 108.17 img/s |
| `efficientnet_v2_s` | 20.18M | 76.99 MB | 232.10 MB | 9.97 ms/image | 100.29 img/s |

Size note:

- pure model size estimates reflect weights only
- checkpoint sizes are larger because they include optimizer and scheduler state plus metadata

VRAM snapshot:

- ResNet18 peak training VRAM: `382.12 MB`
- DenseNet121 peak training VRAM: `1115.13 MB`
- ResNet18 peak benchmark VRAM: `481.78 MB`
- DenseNet121 peak benchmark VRAM: `313.37 MB`
- DenseNet121 formal test benchmark VRAM: `116.81 MB`

## Training-Duration Metadata Caveat

The benchmark sanity audit is saved at:

- `artifacts/model_comparison/benchmark_metadata_audit.json`

Finding:

- `convnext_tiny_seed42` reported `training_duration_seconds = 1.8184558`, which is only the latest resumed segment, not a trustworthy cumulative training total

What was fixed:

- future resumed runs now persist cumulative training duration in `training_state.json`
- existing comparison artifacts were corrected without retraining

What could not be reconstructed:

- the true cumulative ConvNeXt-Tiny training duration could not be derived reliably from existing artifacts
- its duration is therefore marked incomplete / unknown in the updated comparison outputs instead of being invented

## Formal DenseNet121 Held-Out Test

DenseNet121 was evaluated exactly once on the same `702` test images used for the formal ResNet18 baseline.

DenseNet121 overall metrics:

- accuracy: `0.9928774928774928`
- balanced accuracy: `0.992359229147847`
- macro precision: `0.9927140509583227`
- macro recall: `0.992359229147847`
- macro F1: `0.9924979602209649`
- weighted precision: `0.9929225723493396`
- weighted recall: `0.9928774928774928`
- weighted F1: `0.9928626768280072`
- total errors: `5`

Per-class metrics:

| Class | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `glioma` | 0.9938 | 0.9938 | 0.9938 | 0.9981 | 0.99998 | 0.99992 | 162 |
| `meningioma` | 0.9938 | 0.9756 | 0.9846 | 0.9981 | 0.99995 | 0.99985 | 164 |
| `pituitary` | 0.9832 | 1.0000 | 0.9915 | 0.9943 | 1.00000 | 1.00000 | 176 |
| `no_tumor` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.00000 | 1.00000 | 200 |

Confusion matrix:

```text
[[161, 1,   0,   0],
 [  1, 160, 3,   0],
 [  0, 0, 176,   0],
 [  0, 0,   0, 200]]
```

Major confusion pairs:

- `meningioma -> pituitary`: `3`
- `glioma -> meningioma`: `1`
- `meningioma -> glioma`: `1`

Saved evaluation artifacts:

- `artifacts/evaluation/densenet121_finalist/metrics.json`
- `artifacts/evaluation/densenet121_finalist/classification_report.json`
- `artifacts/evaluation/densenet121_finalist/confusion_matrix.json`
- `artifacts/evaluation/densenet121_finalist/confusion_pairs.csv`
- `artifacts/evaluation/densenet121_finalist/performance.json`
- `artifacts/evaluation/densenet121_finalist/paired_comparison.json`

## ResNet18 vs DenseNet121

Reference ResNet18 formal test:

- accuracy: `0.9814814814814815`
- macro F1: `0.9805153483136225`
- total errors: `13`

DenseNet121 delta over ResNet18:

- accuracy difference: `+0.011396011396011357`
- macro F1 difference: `+0.0119826119073424`
- error reduction: `8` fewer errors

Paired disagreement counts:

- both correct: `685`
- both wrong: `1`
- ResNet18 correct / DenseNet121 wrong: `4`
- DenseNet121 correct / ResNet18 wrong: `12`

Exact McNemar result:

- discordant pairs: `16`
- two-sided p-value: `0.076812744140625`

Interpretation:

- the paired evidence favors DenseNet121
- the test set is still small enough that we should avoid overstating the statistical certainty
- the practical improvement is meaningful because it is larger than a 1-2 image swing

## Glioma vs Meningioma Improvement

This was the main clinically interesting error pattern in the ResNet18 baseline.

Confusion change:

- `glioma -> meningioma`: `5` down to `1`
- `meningioma -> glioma`: `3` down to `1`

DenseNet121 materially reduced the dominant cross-class confusion that previously drove several of the baseline errors.

## Final Backbone Decision

Final BrainScanAI backbone:

- `densenet121`

Rationale:

- best validation evidence overall
- stronger formal held-out accuracy and macro F1
- `5` errors instead of `13`
- better `glioma <-> meningioma` behavior
- fewer parameters and smaller pure weight footprint than ResNet18
- inference is slower, but MRI review is not a real-time workload

Why ResNet18 was not kept:

- the DenseNet121 improvement was not microscopic
- the gain was large enough to justify planning a controlled downstream migration

Why DenseNet121 was not declared a complete drop-in replacement yet:

- existing ResNet18 explainability, calibration, OOD, safe inference, and robustness artifacts do not automatically transfer

Required next migration work:

- Grad-CAM target-layer resolver update
- DenseNet121 uncertainty and calibration verification
- DenseNet121 OOD reference rebuild
- safe inference integration
- robustness revalidation

Final decision artifact:

- `artifacts/model_comparison/final_architecture_decision.json`
