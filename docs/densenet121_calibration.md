# DenseNet121 Calibration

## Scope

Phase 2D recalibrates uncertainty for the frozen DenseNet121 finalist using validation-only temperature scaling.

- checkpoint: `artifacts/models/comparison/densenet121_seed42/best.pt`
- checkpoint epoch: `13`
- architecture: `densenet121`
- calibration method: scalar temperature scaling
- fitting split: validation only
- learned temperature: `0.8621`

Historical ResNet18 calibration artifacts remain preserved under `artifacts/calibration/resnet18_baseline/`.

## Validation-Only Rule

DenseNet calibration fitting used only the frozen validation split.

No held-out test sample was used to:

- fit temperature
- derive uncertainty thresholds
- choose the default probability mode

The default probability mode was selected using validation metrics only.

## Metrics

### Validation

- accuracy before/after: `0.9943 / 0.9943`
- macro F1 before/after: `0.9944 / 0.9944`
- ECE before/after: `0.004602 / 0.004211`
- NLL before/after: `0.018061 / 0.017713`
- Brier before/after: `0.009458 / 0.009294`

### Held-out test

- accuracy before/after: `0.9929 / 0.9929`
- macro F1 before/after: `0.9925 / 0.9925`
- ECE before/after: `0.005982 / 0.005018`
- NLL before/after: `0.018250 / 0.018494`
- Brier before/after: `0.009850 / 0.010014`

Important interpretation:

- validation calibration improved ECE, NLL, and Brier simultaneously
- held-out test ECE improved
- held-out test NLL and Brier worsened slightly

## Default Probability Decision

Final default probability mode: `calibrated`

Reason:

- validation ECE improved
- validation NLL improved
- validation Brier improved
- accuracy and macro F1 stayed unchanged

The default decision was frozen from validation behavior, not from test behavior.

## DenseNet Uncertainty Thresholds

DenseNet-specific thresholds derived from calibrated validation probabilities only:

- `entropy_q25 = 9.024162409332348e-06`
- `entropy_q75 = 0.0005972670041956007`
- `margin_q25 = 0.9998575448989868`
- `margin_q75 = 0.9999985694885254`

Classification rule:

- `HIGH` if normalized entropy `>= entropy_q75` or margin `<= margin_q25`
- `LOW` if normalized entropy `<= entropy_q25` and margin `>= margin_q75`
- otherwise `MEDIUM`

## Error Detection

DenseNet test-set error count: `5`

Selected AUROC using the frozen calibrated mode:

- entropy AUROC: `0.9968`
- `1 - confidence` AUROC: `0.9954`

These estimates are based on only 5 errors, so they should be treated as high-variance indicators rather than stable clinical-style evidence.

## Artifacts

- `artifacts/calibration/densenet121_final/metrics.json`
- `artifacts/calibration/densenet121_final/temperature.json`
- `artifacts/calibration/densenet121_final/reliability_before.png`
- `artifacts/calibration/densenet121_final/reliability_after.png`

## Takeaway

DenseNet121 benefited enough on validation to justify calibrated probabilities as the default uncertainty mode for the BrainScanAI safe-inference stack, but the held-out test tradeoff is mixed enough that calibration should still be described carefully rather than oversold.
