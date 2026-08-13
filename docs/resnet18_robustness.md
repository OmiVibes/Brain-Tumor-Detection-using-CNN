# ResNet18 Robustness And Abstention Study

## Scope

This document summarizes BrainScanAI 2.0 Phase 2B for the frozen ResNet18 baseline on Thursday, August 13, 2026.

The objective was to freeze the existing safe-inference policy from Commit 2 and evaluate how it behaves on:

- clean validation MRI samples
- clean held-out test MRI samples
- obvious synthetic OOD probes
- controlled perturbations of valid MRI images
- selective-classification risk/coverage analysis

This is a research robustness layer. It does **not** establish clinical validity.

## Safe Inference Architecture

Phase 2B uses the following pipeline:

```text
input validation
-> quality checks
-> frozen ResNet18 inference
-> penultimate-feature OOD score
-> uncertainty analysis
-> ACCEPT / REVIEW / ABSTAIN
```

Core implementation paths:

- `src/brainscan/inference/pipeline.py`
- `src/brainscan/robustness/input_validation.py`
- `src/brainscan/robustness/quality.py`
- `src/brainscan/robustness/ood.py`
- `src/brainscan/robustness/abstention.py`
- `scripts/evaluate_robustness.py`

## Frozen Policy

Frozen policy artifact:

- `artifacts/robustness/resnet18_baseline/abstention_policy.json`

Checkpoint and data identity:

- checkpoint: `artifacts/models/resnet18_baseline_best.pt`
- checkpoint epoch: `9`
- architecture: `resnet18`
- dataset fingerprint: `b15363ff85ecf29d11c67a2c38fad723a7bb339cb4489e53dd637ab16f8c24b1`

### OOD method

- feature layer: `avgpool`
- feature dimension: `512`
- OOD method: class-conditional diagonal Mahalanobis distance
- warning threshold: `34.5757531321482`
- fail threshold: `45.239676435455095`
- threshold source: validation `90th` and `95th` percentiles only

### Quality thresholds

- brightness low: `20.2136`
- brightness high: `107.8885`
- contrast low: `27.7216`
- blur low: `18.6710`
- near-black high: `0.72498`
- near-white high: `0.09227`
- dynamic-range low: `216.0`
- low-information dynamic range: `5.0`
- low-information contrast std: `2.0`

### Uncertainty thresholds

- entropy q25: `0.0005596511`
- entropy q75: `0.0079716831`
- margin q25: `0.9975114167`
- margin q75: `0.9998722672`

### Decision rules

`ABSTAIN`

- invalid or corrupt input
- `LOW_INFORMATION_IMAGE`
- OOD status `FAIL`

`REVIEW`

- OOD status `BORDERLINE`
- quality warnings
- uncertainty level `MEDIUM` or `HIGH`

`ACCEPT`

- valid input
- no low-information flag
- OOD status `PASS`
- uncertainty level `LOW`
- no quality warnings

## Validation Behavior

Validation sample count: `703`

- `ACCEPT`: `297` (`42.25%`)
- `REVIEW`: `370` (`52.63%`)
- `ABSTAIN`: `36` (`5.12%`)

This validation distribution was frozen before held-out test robustness evaluation.

## Held-Out Test Behavior

Held-out test sample count: `702`

- `ACCEPT`: `286` (`40.74%`)
- `REVIEW`: `375` (`53.42%`)
- `ABSTAIN`: `41` (`5.84%`)

Coverage, where coverage means `ACCEPT / 702`:

- coverage: `40.74%`

Per-sample artifact:

- `artifacts/robustness/resnet18_baseline/test_policy_results.csv`

## Correct Prediction Non-Acceptance

Correct classifier predictions: `689`

- correct + `ACCEPT`: `286`
- correct + `REVIEW`: `373`
- correct + `ABSTAIN`: `30`

Derived rates:

- correct acceptance rate: `41.51%`
- correct review rate: `54.14%`
- false abstention rate: `4.35%`
- correct non-acceptance rate (`REVIEW + ABSTAIN`): `58.49%`

This is the main reason the current policy feels conservative.

## Error Capture

Phase 1E reported `13` held-out classifier errors. Under the frozen Phase 2B policy:

- wrong + `ACCEPT`: `0`
- wrong + `REVIEW`: `2`
- wrong + `ABSTAIN`: `11`

Derived robustness metrics:

- unsafe accepted-error count: `0`
- unsafe accepted-error rate among all test samples: `0.0%`
- error-capture rate (`REVIEW + ABSTAIN` among wrong predictions): `100.0%`
- hard error-capture rate (`ABSTAIN` among wrong predictions): `84.62%`

Detailed error artifact:

- `artifacts/robustness/resnet18_baseline/error_capture.csv`

## Selective Accuracy And Risk

Baseline held-out classifier:

- baseline accuracy: `98.15%`
- baseline risk: `1.85%`

`ACCEPT` only:

- accepted sample count: `286`
- accepted-only accuracy: `100.0%`
- accepted-only selective risk: `0.0%`
- accepted-only coverage: `40.74%`

`ACCEPT + REVIEW`:

- sample count: `661`
- coverage: `94.16%`
- accuracy: `99.70%`
- risk: `0.30%`

Interpretation:

- the policy removes all wrong predictions from automatic acceptance
- but it does so by pushing most correct predictions into `REVIEW`

## Risk-Coverage Analysis

Artifacts:

- `artifacts/robustness/resnet18_baseline/risk_coverage.json`
- `artifacts/robustness/resnet18_baseline/risk_coverage_curve.png`

Ranking score:

- normalized entropy only
- thresholds derived from validation quantiles only

Observed behavior:

- on validation, selective risk stayed `0` until full coverage
- on held-out test, selective risk also stayed `0` until about `94.0%` coverage
- full coverage returns the baseline test risk of `1.85%`

Important nuance:

- this curve is an uncertainty-ranking study
- it does not retune the deployed Phase 2B policy
- it shows that the current explicit `ACCEPT` policy is more conservative than pure entropy-based selective classification

## Synthetic OOD Probes

Probe set size: `8`

Probe types:

- black
- white
- constant gray
- gradient
- checkerboard
- random RGB noise
- Gaussian noise low
- Gaussian noise high

Results:

- `ACCEPT`: `0`
- `REVIEW`: `0`
- `ABSTAIN`: `8`

Derived rates:

- OOD abstention rate: `100.0%`
- OOD non-acceptance rate: `100.0%`

Summary artifact:

- `artifacts/robustness/resnet18_baseline/synthetic_ood_summary.csv`

This is an obvious synthetic stress test, not an external clinical OOD benchmark.

## Controlled Perturbation Study

Stratified test subset:

- `20` samples per class
- total subset size: `80`
- fixed seed: `42`

Perturbations:

- mild Gaussian noise
- strong Gaussian noise
- mild blur
- strong blur
- brightness decrease
- brightness increase
- contrast decrease

Summary artifact:

- `artifacts/robustness/resnet18_baseline/perturbation_summary.csv`

### Results

Mild Gaussian noise:

- prediction consistency: `97.5%`
- mean OOD score change: `+0.42`
- `ACCEPT / REVIEW / ABSTAIN`: `37.5% / 55.0% / 7.5%`

Strong Gaussian noise:

- prediction consistency: `95.0%`
- mean OOD score change: `+18.05`
- `ACCEPT / REVIEW / ABSTAIN`: `7.5% / 52.5% / 40.0%`

Mild blur:

- prediction consistency: `96.25%`
- mean OOD score change: `+3.97`
- `ACCEPT / REVIEW / ABSTAIN`: `10.0% / 76.25% / 13.75%`

Strong blur:

- prediction consistency: `81.25%`
- mean OOD score change: `+16.20`
- `ACCEPT / REVIEW / ABSTAIN`: `0.0% / 61.25% / 38.75%`

Brightness decrease:

- prediction consistency: `98.75%`
- mean OOD score change: `+1.82`
- `ACCEPT / REVIEW / ABSTAIN`: `0.0% / 90.0% / 10.0%`

Brightness increase:

- prediction consistency: `97.5%`
- mean OOD score change: `+1.14`
- `ACCEPT / REVIEW / ABSTAIN`: `37.5% / 53.75% / 8.75%`

Contrast decrease:

- prediction consistency: `98.75%`
- mean OOD score change: `+2.15`
- `ACCEPT / REVIEW / ABSTAIN`: `0.0% / 91.25% / 8.75%`

Interpretation:

- stronger corruptions increase OOD score and abstention, which is desirable
- mild perturbations often preserve the class prediction
- however, the deployed policy still routes many mildly perturbed but still-correct MRIs into `REVIEW`

## Classwise Policy Behavior

Per true class `ACCEPT / REVIEW / ABSTAIN` counts on held-out test:

- glioma: `67 / 84 / 11`
- meningioma: `48 / 96 / 20`
- pituitary: `97 / 76 / 3`
- no_tumor: `74 / 119 / 7`

Most conservative class:

- meningioma had the lowest `ACCEPT` count and highest `ABSTAIN` count

This aligns with Phase 1E, where glioma and meningioma formed the main confusion pair.

## OOD Scores And Uncertainty

OOD score summaries:

- validation clean mean / median: `23.79 / 20.65`
- test clean mean / median: `24.36 / 21.25`
- synthetic OOD mean / median: `65.30 / 58.96`
- perturbed mean / median: `30.95 / 25.74`

Entropy / OOD relationship on clean test:

- Pearson correlation between normalized entropy and OOD score: `0.7588`

Interpretation:

- higher uncertainty and higher OOD distance tend to co-occur
- they are related but still conceptually different signals

## Quality Warnings

Most common quality warnings on clean held-out test:

- `LOW_DYNAMIC_RANGE`: `15`
- `VERY_DARK_IMAGE`: `10`
- `LOW_CONTRAST_IMAGE`: `10`
- `NEAR_BLACK_DOMINANT`: `9`
- `NEAR_WHITE_DOMINANT`: `7`

This suggests the quality heuristics are not firing on most clean MRIs, but they still contribute to the large `REVIEW` bucket in a minority of cases.

## Latency

Held-out sequential safe inference:

- mean safe-inference latency: `26.33 ms/image`
- safe throughput: `37.97 images/sec`

Phase 1E classifier-only reference:

- classifier-only mean latency: `7.38 ms/image`

The safe pipeline is about `3.57x` slower than plain classifier inference, which is expected because it performs:

- file validation
- image quality analysis
- feature extraction
- OOD scoring
- uncertainty logic
- explicit decision routing

## Limitations

- this robustness layer is a research safety mechanism, not a medical safety guarantee
- synthetic OOD probes are obvious stress tests, not a real external OOD benchmark
- subject-level leakage still cannot be ruled out in the dataset
- near-duplicate candidates still exist across splits
- the current explicit `ACCEPT` policy is conservative and creates substantial review burden
- the test-set risk-coverage curve suggests more permissive future policies may be possible, but Phase 2B does not retune thresholds after seeing test outcomes

## Bottom Line

The current Phase 2B policy is useful because it eliminated unsafe accepted errors on the held-out test set and captured all `13` known classifier mistakes.

At the same time, it is conservative:

- only `40.7%` of held-out test MRIs are automatically accepted
- `58.5%` of correct classifier predictions are not automatically accepted

The most accurate factual conclusion is:

> The current policy significantly reduces accepted error risk, but it does so with substantial review burden and is likely too conservative for automatic acceptance in its present form.
