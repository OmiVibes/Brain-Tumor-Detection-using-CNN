# DenseNet121 Robustness

## Scope

Phase 2D completes the DenseNet121 migration of the BrainScanAI safe-inference stack.

Frozen deployment backbone:

- architecture: `densenet121`
- checkpoint: `artifacts/models/comparison/densenet121_seed42/best.pt`
- checkpoint epoch: `13`
- policy artifact: `artifacts/robustness/densenet121_final/abstention_policy.json`
- policy version: `phase2d_densenet121_v1`

Historical ResNet18 robustness artifacts remain preserved under `artifacts/robustness/resnet18_baseline/`.

## Frozen Policy Inputs

The DenseNet policy was frozen before held-out robustness evaluation.

Policy metadata:

- feature layer: `features.norm5 -> relu -> adaptive_avg_pool2d`
- feature dimension: `1024`
- OOD method: class-conditional diagonal Mahalanobis distance
- OOD warning threshold: `40.4293`
- OOD fail threshold: `47.4629`
- uncertainty probability mode: `calibrated`
- quality thresholds: reused from image-statistics-derived baseline thresholds

Threshold derivation rules:

- OOD thresholds: validation only
- uncertainty thresholds: validation only
- quality thresholds: train + validation image statistics only

## Validation Policy Summary

Frozen validation summary from `validation_policy_summary.json`:

- total: `703`
- `ACCEPT`: `148` (`21.1%`)
- `REVIEW`: `519` (`73.8%`)
- `ABSTAIN`: `36` (`5.1%`)

## Held-out Test Policy Behavior

Held-out test summary on `702` samples:

- `ACCEPT`: `153` (`21.8%`)
- `REVIEW`: `519` (`73.9%`)
- `ABSTAIN`: `30` (`4.3%`)

DenseNet classifier error count on held-out test: `5`

DenseNet error routing:

- error `ACCEPT`: `0`
- error `REVIEW`: `1`
- error `ABSTAIN`: `4`

## Safety Metrics

- unsafe accepted errors: `0`
- error-capture rate: `100.0%`
- hard error-capture rate: `80.0%`
- ACCEPT-only accuracy: `100.0%`
- ACCEPT-only risk: `0.0%`
- ACCEPT-only coverage: `21.8%`
- ACCEPT+REVIEW coverage: `95.7%`
- ACCEPT+REVIEW accuracy: `99.85%`
- ACCEPT+REVIEW risk: `0.15%`

Conservatism:

- correct predictions not automatically accepted: `544 / 697` (`78.0%`)
- false abstentions among correct predictions: `26 / 697` (`3.73%`)

Interpretation:

- the DenseNet policy stayed perfectly safe on accepted cases for this held-out split
- the tradeoff is a much heavier review burden than the historical ResNet baseline

## Synthetic OOD

Using the same 8 reproducible synthetic probes as the baseline:

- `ACCEPT`: `0`
- `REVIEW`: `0`
- `ABSTAIN`: `8`

DenseNet therefore preserved full non-acceptance on this synthetic OOD probe set.

## Perturbation Study

Design:

- `80` MRI slices
- `20` per class
- seed `42`
- same perturbations as the historical baseline

Prediction consistency:

- brightness decrease: `98.8%`
- brightness increase: `100.0%`
- contrast decrease: `97.5%`
- mild blur: `97.5%`
- mild Gaussian noise: `98.8%`
- strong blur: `85.0%`
- strong Gaussian noise: `95.0%`

Status behavior under perturbation:

- strong blur produced the harshest safety response: `63.8% ABSTAIN`
- strong Gaussian noise produced `31.2% ABSTAIN`
- mild perturbations usually stayed in `REVIEW`, with very little automatic acceptance

## Risk-Coverage

The DenseNet risk-coverage curve was derived from validation entropy quantiles only.

Useful held-out points:

- at `95.7%` test coverage, selective risk stayed `0.0%`
- at `99.6%` test coverage, selective risk rose to `0.43%`
- full held-out classifier risk without selection was `0.71%`

This shows the uncertainty signal remains strong, but the frozen production-style policy is substantially more conservative than the raw risk-coverage curve alone.

## Performance

DenseNet latency:

- classifier-only mean latency: `6.93 ms/image`
- classifier-only throughput: `144.22 images/sec`
- safe-inference mean latency: `59.34 ms/image`
- safe-inference throughput: `16.85 images/sec`

Historical ResNet references:

- classifier-only mean latency: `7.38 ms/image`
- safe-inference mean latency: `26.33 ms/image`

Interpretation:

- DenseNet classifier-only inference is slightly faster than the historical ResNet classifier benchmark
- the full DenseNet safe pipeline is substantially slower because it runs the new feature extractor, OOD scoring, calibrated uncertainty path, and decision logic

## ResNet18 vs DenseNet121 System-Level Comparison

Classifier errors:

- ResNet18: `13`
- DenseNet121: `5`

Accepted coverage:

- ResNet18: `40.7%`
- DenseNet121: `21.8%`

Unsafe accepted errors:

- ResNet18: `0`
- DenseNet121: `0`

Error-capture rate:

- ResNet18: `100.0%`
- DenseNet121: `100.0%`

Safe-pipeline latency:

- ResNet18: `26.33 ms/image`
- DenseNet121: `59.34 ms/image`

Bottom line:

- DenseNet121 improved classifier accuracy and kept the same zero unsafe-accept outcome
- the DenseNet safety layer became much more conservative and slower than the historical ResNet safety layer

## Artifacts

- `artifacts/robustness/densenet121_final/abstention_policy.json`
- `artifacts/robustness/densenet121_final/validation_policy_summary.json`
- `artifacts/robustness/densenet121_final/test_policy_results.csv`
- `artifacts/robustness/densenet121_final/error_capture.csv`
- `artifacts/robustness/densenet121_final/synthetic_ood_summary.csv`
- `artifacts/robustness/densenet121_final/perturbation_summary.csv`
- `artifacts/robustness/densenet121_final/risk_coverage.json`
- `artifacts/robustness/densenet121_final/risk_coverage_curve.png`
- `artifacts/robustness/densenet121_final/robustness_metrics.json`

## Conclusion

DenseNet121 is now the correct default BrainScanAI backbone because it improved the frozen classifier itself and preserved zero unsafe accepted errors on the held-out safety run.

At the same time, the migrated DenseNet safe-inference stack is not a free win:

- it accepts far fewer cases automatically
- it pushes much more traffic into review
- it roughly doubles the end-to-end safety latency relative to the historical ResNet safe pipeline

That tradeoff should be stated honestly in interviews and documentation.
