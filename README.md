# BrainScanAI 2.0

BrainScanAI 2.0 is the upgrade path for a legacy student brain MRI tumor classification project into a cleaner, reproducible, interview-ready research prototype.

## Purpose

The long-term goal of this repository is to evolve from a basic Flask + CNN demo into a better engineered brain MRI analysis system with:

- reproducible data, training, and evaluation foundations
- clearer project structure
- valid explainability
- stronger documentation
- safer engineering defaults

The repository is being upgraded in bounded phases rather than rewritten all at once.

## Legacy Background

This repository started as a college project for brain tumor classification from MRI images using TensorFlow/Keras and a Flask upload UI. The legacy implementation already includes:

- MRI upload through Flask
- saved Keras `.h5` models
- training and evaluation scripts
- Grad-CAM-style visualization
- legacy dataset folders under `dataset/`

The preserved legacy system is documented in [docs/legacy_system.md](docs/legacy_system.md), and the exact preserved Git snapshot is tagged as `legacy-v1`.

## Current Upgrade Status

Current milestone: **Phase 2B - robust safe inference and abstention analysis for the frozen ResNet18 baseline**

What is implemented through Phase 1E:

- environment and dependency scaffolding
- central canonical class mapping
- YAML configuration loading
- deterministic dataset audit outputs
- deterministic train/validation/test manifest files
- manifest-backed PyTorch dataset and dataloader utilities
- centralized preprocessing transforms
- reproducibility helpers for seeded loading and training
- ResNet18 classifier builder using torchvision pretrained weights
- reusable training loop with AdamW, ReduceLROnPlateau, AMP-on-CUDA support, checkpointing, and early stopping
- reusable held-out evaluation pipeline with metrics, confusion analysis, confidence analysis, near-duplicate sensitivity analysis, and inference benchmarking
- structured training-history export and training-curve generation
- unit tests for config, data audit, manifests, dataset loading, preprocessing, loaders, reproducibility, classifier building, checkpointing, training behavior, and evaluation logic

What is implemented in Phase 1F:

- reusable Grad-CAM explainability engine for the frozen ResNet18 baseline
- verified ResNet18 target-layer resolution at `layer4[-1].conv2`
- explanation generation from the actual prediction model rather than a surrogate model
- explainability diagnostics and border-attention checks
- review-grid generation for correct and incorrect examples
- unit tests for Grad-CAM behavior and parameter preservation

What is implemented in Phase 2A:

- scalar temperature scaling fitted on `val.csv` only
- reusable calibration and uncertainty package in `src/brainscan/uncertainty/`
- ECE, MCE, NLL, multiclass Brier score, entropy, normalized entropy, and top1-top2 margin
- reliability diagram generation and portable calibration artifacts
- uncertainty-threshold derivation from the validation distribution
- error-detection analysis for incorrect predictions

What is implemented in Phase 2B:

- safe inference pipeline in `src/brainscan/inference/`
- explicit `ACCEPT / REVIEW / ABSTAIN` decision logic
- image validation and low-information rejection
- validation-derived Mahalanobis OOD detection in ResNet18 penultimate feature space
- quality heuristics for blur, brightness, contrast, and low-information inputs
- prediction CLI in `scripts/predict.py`
- frozen-policy robustness evaluation in `scripts/evaluate_robustness.py`
- held-out abstention/error-capture analysis and risk-coverage artifacts

What is intentionally **not** implemented yet:

- model comparison experiments
- FastAPI
- React frontend
- segmentation
- uncertainty estimation

## Current Supported Functionality

### Legacy functionality

- `main.py` legacy Flask app
- `train_model.py` legacy training script
- `test.py` legacy evaluation script
- `multi_stage_classification.py` legacy inference helper
- `grad_cam.py` legacy Grad-CAM utility

### BrainScanAI functionality currently available

- canonical class mapping in `src/brainscan/data/constants.py`
- config loading in `src/brainscan/core/config.py`
- reproducibility helpers in `src/brainscan/core/reproducibility.py`
- dataset audit and split generation scripts
- manifest-backed PyTorch dataset and dataloader utilities
- train/eval preprocessing builders
- inspection script for real MRI dataloaders
- ResNet18 classifier creation in `src/brainscan/models/classifier.py`
- reusable classifier training loop in `src/brainscan/training/train_classifier.py`
- training orchestration script in `scripts/train.py`
- scientifically correct ResNet18 Grad-CAM in `src/brainscan/explainability/gradcam.py`
- explanation generation script in `scripts/generate_gradcam.py`
- calibration and uncertainty tooling in `src/brainscan/uncertainty/`
- calibration study script in `scripts/calibrate.py`
- safe inference pipeline in `src/brainscan/inference/pipeline.py`
- robustness evaluation script in `scripts/evaluate_robustness.py`

## Current Limitations

The repository is still in transition. Important limitations remain:

- the legacy two-stage TensorFlow pipeline still exists unchanged
- the legacy Grad-CAM wiring still exists in legacy code and remains scientifically invalid for BrainScanAI 2.0
- subject-level leakage cannot be ruled out from the available dataset filenames
- dataset provenance/license/citation could not be established from the repository contents
- no API, database, deployment, or product workflow exists yet
- subject-level leakage still cannot be ruled out
- near-duplicate candidates still exist across the dataset and require caution when interpreting strong results

## Research / Educational Disclaimer

BrainScanAI 2.0 is a **research and educational AI prototype**.

It is **not** a medical device, not clinically validated, and not intended to replace expert medical judgment or professional diagnosis.

## Repository Structure

```text
.
|-- artifacts/data_audit/        # dataset audit outputs
|-- configs/                     # project configuration
|-- data/splits/                 # deterministic split manifests
|-- dataset/                     # legacy MRI dataset kept in place
|-- docs/                        # legacy and dataset audit docs
|-- models/                      # legacy model binaries kept in place
|-- requirements/                # dependency group files
|-- scripts/                     # audit, split, inspection, and training scripts
|-- src/brainscan/               # reusable BrainScanAI package
|-- static/                      # legacy uploads and Grad-CAM outputs
|-- templates/                   # legacy Flask template
|-- tests/unit/                  # unit tests
|-- main.py                      # legacy Flask app
|-- train_model.py               # legacy training script
|-- test.py                      # legacy evaluation script
`-- UPGRADE_PLAN.md              # roadmap source of truth
```

## Setup

The original legacy environment versions were not preserved, so dependency files are best-effort compatible constraints rather than exact historical pins.

### 1. Create a project-local virtual environment

For the verified training setup, use a dedicated Python 3.12 virtual environment so the upgraded PyTorch stack stays isolated from legacy/system packages.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the package and base/dev dependencies

```powershell
python -m pip install -e .
python -m pip install -r requirements\base.txt
python -m pip install -r requirements\dev.txt
```

### 3. Install the ML stack

```powershell
python -m pip install -r requirements\ml.txt
```

TensorFlow remains listed because legacy project files still depend on it.

### 4. Install the appropriate PyTorch build

The repository requirements stay generic so the project remains installable on CPU-only and GPU systems. For the verified Phase 1D training run on Windows + NVIDIA GPU, the isolated `.venv` used:

- Python `3.12.8`
- PyTorch `2.12.1+cu130`
- torchvision `0.27.1+cu130`

Install the official PyTorch build appropriate for your machine before running real training.

## Manifest-Based Data Pipeline

Phase 1D continues to use the reusable manifest-based PyTorch data path:

```text
data/splits/*.csv
  ->
BrainMRIDataset
  ->
train / val / test transforms
  ->
PyTorch DataLoaders
```

Important design rules:

- labels come from the manifests, not from scanning folder names
- canonical class IDs come only from `src/brainscan/data/constants.py`
- images are always converted to 3-channel RGB
- the dataset returns image tensors and canonical integer labels
- validation and test preprocessing are deterministic

### Why RGB?

The source MRI images may be grayscale, but Phase 1 converts every sample to RGB because ImageNet-pretrained backbones expect 3-channel input.

### Why 224x224?

Phase 1 training config defaults to `224x224` instead of the legacy `150x150` because `224x224` is the standard transfer-learning input size for common ImageNet-pretrained CNN backbones.

This does **not** change the legacy Flask preprocessing. The old app remains untouched until later migration.

## Preprocessing Strategy

### Train preprocessing

Current conservative train-time pipeline:

- resize to configured image size
- random rotation up to 5 degrees
- tensor conversion
- ImageNet normalization

No aggressive augmentation is used yet.

### Validation and test preprocessing

Validation and test pipelines are deterministic and identical:

- resize to configured image size
- tensor conversion
- ImageNet normalization

### Normalization

ImageNet normalization values are used:

- mean: `(0.485, 0.456, 0.406)`
- std: `(0.229, 0.224, 0.225)`

This is not because MRI intensities naturally match ImageNet statistics. It is used because the first transfer-learning baselines follow the preprocessing convention expected by ImageNet-pretrained backbones.

## Reproducibility

Phase 1D includes reproducibility helpers for:

- Python `random`
- NumPy
- PyTorch
- CUDA seeding where available
- seeded DataLoader worker initialization
- seeded train-loader ordering with a PyTorch `Generator`
- deterministic training configuration capture in checkpoints

This phase aims for deterministic data ordering and reproducible training configuration under controlled settings. It does **not** claim full GPU mathematical determinism for all future workloads.

## ResNet18 Baseline

Phase 1D adds the first clean BrainScanAI baseline:

- architecture: `resnet18`
- initialization: `ResNet18_Weights.IMAGENET1K_V1`
- classes: `glioma`, `meningioma`, `pituitary`, `no_tumor`
- image size: `224x224`
- batch size: `16`
- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `ReduceLROnPlateau`
- loss: `CrossEntropyLoss`
- early stopping monitor: validation macro F1
- dataset fingerprint: `b15363ff85ecf29d11c67a2c38fad723a7bb339cb4489e53dd637ab16f8c24b1`

The baseline fine-tunes the full network and selects the best checkpoint using validation macro F1. The test split is intentionally not used during checkpoint selection.

### Verified baseline run

The first verified full training run used:

- device: `cuda`
- GPU: `NVIDIA GeForce GTX 1650`
- available VRAM: about `4.0 GB`
- AMP: enabled
- maximum epochs configured: `20`
- epochs completed: `14`
- best epoch: `9`

Best validation metrics from the saved history:

- loss: `0.0378`
- accuracy: `0.9858`
- macro precision: `0.9859`
- macro recall: `0.9859`
- macro F1: `0.9858`

Final epoch metrics:

- train loss: `0.0105`
- train accuracy: `0.9977`
- validation loss: `0.0652`
- validation accuracy: `0.9801`
- validation macro F1: `0.9800`

Saved artifacts are written to:

- `artifacts/models/resnet18_baseline_best.pt`
- `artifacts/models/resnet18_baseline_last.pt`
- `artifacts/training/resnet18_baseline_history.json`
- `artifacts/training/resnet18_baseline_history.csv`
- `artifacts/training/loss_curve.png`
- `artifacts/training/accuracy_curve.png`
- `artifacts/training/f1_curve.png`

These generated artifacts are intentionally ignored by Git.

## Held-Out Evaluation

Phase 1E performs the first formal evaluation of the frozen best checkpoint:

- checkpoint: `artifacts/models/resnet18_baseline_best.pt`
- checkpoint epoch: `9`
- dataset fingerprint: `b15363ff85ecf29d11c67a2c38fad723a7bb339cb4489e53dd637ab16f8c24b1`
- formal test sample count: `702`
- device used for evaluation: `cuda` on `NVIDIA GeForce GTX 1650`

### Held-out test metrics

- accuracy: `0.9815`
- balanced accuracy: `0.9801`
- macro precision: `0.9810`
- macro recall: `0.9801`
- macro F1: `0.9805`
- weighted precision: `0.9814`
- weighted recall: `0.9815`
- weighted F1: `0.9814`
- macro ROC-AUC (OvR): `0.9996`

### Per-class held-out results

- glioma: precision `0.9813`, recall `0.9691`, F1 `0.9752`
- meningioma: precision `0.9689`, recall `0.9512`, F1 `0.9600`
- pituitary: precision `0.9888`, recall `1.0000`, F1 `0.9944`
- no_tumor: precision `0.9852`, recall `1.0000`, F1 `0.9926`

### Validation vs held-out test

Keep the two result sets separate:

- best validation macro F1 from training: `0.9858`
- held-out test macro F1: `0.9805`

### Error-analysis highlights

- total incorrect predictions: `13`
- most common confusion pair: `glioma -> meningioma` (`5`)
- next most common: `meningioma -> glioma` (`3`)
- meningioma -> no_tumor also appears (`3`)
- incorrect predictions with model confidence `>= 0.90`: `3`
- incorrect predictions with model confidence `>= 0.95`: `1`

### Near-duplicate sensitivity check

Using the existing Phase 1B near-duplicate report:

- flagged test images with cross-split near-duplicate candidates: `74`
- near-duplicate-filtered sensitivity subset size: `628`
- subset accuracy: `0.9825`
- subset macro F1: `0.9821`

This subset is only a sensitivity analysis. It is **not** a guaranteed subject-independent test set.

### Inference performance

- mean batch latency: `118.06 ms` for batch size `16`
- mean per-image latency: `7.38 ms`
- pure-model throughput: `135.52 images/sec`
- end-to-end evaluation throughput: `109.74 images/sec`

### Limitations

- subject-level leakage cannot currently be ruled out
- near-duplicate candidates remain in the dataset
- dataset provenance/license is still unclear from repository contents
- softmax confidence is raw model confidence, not calibrated medical certainty
- this is a research/educational prototype, not clinical validation

## ResNet18 Explainability

Phase 1F implements scientifically correct Grad-CAM using the exact frozen inference checkpoint:

- checkpoint: `artifacts/models/resnet18_baseline_best.pt`
- checkpoint epoch: `9`
- target layer: `layer4[-1].conv2`
- explanation method: standard Grad-CAM

This replaces the old invalid approach where prediction and explanation were produced by different models.

Important limitation:

- Grad-CAM highlights model attention regions
- it does not prove tumor boundaries, true tumor location, causality, or clinical validity

Representative explainability artifacts:

- `artifacts/explainability/resnet18_baseline/review_grid.png`
- `artifacts/explainability/resnet18_baseline/summary.json`
- [docs/resnet18_explainability.md](docs/resnet18_explainability.md)

## Calibration And Uncertainty

Phase 2A adds validation-only scalar temperature scaling and explicit uncertainty diagnostics for the frozen ResNet18 checkpoint.

Learned temperature:

- `T = 1.1344`

Validation metrics:

- ECE before: `0.007234`
- ECE after: `0.010085`
- NLL before: `0.037927`
- NLL after: `0.037378`

Held-out test metrics:

- ECE before: `0.009226`
- ECE after: `0.010585`
- accuracy unchanged: `0.9815`
- macro F1 unchanged: `0.9805`

Important conclusion:

- temperature scaling is implemented and reusable
- it reduced some high-confidence overconfidence on individual errors
- it did **not** improve overall ECE on validation or test
- it therefore should not become the default BrainScanAI probability calibration yet

Available uncertainty diagnostics:

- raw confidence
- calibrated confidence
- entropy
- normalized entropy
- top1-top2 margin

Calibration artifacts and write-up:

- `artifacts/calibration/resnet18_baseline/temperature.json`
- `artifacts/calibration/resnet18_baseline/metrics.json`
- `artifacts/calibration/resnet18_baseline/reliability_before.png`
- `artifacts/calibration/resnet18_baseline/reliability_after.png`
- [docs/resnet18_calibration.md](docs/resnet18_calibration.md)

## Robust Safe Inference

Phase 2B adds a frozen safety layer around the same ResNet18 classifier:

```text
input validation
-> quality checks
-> ResNet18 inference
-> penultimate-feature OOD score
-> uncertainty analysis
-> ACCEPT / REVIEW / ABSTAIN
```

The deployed robustness policy uses:

- feature layer: `avgpool`
- feature dimension: `512`
- OOD method: class-conditional diagonal Mahalanobis distance
- OOD warning threshold: validation `90th` percentile (`34.5758`)
- OOD fail threshold: validation `95th` percentile (`45.2397`)
- uncertainty thresholds: validation entropy/margin quantiles from Phase 2A

Important limitation:

- this is a research robustness mechanism, not a claim of clinical validity

### Held-out policy behavior on `test.csv`

- total test samples: `702`
- `ACCEPT`: `286` (`40.7%`)
- `REVIEW`: `375` (`53.4%`)
- `ABSTAIN`: `41` (`5.8%`)

### Selective performance

- baseline test accuracy: `98.15%`
- baseline risk: `1.85%` (`13 / 702`)
- ACCEPT-only accuracy: `100.0%`
- ACCEPT-only risk: `0.0%`
- ACCEPT-only coverage: `40.7%`
- ACCEPT+REVIEW coverage: `94.2%`
- ACCEPT+REVIEW accuracy: `99.70%`
- ACCEPT+REVIEW risk: `0.30%`

### Error capture

- total classifier errors on held-out test: `13`
- wrong predictions still `ACCEPT`ed: `0`
- wrong predictions `REVIEW`ed: `2`
- wrong predictions `ABSTAIN`ed: `11`
- total error-capture rate: `100.0%`
- hard error-capture rate (`ABSTAIN` only): `84.6%`

### Conservatism

- correct predictions not automatically accepted: `403 / 689` (`58.5%`)
- false abstentions among correct predictions: `30 / 689` (`4.35%`)

This means the current policy is useful for risk reduction, but still conservative enough to create substantial review burden.

### Synthetic OOD

On 8 obvious synthetic probes:

- `ACCEPT`: `0`
- `REVIEW`: `0`
- `ABSTAIN`: `8`

### Safe inference performance

- safe pipeline mean latency: `26.33 ms/image`
- safe pipeline throughput: `37.97 images/sec`
- Phase 1E classifier-only reference: `7.38 ms/image`

The robustness layer therefore adds meaningful overhead, which is expected because it performs quality checks, feature extraction, OOD scoring, and explicit decision logic on top of classification.

Artifacts and write-up:

- `artifacts/robustness/resnet18_baseline/abstention_policy.json`
- `artifacts/robustness/resnet18_baseline/robustness_metrics.json`
- `artifacts/robustness/resnet18_baseline/test_policy_results.csv`
- `artifacts/robustness/resnet18_baseline/risk_coverage.json`
- `artifacts/robustness/resnet18_baseline/risk_coverage_curve.png`
- [docs/resnet18_robustness.md](docs/resnet18_robustness.md)

## Training

Run the full baseline:

```powershell
.venv\Scripts\python.exe scripts\train.py
```

Run the short smoke test:

```powershell
.venv\Scripts\python.exe scripts\train.py --smoke-test
```

Run the formal held-out evaluation:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py
```

Run Grad-CAM explainability generation:

```powershell
.venv\Scripts\python.exe scripts\generate_gradcam.py
```

Run Grad-CAM for one specific image:

```powershell
.venv\Scripts\python.exe scripts\generate_gradcam.py --image dataset/test/glioma/Te-gl_0232.jpg
```

Run calibration and uncertainty analysis:

```powershell
.venv\Scripts\python.exe scripts\calibrate.py
```

Run safe inference for one image:

```powershell
.venv\Scripts\python.exe scripts\predict.py --image dataset/test/glioma/Te-glTr_0001.jpg
```

Run the formal robustness and abstention study:

```powershell
.venv\Scripts\python.exe scripts\evaluate_robustness.py
```

## Inspecting Real DataLoaders

To inspect the real manifest-backed MRI dataloaders:

```powershell
.venv\Scripts\python.exe scripts\inspect_dataloaders.py
```

The script prints:

- train/val/test dataset sizes
- per-class counts
- one real batch shape
- label tensor shape
- label IDs
- tensor dtypes
- NaN/Inf checks

## Verification

Current verification commands:

```powershell
.venv\Scripts\python.exe -m compileall src scripts
.venv\Scripts\python.exe -m pytest tests/unit -q
.venv\Scripts\python.exe scripts\inspect_dataloaders.py
.venv\Scripts\python.exe scripts\train.py --smoke-test
.venv\Scripts\python.exe scripts\evaluate.py
.venv\Scripts\python.exe scripts\generate_gradcam.py
.venv\Scripts\python.exe scripts\calibrate.py
.venv\Scripts\python.exe scripts\predict.py --image dataset/test/glioma/Te-glTr_0001.jpg
.venv\Scripts\python.exe scripts\evaluate_robustness.py
```

## Roadmap

The upgrade roadmap is defined in [UPGRADE_PLAN.md](UPGRADE_PLAN.md).

The next phases will build on this foundation, but the project should only claim features that are actually implemented and verified.
