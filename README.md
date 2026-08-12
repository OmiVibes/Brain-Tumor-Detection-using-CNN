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

Current milestone: **Phase 1C - manifest-backed PyTorch data pipeline**

What is implemented through Phase 1C:

- environment and dependency scaffolding
- central canonical class mapping
- YAML configuration loading
- deterministic dataset audit outputs
- deterministic train/validation/test manifest files
- manifest-backed PyTorch dataset and dataloader utilities
- centralized preprocessing transforms
- reproducibility helpers for seeded loading
- unit tests for config, data audit, manifests, dataset loading, preprocessing, loaders, and reproducibility

What is intentionally **not** implemented yet:

- model training
- ResNet18 or any other classifier architecture
- loss/optimizer logic
- Grad-CAM v2
- evaluation metrics for a new model
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

### BrainScanAI foundation currently available

- canonical class mapping in `src/brainscan/data/constants.py`
- config loading in `src/brainscan/core/config.py`
- reproducibility helpers in `src/brainscan/core/reproducibility.py`
- dataset audit and split generation scripts
- manifest-backed PyTorch dataset and dataloader utilities
- train/eval preprocessing builders
- inspection script for real MRI dataloaders

## Current Limitations

The repository is still in transition. Important limitations remain:

- the legacy two-stage TensorFlow pipeline still exists unchanged
- the legacy Grad-CAM wiring is still scientifically invalid
- no new model has been trained yet
- subject-level leakage cannot be ruled out from the available dataset filenames
- dataset provenance/license/citation could not be established from the repository contents
- no API, database, deployment, or product workflow exists yet

## Research / Educational Disclaimer

BrainScanAI 2.0 is a **research and educational AI prototype**.

It is **not** a medical device, not clinically validated, and not intended to replace expert medical judgment or professional diagnosis.

## Repository Structure

```text
.
├── artifacts/data_audit/        # dataset audit outputs
├── configs/                     # project configuration
├── data/splits/                 # deterministic split manifests
├── dataset/                     # legacy MRI dataset kept in place
├── docs/                        # legacy and dataset audit docs
├── models/                      # legacy model binaries kept in place
├── requirements/                # dependency group files
├── scripts/                     # audit, split, and inspection scripts
├── src/brainscan/               # reusable BrainScanAI package
├── static/                      # legacy uploads and Grad-CAM outputs
├── templates/                   # legacy Flask template
├── tests/unit/                  # unit tests
├── main.py                      # legacy Flask app
├── train_model.py               # legacy training script
├── test.py                      # legacy evaluation script
└── UPGRADE_PLAN.md              # roadmap source of truth
```

## Setup

The original legacy environment versions were not preserved, so dependency files are best-effort compatible constraints rather than exact historical pins.

### 1. Create a virtual environment

```powershell
python -m venv .venv
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

The current repository environment already supports the Phase 1C data pipeline with PyTorch and torchvision. TensorFlow remains listed because legacy project files still depend on it.

## Manifest-Based Data Pipeline

Phase 1C introduces a reusable manifest-based PyTorch data path:

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

The source MRI images may be grayscale, but Phase 1 converts every sample to RGB because upcoming ImageNet-pretrained backbones expect 3-channel input.

### Why 224x224?

Phase 1 training config now defaults to `224x224` instead of the legacy `150x150` because `224x224` is the standard transfer-learning input size for common ImageNet-pretrained CNN backbones.

This does **not** change the legacy Flask preprocessing yet. The old app remains untouched until later migration.

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

This is not because MRI intensities naturally match ImageNet statistics. It is used because the first transfer-learning baselines will follow the preprocessing convention expected by ImageNet-pretrained backbones.

## Reproducibility

Phase 1C includes reproducibility helpers for:

- Python `random`
- NumPy
- PyTorch
- CUDA seeding where available
- seeded DataLoader worker initialization
- seeded train-loader ordering with a PyTorch `Generator`

This phase aims for deterministic data ordering under controlled settings. It does **not** claim full GPU mathematical determinism for all future training workloads.

## Inspecting Real DataLoaders

To inspect the real manifest-backed MRI dataloaders:

```powershell
python scripts\inspect_dataloaders.py
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
python -m compileall src scripts
python -m pytest tests/unit -q
python scripts\inspect_dataloaders.py
```

## Roadmap

The upgrade roadmap is defined in [UPGRADE_PLAN.md](UPGRADE_PLAN.md).

The next phases will build on this foundation, but the project should only claim features that are actually implemented and verified.
