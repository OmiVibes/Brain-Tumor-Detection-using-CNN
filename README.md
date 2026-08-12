# BrainScanAI 2.0

BrainScanAI 2.0 is the planned upgrade path for a legacy student brain MRI tumor classification project into a cleaner, reproducible, interview-ready research prototype.

## Purpose

The long-term goal of this repository is to evolve from a basic Flask + CNN demo into a better engineered brain MRI analysis system with:

- reproducible training and evaluation
- clearer project structure
- valid explainability
- stronger documentation
- safer engineering defaults

Phase 1A does **not** implement the full upgraded system. It establishes the repository foundation needed for later migration work.

## Legacy Background

This repository started as a college project for brain tumor classification from MRI images using TensorFlow/Keras and a Flask upload UI. The original implementation already includes:

- MRI image upload through Flask
- saved Keras models
- tumor detection and classification scripts
- evaluation plotting
- Grad-CAM-style visualization

The legacy code is preserved and documented in [docs/legacy_system.md](docs/legacy_system.md). The exact preserved legacy snapshot is tagged in Git as `legacy-v1`.

## Current Upgrade Status

Current milestone: **Phase 1A - repository foundation and reproducibility setup**

What Phase 1A adds:

- professional repository documentation
- dependency and environment scaffolding
- initial YAML configuration files
- minimal `src/brainscan` package structure
- canonical four-class mapping
- lightweight configuration loader
- initial unit tests

What Phase 1A does **not** change yet:

- legacy ML architecture
- saved legacy models
- legacy dataset placement
- legacy Flask app behavior
- legacy training script behavior

## Current Supported Functionality

Today, the repository contains two layers of functionality:

### Legacy functionality

- `main.py` legacy Flask app
- `train_model.py` legacy training script
- `test.py` legacy evaluation script
- `multi_stage_classification.py` legacy inference helper
- `grad_cam.py` legacy Grad-CAM utility

### New Phase 1A foundation

- central class mapping in `src/brainscan/data/constants.py`
- YAML config loading in `src/brainscan/core/config.py`
- initial `configs/` files
- initial tests under `tests/unit/`

## Current Limitations

The repository is still in transition. Important limitations remain:

- legacy two-stage classification logic has not been replaced yet
- legacy Grad-CAM is still scientifically invalid for the real prediction model
- reproducible train/validation/test manifests have not been built yet
- legacy environment versions were not recorded originally
- no new classifier training pipeline exists yet
- no FastAPI, database, segmentation, DICOM, or frontend rebuild has started

## Research / Educational Disclaimer

BrainScanAI 2.0 is a **research and educational AI prototype**.

It is **not** a medical device, not clinically validated, and not intended to replace expert medical judgment or professional diagnosis.

## Repository Structure

Current important paths:

```text
.
├── configs/                  # Phase 1A configuration files
├── docs/                     # legacy system preservation docs
├── requirements/             # dependency group files
├── src/brainscan/            # new package foundation
├── templates/                # legacy Flask template
├── tests/unit/               # Phase 1A unit tests
├── dataset/                  # legacy MRI image dataset (kept in place)
├── models/                   # legacy model binaries (kept in place)
├── static/                   # legacy uploads and Grad-CAM outputs
├── main.py                   # legacy Flask app
├── train_model.py            # legacy training script
├── test.py                   # legacy evaluation script
├── UPGRADE_PLAN.md           # source-of-truth roadmap
└── docs/legacy_system.md     # preserved legacy architecture audit
```

## Setup

Phase 1A only makes setup partially reproducible. The original legacy environment versions were not recorded, so the dependency files are best-effort compatible constraints rather than exact historical pins.

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the Phase 1A package and base/dev dependencies

```powershell
python -m pip install -e .
python -m pip install -r requirements\base.txt
python -m pip install -r requirements\dev.txt
```

### 3. Optional: install legacy/ML-oriented dependencies

```powershell
python -m pip install -r requirements\ml.txt
```

Note: depending on Python version and platform, TensorFlow compatibility may require adjustment because the original project did not preserve its exact package versions.

## Verification

Phase 1A verification commands:

```powershell
python -m compileall src
python -m pytest tests/unit -q
```

## Roadmap

The upgrade roadmap is defined in [UPGRADE_PLAN.md](UPGRADE_PLAN.md).

For now, this repository is intentionally moving in small verified steps rather than rewriting the whole system at once.
