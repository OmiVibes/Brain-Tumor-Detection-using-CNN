# Legacy System Audit

## Purpose

This document preserves the current legacy project state before the BrainScanAI 2.0 upgrade begins. It describes the legacy architecture, runtime behavior, dataset and model assets present in the repository, and the known scientific and engineering issues that must be addressed later.

This is a preservation document only. It does not change the legacy ML logic.

## Current Architecture

The legacy project is a single-folder TensorFlow + Flask application with training, inference, evaluation, explainability, and UI logic mixed together in the repository root.

Primary code files:

- `main.py`: Flask web application and inference entrypoint
- `train_model.py`: training script for the saved models
- `test.py`: model evaluation script
- `multi_stage_classification.py`: standalone 2-stage inference script
- `grad_cam.py`: Grad-CAM generation utility
- `abc.py`: debug helper that prints model summary / last conv layer
- `templates/index.html`: single-page HTML frontend

Current top-level asset folders:

- `dataset/`
- `models/`
- `Matrix/`
- `static/uploads/`
- `static/gradcam/`

## Current Runtime Flow

Legacy runtime flow in `main.py`:

1. Start Flask app.
2. Load `models/tumor_detection.h5`.
3. Load `models/tumor_classification.h5`.
4. Also load an ImageNet `MobileNetV2` model for Grad-CAM generation.
5. User uploads a PNG/JPG/JPEG image through the web UI.
6. Image is saved directly to `static/uploads/`.
7. Image is resized to `150x150`, converted to RGB array, normalized to `[0, 1]`.
8. Stage 1 model predicts tumor vs no tumor.
9. If tumor is detected, Stage 2 model predicts one of:
   - Glioma
   - Meningioma
   - No Tumor
   - Pituitary
10. Grad-CAM output is generated and saved to `static/gradcam/`.
11. The result page renders the uploaded image, prediction text, confidence, and Grad-CAM image.

## Existing Dataset Structure

The repository currently contains a folder-per-class image dataset in Keras-style layout:

```text
dataset/
  train/
    glioma/
    meningioma/
    notumor/
    pituitary/
  test/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

Observed counts:

- Train total: 5712 images
  - `glioma`: 1321
  - `meningioma`: 1339
  - `notumor`: 1595
  - `pituitary`: 1457
- Test total: 1311 images
  - `glioma`: 300
  - `meningioma`: 306
  - `notumor`: 405
  - `pituitary`: 300

Total observed image count: 7023

The dataset remains in place and is intentionally not moved during Phase 0.

## Existing Models

Observed saved model files:

- `models/tumor_detection.h5`
  - size: 127,678,168 bytes
- `models/tumor_classification.h5`
  - size: 57,993,776 bytes

These model binaries remain in place and are intentionally not modified during Phase 0.

## Existing Evaluation Outputs

Observed evaluation / visualization outputs:

- `Matrix/confusion_matrix.png`
- `Matrix/roc_curve.png`

Observed generated web artifacts:

- `static/uploads/` contains previously uploaded/sample input images
- `static/gradcam/` contains generated Grad-CAM overlay images

These artifacts are preserved on disk as part of the legacy project state, but they are not appropriate source-controlled assets for the new Git history.

## Known Architectural Problems

### Redundant two-stage pipeline

The legacy app uses:

```text
MRI
 -> tumor / no tumor detector
 -> 4-class classifier
```

But the 4-class classifier itself already includes `No Tumor`, so the two-stage design is architecturally inconsistent and redundant.

### Binary training logic conflict

`train_model.py` trains a binary detector using `flow_from_directory(..., class_mode='binary')` against a dataset directory that contains four class folders. This is scientifically questionable and may not represent a valid tumor-vs-no-tumor training setup.

### Mixed concerns

Training, evaluation, inference, Grad-CAM, debugging, and web UI logic are all mixed into a flat project layout with no reusable package structure.

### Missing reproducibility foundation

The legacy project has:

- no environment definition
- no dependency lock/setup file
- no deterministic split manifest
- no saved validation split artifact
- no seed management
- no test suite

## Known Grad-CAM Problem

The Grad-CAM shown by the Flask app is not generated from the actual prediction model.

In `main.py`, the application loads a separate ImageNet `MobileNetV2` model and passes that model into `grad_cam.generate_gradcam(...)`. As a result, the displayed heatmap is not a valid explanation of the classifier that produced the prediction.

This is the most important scientific issue in the legacy explainability implementation.

## Unsupported UI Claims

`templates/index.html` currently contains claims that are not supported by the code or reproducible artifacts in the repository:

- Claim of `98.7% accuracy`
- Claim that the system uses transfer learning
- Claim that uploaded images are deleted after analysis
- General wording that implies stronger maturity than the code demonstrates

The legacy UI should be treated as a student/demo interface rather than a validated product.

## Current Dependencies Identifiable From Code

Python dependencies identifiable from imports:

- `flask`
- `tensorflow`
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `seaborn`
- `scikit-learn`

TensorFlow/Keras features used:

- `tensorflow.keras.models.load_model`
- `tensorflow.keras.preprocessing.image`
- `tensorflow.keras.applications.MobileNetV2`
- `ImageDataGenerator`
- Keras Sequential CNN layers and callbacks

Frontend/CDN dependencies identifiable from HTML:

- Google Fonts (`Poppins`, `Roboto`)
- Font Awesome CDN

Exact package versions are not currently documented in the legacy project.

## Files Preserved In Git For Legacy Source Snapshot

The Phase 0 Git snapshot is intended to preserve the legacy source code and documentation, not the full raw dataset or generated artifacts.

Legacy source files expected to be preserved in Git:

- `main.py`
- `train_model.py`
- `test.py`
- `multi_stage_classification.py`
- `grad_cam.py`
- `abc.py`
- `templates/index.html`
- `UPGRADE_PLAN.md`
- `docs/legacy_system.md`

## Files Intentionally Left Out Of Git Snapshot

The following remain on disk but are intentionally not committed in Phase 0:

- `dataset/` image files
- `models/*.h5`
- `static/uploads/`
- `static/gradcam/`
- `Matrix/`
- `__pycache__/`

This preserves the working legacy project locally while avoiding a first commit full of large binaries, generated artifacts, and cache files.
