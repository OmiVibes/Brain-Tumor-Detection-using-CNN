from __future__ import annotations

from inspect import getsource

import numpy as np

from brainscan.segmentation.training.evaluation import compute_subject_metrics, reconstruct_subject_volume
from brainscan.segmentation.training import evaluation as evaluation_module
from brainscan.segmentation.data import dataset as dataset_module


def test_volume_reconstruction_preserves_slice_order() -> None:
    slices = [
        (2, np.full((2, 2), 2, dtype=np.uint8)),
        (0, np.full((2, 2), 0, dtype=np.uint8)),
        (1, np.full((2, 2), 1, dtype=np.uint8)),
    ]
    volume = reconstruct_subject_volume(slices, depth=3)
    assert int(volume[0, 0, 0]) == 0
    assert int(volume[0, 0, 1]) == 1
    assert int(volume[0, 0, 2]) == 2


def test_subject_level_dice_is_one_for_perfect_prediction() -> None:
    target = np.zeros((4, 4, 2), dtype=np.uint8)
    target[1:3, 1:3, :] = 1
    metrics = compute_subject_metrics(target.copy(), target.copy(), subject_id="subject-a")
    assert metrics.dice > 0.99
    assert metrics.iou > 0.99


def test_segmentation_pipeline_does_not_depend_on_tensorflow_or_gradcam_masks() -> None:
    combined_source = getsource(evaluation_module) + getsource(dataset_module)
    assert "tensorflow" not in combined_source.lower()
    assert "gradcam" not in combined_source.lower()
