from pathlib import Path

from brainscan.core.config import get_project_root
from brainscan.data.constants import (
    CANONICAL_CLASS_NAMES,
    CANONICAL_CLASS_TO_ID,
    DATASET_FOLDER_TO_CANONICAL,
    DATASET_FOLDER_TO_ID,
)


def test_exactly_four_canonical_target_classes_exist():
    assert len(CANONICAL_CLASS_NAMES) == 4
    assert CANONICAL_CLASS_NAMES == (
        "glioma",
        "meningioma",
        "pituitary",
        "no_tumor",
    )


def test_canonical_class_ids_are_deterministic():
    assert CANONICAL_CLASS_TO_ID == {
        "glioma": 0,
        "meningioma": 1,
        "pituitary": 2,
        "no_tumor": 3,
    }


def test_physical_dataset_folder_mapping_matches_available_dataset():
    train_dir = get_project_root() / "dataset" / "train"
    actual_folders = sorted(path.name for path in train_dir.iterdir() if path.is_dir())

    assert actual_folders == sorted(DATASET_FOLDER_TO_CANONICAL.keys())
    assert DATASET_FOLDER_TO_CANONICAL["notumor"] == "no_tumor"


def test_dataset_folder_to_id_mapping_is_valid():
    assert DATASET_FOLDER_TO_ID == {
        "glioma": 0,
        "meningioma": 1,
        "pituitary": 2,
        "notumor": 3,
    }
