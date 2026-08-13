from pathlib import Path

from brainscan.core.config import (
    get_project_root,
    load_inference_config,
    load_train_config,
    make_project_relative_path,
    resolve_project_path,
)


def test_train_config_loads_with_required_sections():
    config = load_train_config()

    assert "training" in config
    assert "dataset" in config
    assert "model" in config


def test_inference_config_loads_with_required_sections():
    config = load_inference_config()

    assert "dataset" in config
    assert "model" in config
    assert "inference" in config


def test_project_relative_paths_resolve_to_repo_root():
    project_root = get_project_root()
    expected = (project_root / "dataset" / "train").resolve()

    resolved = resolve_project_path("dataset/train")

    assert resolved == expected
    assert resolved.is_absolute()


def test_config_resolves_declared_paths():
    config = load_train_config()

    assert Path(config["dataset"]["root"]).name == "dataset"
    assert Path(config["dataset"]["train_dir"]).name == "train"
    assert Path(config["dataset"]["test_dir"]).name == "test"


def test_make_project_relative_path_returns_portable_repo_path():
    relative = make_project_relative_path("artifacts/explainability/resnet18_baseline/review_grid.png")

    assert relative == "artifacts/explainability/resnet18_baseline/review_grid.png"


def test_make_project_relative_path_rejects_external_absolute_path():
    external = Path.cwd().anchor + "outside-project" if Path.cwd().anchor else "/outside-project"

    try:
        make_project_relative_path(external)
    except ValueError as exc:
        assert "outside the project root" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a path outside the project root.")
