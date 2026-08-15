from __future__ import annotations

from pathlib import Path
import json

from brainscan.core import load_best_validation_reference


def test_load_best_validation_reference_from_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"best_epoch": 13, "best_validation_macro_f1": 0.9944}),
        encoding="utf-8",
    )

    payload = load_best_validation_reference(path)

    assert payload["best_epoch"] == 13
    assert payload["best_validation_macro_f1"] == 0.9944


def test_load_best_validation_reference_from_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {"epoch": 1, "val_macro_f1": 0.80},
                {"epoch": 3, "val_macro_f1": 0.91},
                {"epoch": 2, "val_macro_f1": 0.89},
            ]
        ),
        encoding="utf-8",
    )

    payload = load_best_validation_reference(path)

    assert payload["best_epoch"] == 3
    assert payload["best_validation_macro_f1"] == 0.91
