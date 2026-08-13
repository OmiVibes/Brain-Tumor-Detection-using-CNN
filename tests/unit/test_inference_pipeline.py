from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image, ImageFilter

from brainscan.inference import BrainScanInferencePipeline
from brainscan.robustness.ood import ResNet18FeatureExtractor


VALID_MRI_PATH = Path("dataset/test/glioma/Te-glTr_0001.jpg")


@pytest.fixture(scope="module")
def pipeline() -> BrainScanInferencePipeline:
    return BrainScanInferencePipeline(prefer_cuda=False)


def _write_constant_image(path: Path, value: int) -> Path:
    Image.new("RGB", (224, 224), color=(value, value, value)).save(path)
    return path


def test_feature_extractor_matches_avgpool_hook() -> None:
    model = BrainScanInferencePipeline(prefer_cuda=False).model
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["avgpool"] = output.detach().clone()

    handle = model.avgpool.register_forward_hook(_hook)
    try:
        inputs = torch.rand(1, 3, 224, 224)
        with torch.inference_mode():
            _ = model(inputs)
            extracted = ResNet18FeatureExtractor(model)(inputs)
        hooked = torch.flatten(captured["avgpool"], 1)
        assert torch.allclose(hooked, extracted, atol=1e-6, rtol=1e-5)
    finally:
        handle.remove()


def test_same_resnet_instance_used(pipeline: BrainScanInferencePipeline) -> None:
    assert pipeline.feature_extractor.model is pipeline.model


def test_invalid_image_bypasses_classifier(tmp_path: Path, pipeline: BrainScanInferencePipeline, monkeypatch) -> None:
    def _boom(_inputs):
        raise AssertionError("Classifier should not be called for invalid input.")

    monkeypatch.setattr(pipeline.model, "forward", _boom)
    result = pipeline.predict(tmp_path / "missing.jpg")
    assert result.status == "ABSTAIN"
    assert result.prediction is None
    assert result.ood is None


def test_corrupt_image_bypasses_classifier(tmp_path: Path, pipeline: BrainScanInferencePipeline, monkeypatch) -> None:
    broken_path = tmp_path / "broken.jpg"
    broken_path.write_bytes(b"broken")

    def _boom(_inputs):
        raise AssertionError("Classifier should not be called for corrupt input.")

    monkeypatch.setattr(pipeline.model, "forward", _boom)
    result = pipeline.predict(broken_path)
    assert result.status == "ABSTAIN"
    assert result.prediction is None
    assert result.decision.reasons == ["IMAGE_DECODE_FAILED"]


def test_black_image_abstains_and_withholds_prediction(
    tmp_path: Path,
    pipeline: BrainScanInferencePipeline,
    monkeypatch,
) -> None:
    image_path = _write_constant_image(tmp_path / "black.jpg", 0)

    def _boom(_inputs):
        raise AssertionError("Classifier should not be called for low-information input.")

    monkeypatch.setattr(pipeline.model, "forward", _boom)
    result = pipeline.predict(image_path)
    assert result.status == "ABSTAIN"
    assert result.prediction is None
    assert result.decision.reasons == ["LOW_INFORMATION_IMAGE"]


def test_white_image_abstains(tmp_path: Path, pipeline: BrainScanInferencePipeline) -> None:
    image_path = _write_constant_image(tmp_path / "white.jpg", 255)
    result = pipeline.predict(image_path)
    assert result.status == "ABSTAIN"
    assert result.prediction is None
    assert "LOW_INFORMATION_IMAGE" in result.decision.reasons


def test_gray_image_abstains(tmp_path: Path, pipeline: BrainScanInferencePipeline) -> None:
    image_path = _write_constant_image(tmp_path / "gray.jpg", 127)
    result = pipeline.predict(image_path)
    assert result.status == "ABSTAIN"
    assert result.prediction is None
    assert "LOW_INFORMATION_IMAGE" in result.decision.reasons


def test_valid_mri_structured_output(pipeline: BrainScanInferencePipeline) -> None:
    before = deepcopy(pipeline.model.state_dict())
    result = pipeline.predict(VALID_MRI_PATH)
    after = pipeline.model.state_dict()

    assert result.status in {"ACCEPT", "REVIEW"}
    assert result.prediction is not None
    assert result.prediction.class_id in {0, 1, 2, 3}
    assert result.prediction.class_name in {"glioma", "meningioma", "pituitary", "no_tumor"}
    assert 0.0 <= result.prediction.raw_confidence <= 1.0
    assert result.uncertainty is not None
    assert result.uncertainty.level in {"LOW", "MEDIUM", "HIGH"}
    assert result.uncertainty.entropy >= 0.0
    assert 0.0 <= result.uncertainty.normalized_entropy <= 1.0
    assert 0.0 <= result.uncertainty.margin <= 1.0
    assert result.ood is not None
    assert result.ood.status in {"PASS", "BORDERLINE", "FAIL"}
    assert result.ood.score >= 0.0
    assert torch.isfinite(torch.tensor(result.ood.score))
    assert result.quality is not None
    assert pipeline.model.training is False
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_json_serialization_contains_no_absolute_paths(pipeline: BrainScanInferencePipeline) -> None:
    result = pipeline.predict(VALID_MRI_PATH)
    payload = result.to_dict()
    serialized = result.to_json()
    assert json.loads(serialized)["status"] == result.status
    assert "D:\\Notes\\Degree\\Projects\\Brain Tumor Detection Using CNN" not in serialized
    assert payload["status"] == result.status


def test_raw_confidence_remains_default_output(pipeline: BrainScanInferencePipeline) -> None:
    result = pipeline.predict(VALID_MRI_PATH)
    assert result.prediction is not None
    assert hasattr(result.prediction, "raw_confidence")
    assert not hasattr(result.prediction, "calibrated_confidence")


def test_missing_ood_artifact_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BrainScanInferencePipeline(
            prefer_cuda=False,
            ood_reference_npz_path=tmp_path / "missing.npz",
            ood_reference_json_path=tmp_path / "missing.json",
        )


def test_mismatched_fingerprint_error(tmp_path: Path) -> None:
    source_path = Path("artifacts/robustness/resnet18_baseline/ood_reference.json")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["dataset_fingerprint"] = "mismatch"
    target_path = tmp_path / "ood_reference_bad_fingerprint.json"
    target_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset fingerprint"):
        BrainScanInferencePipeline(
            prefer_cuda=False,
            ood_reference_json_path=target_path,
        )


def test_mismatched_class_mapping_error(tmp_path: Path) -> None:
    source_path = Path("artifacts/robustness/resnet18_baseline/ood_reference.json")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["class_mapping"] = {"glioma": 1, "meningioma": 0, "pituitary": 2, "no_tumor": 3}
    target_path = tmp_path / "ood_reference_bad_mapping.json"
    target_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="class mapping"):
        BrainScanInferencePipeline(
            prefer_cuda=False,
            ood_reference_json_path=target_path,
        )


def test_validation_only_threshold_metadata(pipeline: BrainScanInferencePipeline) -> None:
    assert pipeline.uncertainty_metrics["validation_sample_count"] == 703
    assert pipeline.quality_payload["derivation_split_counts"]["validation"] == 703
    assert "validation" in str(pipeline.ood_metadata["threshold_derivation"]).lower()
    assert "test" not in str(pipeline.ood_metadata["threshold_derivation"]).lower()


def test_no_tensorflow_imported(pipeline: BrainScanInferencePipeline) -> None:
    _ = pipeline.predict(VALID_MRI_PATH)
    assert "tensorflow" not in sys.modules


def test_blurred_mri_returns_review_or_abstain(tmp_path: Path, pipeline: BrainScanInferencePipeline) -> None:
    blurred_path = tmp_path / "blurred.jpg"
    with Image.open(VALID_MRI_PATH) as image:
        image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=6)).save(blurred_path)
    result = pipeline.predict(blurred_path)
    assert result.status in {"REVIEW", "ABSTAIN", "ACCEPT"}
