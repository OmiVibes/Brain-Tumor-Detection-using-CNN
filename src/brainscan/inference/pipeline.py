"""Uncertainty-aware safe inference pipeline for BrainScanAI."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch

from brainscan.core.config import load_train_config, resolve_project_path
from brainscan.data import (
    CANONICAL_CLASS_NAMES,
    CANONICAL_CLASS_TO_ID,
    ID_TO_CANONICAL_CLASS,
    BrainMRIDataset,
    build_eval_transform,
)
from brainscan.evaluation.classification import verify_checkpoint_metadata
from brainscan.models import ArchitectureFeatureExtractor, build_classifier
from brainscan.robustness import (
    compute_min_diagonal_mahalanobis_scores,
    compute_quality_metrics,
    decide_abstention,
    detect_quality_flags,
    load_ood_reference,
    validate_image_file,
)
from brainscan.training import load_checkpoint, load_dataset_fingerprint, resolve_device
from brainscan.uncertainty import load_temperature_artifact
from brainscan.uncertainty.metrics import (
    assign_uncertainty_levels,
    predictive_entropy,
    softmax_probabilities_from_logits,
    top1_top2_margin,
)


DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/resnet18_baseline_best.pt")
DEFAULT_OOD_REFERENCE_NPZ_PATH = Path("artifacts/robustness/resnet18_baseline/ood_reference.npz")
DEFAULT_OOD_REFERENCE_JSON_PATH = Path("artifacts/robustness/resnet18_baseline/ood_reference.json")
DEFAULT_QUALITY_THRESHOLDS_PATH = Path("artifacts/robustness/resnet18_baseline/quality_thresholds.json")
DEFAULT_UNCERTAINTY_METRICS_PATH = Path("artifacts/calibration/resnet18_baseline/metrics.json")

@dataclass(frozen=True)
class PredictionResult:
    class_name: str
    class_id: int
    raw_confidence: float


@dataclass(frozen=True)
class UncertaintyResult:
    entropy: float
    normalized_entropy: float
    margin: float
    level: str


@dataclass(frozen=True)
class QualityResult:
    brightness: float
    contrast: float
    blur_score: float
    near_black_ratio: float
    near_white_ratio: float
    dynamic_range: float
    flags: list[str]


@dataclass(frozen=True)
class OODResult:
    score: float
    warning_threshold: float
    threshold: float
    status: str


@dataclass(frozen=True)
class DecisionResult:
    status: str
    reasons: list[str]


@dataclass(frozen=True)
class SafeInferenceResult:
    status: str
    prediction: PredictionResult | None
    analysis_prediction: PredictionResult | None
    uncertainty: UncertaintyResult | None
    quality: QualityResult | None
    ood: OODResult | None
    decision: DecisionResult

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "prediction": asdict(self.prediction) if self.prediction is not None else None,
            "uncertainty": asdict(self.uncertainty) if self.uncertainty is not None else None,
            "quality": asdict(self.quality) if self.quality is not None else None,
            "ood": asdict(self.ood) if self.ood is not None else None,
            "decision": asdict(self.decision),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BrainScanInferencePipeline:
    """Run safe frozen-model inference with explicit abstention logic."""

    def __init__(
        self,
        *,
        config_path: str | Path = "configs/train.yaml",
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        ood_reference_npz_path: str | Path = DEFAULT_OOD_REFERENCE_NPZ_PATH,
        ood_reference_json_path: str | Path = DEFAULT_OOD_REFERENCE_JSON_PATH,
        quality_thresholds_path: str | Path = DEFAULT_QUALITY_THRESHOLDS_PATH,
        uncertainty_metrics_path: str | Path = DEFAULT_UNCERTAINTY_METRICS_PATH,
        temperature_artifact_path: str | Path | None = None,
        prefer_cuda: bool = True,
    ) -> None:
        self.config = load_train_config(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.ood_reference_npz_path = Path(ood_reference_npz_path)
        self.ood_reference_json_path = Path(ood_reference_json_path)
        self.quality_thresholds_path = Path(quality_thresholds_path)
        self.uncertainty_metrics_path = Path(uncertainty_metrics_path)
        self.temperature_artifact_path = (
            Path(temperature_artifact_path)
            if temperature_artifact_path is not None
            else self.uncertainty_metrics_path.with_name("temperature.json")
        )

        self.dataset_fingerprint = str(load_dataset_fingerprint()["dataset_fingerprint"])
        self.uncertainty_metrics = self._load_json(self.uncertainty_metrics_path)
        self.uncertainty_thresholds = dict(self.uncertainty_metrics["uncertainty_thresholds"])
        self.default_probability_mode = str(self.uncertainty_metrics.get("default_probability_mode", "raw"))
        self.quality_payload = self._load_json(self.quality_thresholds_path)
        self.quality_thresholds = dict(self.quality_payload["thresholds"])
        self.ood_reference, self.ood_metadata = load_ood_reference(
            self.ood_reference_npz_path,
            self.ood_reference_json_path,
        )
        self.temperature_artifact = None
        self.temperature = None
        if self.default_probability_mode == "calibrated":
            self.temperature_artifact = load_temperature_artifact(self.temperature_artifact_path)
            self.temperature = float(self.temperature_artifact["temperature"])

        self.active_architecture = str(self.config["model"]["architecture"])
        self.model = build_classifier(
            architecture=self.active_architecture,
            num_classes=int(self.config["model"]["num_classes"]),
            pretrained=False,
        )
        self.checkpoint = load_checkpoint(self.checkpoint_path, self.model, map_location="cpu")
        checkpoint_metadata = verify_checkpoint_metadata(
            self.checkpoint,
            expected_dataset_fingerprint=self.dataset_fingerprint,
            expected_architecture=self.active_architecture,
            expected_epoch=int(self.checkpoint["epoch"]),
            expected_image_size=list(self.config["training"]["image_size"]),
        )
        self.device, self.device_info = resolve_device(prefer_cuda=prefer_cuda)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.feature_extractor = ArchitectureFeatureExtractor(self.model, self.active_architecture).to(self.device)
        self.feature_extractor.eval()
        self.transform = build_eval_transform(self.config["training"]["image_size"])
        self._verify_artifact_consistency(checkpoint_metadata)

    @staticmethod
    def _load_json(path_value: str | Path) -> dict[str, object]:
        resolved_path = resolve_project_path(path_value)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Required inference artifact not found: {resolved_path}")
        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Inference artifact must contain a JSON object: {resolved_path}")
        return payload

    def _verify_artifact_consistency(self, checkpoint_metadata: dict[str, object]) -> None:
        expected_architecture = str(checkpoint_metadata["architecture"])
        expected_checkpoint = self._to_posix_relative(self.checkpoint_path)
        expected_epoch = int(self.checkpoint["epoch"])
        expected_mapping = dict(CANONICAL_CLASS_TO_ID)
        expected_class_order = list(CANONICAL_CLASS_NAMES)

        if str(self.ood_metadata["dataset_fingerprint"]) != self.dataset_fingerprint:
            raise ValueError("OOD reference dataset fingerprint does not match the frozen checkpoint.")
        if str(self.quality_payload["dataset_fingerprint"]) != self.dataset_fingerprint:
            raise ValueError("Quality-threshold dataset fingerprint does not match the frozen checkpoint.")
        if str(self.uncertainty_metrics["dataset_fingerprint"]) != self.dataset_fingerprint:
            raise ValueError("Uncertainty artifact dataset fingerprint does not match the frozen checkpoint.")

        if str(self.ood_metadata["architecture"]) != expected_architecture:
            raise ValueError("OOD reference architecture does not match the frozen checkpoint.")
        if str(self.quality_payload["architecture"]) != expected_architecture:
            raise ValueError("Quality-threshold architecture does not match the frozen checkpoint.")
        if str(self.uncertainty_metrics["architecture"]) != expected_architecture:
            raise ValueError("Uncertainty artifact architecture does not match the frozen checkpoint.")

        if dict(self.ood_metadata["class_mapping"]) != expected_mapping:
            raise ValueError("OOD reference class mapping does not match canonical class constants.")
        if list(self.ood_metadata["class_order"]) != expected_class_order:
            raise ValueError("OOD reference class order does not match canonical class constants.")

        if str(self.ood_metadata["checkpoint"]) != expected_checkpoint:
            raise ValueError("OOD reference checkpoint path does not match the frozen checkpoint.")
        if str(self.quality_payload["checkpoint"]) != expected_checkpoint:
            raise ValueError("Quality-threshold checkpoint path does not match the frozen checkpoint.")
        if str(self.uncertainty_metrics["checkpoint"]) != expected_checkpoint:
            raise ValueError("Uncertainty artifact checkpoint path does not match the frozen checkpoint.")

        if int(self.ood_metadata["checkpoint_epoch"]) != expected_epoch:
            raise ValueError("OOD reference checkpoint epoch does not match the frozen checkpoint.")
        if int(self.quality_payload["checkpoint_epoch"]) != expected_epoch:
            raise ValueError("Quality-threshold checkpoint epoch does not match the frozen checkpoint.")
        if int(self.uncertainty_metrics["checkpoint_epoch"]) != expected_epoch:
            raise ValueError("Uncertainty artifact checkpoint epoch does not match the frozen checkpoint.")
        if self.temperature_artifact is not None:
            if str(self.temperature_artifact["dataset_fingerprint"]) != self.dataset_fingerprint:
                raise ValueError("Temperature artifact dataset fingerprint does not match the frozen checkpoint.")
            if str(self.temperature_artifact["checkpoint"]) != expected_checkpoint:
                raise ValueError("Temperature artifact checkpoint path does not match the frozen checkpoint.")
            if int(self.temperature_artifact["checkpoint_epoch"]) != expected_epoch:
                raise ValueError("Temperature artifact checkpoint epoch does not match the frozen checkpoint.")
            if str(self.temperature_artifact.get("architecture", expected_architecture)) != expected_architecture:
                raise ValueError("Temperature artifact architecture does not match the frozen checkpoint.")

        if str(self.ood_metadata["feature_layer"]) != self.feature_extractor.feature_layer_name:
            raise ValueError("OOD reference feature layer does not match the active feature extractor.")
        if int(self.ood_metadata["feature_dimension"]) != self.feature_extractor.feature_dimension:
            raise ValueError("OOD reference feature dimension does not match the active feature extractor.")

        if int(self.quality_payload["derivation_split_counts"]["validation"]) != 703:
            raise ValueError("Quality thresholds were not derived from the expected validation split size.")
        if int(self.uncertainty_metrics["validation_sample_count"]) != 703:
            raise ValueError("Uncertainty thresholds were not derived from the expected validation split size.")

        threshold_derivation = str(self.ood_metadata["threshold_derivation"]).lower()
        warning_derivation = str(
            self.ood_metadata.get("warning_threshold_derivation", self.ood_metadata["threshold_derivation"])
        ).lower()
        if "validation" not in threshold_derivation or "test" in threshold_derivation:
            raise ValueError("OOD fail threshold derivation must use validation only.")
        if "validation" not in warning_derivation or "test" in warning_derivation:
            raise ValueError("OOD warning threshold derivation must use validation only.")

    @staticmethod
    def _to_posix_relative(path_value: str | Path) -> str:
        return str(Path(path_value)).replace("\\", "/")

    def _early_abstain(
        self,
        *,
        reasons: list[str],
        quality: QualityResult | None = None,
    ) -> SafeInferenceResult:
        return SafeInferenceResult(
            status="ABSTAIN",
            prediction=None,
            analysis_prediction=None,
            uncertainty=None,
            quality=quality,
            ood=None,
            decision=DecisionResult(status="ABSTAIN", reasons=reasons),
        )

    def _ood_status(self, score: float) -> str:
        if score > float(self.ood_reference.threshold):
            return "FAIL"
        if score > float(self.ood_reference.warning_threshold):
            return "BORDERLINE"
        return "PASS"

    def predict(self, image_path: str | Path) -> SafeInferenceResult:
        self.model.eval()
        self.feature_extractor.eval()

        validation = validate_image_file(image_path)
        if not validation.valid or validation.image is None:
            return self._early_abstain(reasons=list(validation.reasons))

        quality_metrics = compute_quality_metrics(validation.image)
        quality_flags = detect_quality_flags(quality_metrics, self.quality_thresholds)
        quality_result = QualityResult(
            brightness=quality_metrics.brightness_mean,
            contrast=quality_metrics.contrast_std,
            blur_score=quality_metrics.blur_laplacian_variance,
            near_black_ratio=quality_metrics.near_black_ratio,
            near_white_ratio=quality_metrics.near_white_ratio,
            dynamic_range=quality_metrics.dynamic_range,
            flags=list(quality_flags),
        )
        if "LOW_INFORMATION_IMAGE" in quality_flags:
            return self._early_abstain(reasons=["LOW_INFORMATION_IMAGE"], quality=quality_result)

        input_tensor = self.transform(validation.image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(input_tensor)
            features = self.feature_extractor(input_tensor)

        raw_probabilities = softmax_probabilities_from_logits(logits)
        uncertainty_probabilities = raw_probabilities
        if self.default_probability_mode == "calibrated":
            assert self.temperature is not None
            uncertainty_probabilities = softmax_probabilities_from_logits(logits / self.temperature)

        predicted_class_id = int(raw_probabilities.argmax(dim=1).item())
        raw_confidence = float(raw_probabilities.max(dim=1).values.item())
        entropy = float(predictive_entropy(uncertainty_probabilities).item())
        normalized_entropy = float(predictive_entropy(uncertainty_probabilities, normalize=True).item())
        margin = float(top1_top2_margin(uncertainty_probabilities).item())
        uncertainty_level = assign_uncertainty_levels(
            [normalized_entropy],
            [margin],
            self.uncertainty_thresholds,
        )[0]

        feature_np = features.detach().cpu().numpy().astype(np.float64)
        ood_score = float(
            compute_min_diagonal_mahalanobis_scores(
                feature_np,
                self.ood_reference.class_centroids,
                self.ood_reference.diagonal_variance,
            )[0]
        )
        ood_status = self._ood_status(ood_score)

        prediction = PredictionResult(
            class_name=ID_TO_CANONICAL_CLASS[predicted_class_id],
            class_id=predicted_class_id,
            raw_confidence=raw_confidence,
        )
        uncertainty = UncertaintyResult(
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            margin=margin,
            level=uncertainty_level,
        )
        ood = OODResult(
            score=ood_score,
            warning_threshold=float(self.ood_reference.warning_threshold),
            threshold=float(self.ood_reference.threshold),
            status=ood_status,
        )
        abstention = decide_abstention(
            validation_reasons=[],
            quality_flags=quality_result.flags,
            ood_status=ood.status,
            uncertainty_level=uncertainty.level,
        )
        public_prediction = prediction if abstention.status != "ABSTAIN" else None
        return SafeInferenceResult(
            status=abstention.status,
            prediction=public_prediction,
            analysis_prediction=prediction,
            uncertainty=uncertainty,
            quality=quality_result,
            ood=ood,
            decision=DecisionResult(status=abstention.status, reasons=abstention.reasons),
        )

    def run_manifest(self, manifest_path: str | Path) -> list[SafeInferenceResult]:
        dataset = BrainMRIDataset(
            manifest_path=manifest_path,
            transform=self.transform,
            return_metadata=True,
        )
        results: list[SafeInferenceResult] = []
        for record in dataset.records:
            results.append(self.predict(record.relative_path))
        return results

    def summarize_manifest(self, manifest_path: str | Path) -> dict[str, object]:
        results = self.run_manifest(manifest_path)
        counts = Counter(result.status for result in results)
        total = len(results)
        return {
            "total": total,
            "counts": {
                "ACCEPT": counts.get("ACCEPT", 0),
                "REVIEW": counts.get("REVIEW", 0),
                "ABSTAIN": counts.get("ABSTAIN", 0),
            },
            "rates": {
                "ACCEPT": counts.get("ACCEPT", 0) / total if total else 0.0,
                "REVIEW": counts.get("REVIEW", 0) / total if total else 0.0,
                "ABSTAIN": counts.get("ABSTAIN", 0) / total if total else 0.0,
            },
        }
