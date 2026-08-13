from __future__ import annotations

from brainscan.robustness.abstention import decide_abstention


def test_accept_logic() -> None:
    decision = decide_abstention(
        validation_reasons=[],
        quality_flags=[],
        ood_status="PASS",
        uncertainty_level="LOW",
    )
    assert decision.status == "ACCEPT"
    assert decision.reasons == []


def test_review_logic() -> None:
    decision = decide_abstention(
        validation_reasons=[],
        quality_flags=["SEVERE_BLUR"],
        ood_status="BORDERLINE",
        uncertainty_level="HIGH",
    )
    assert decision.status == "REVIEW"
    assert "SEVERE_BLUR" in decision.reasons
    assert "BORDERLINE_OOD" in decision.reasons
    assert "HIGH_UNCERTAINTY" in decision.reasons


def test_abstain_logic() -> None:
    decision = decide_abstention(
        validation_reasons=[],
        quality_flags=["LOW_INFORMATION_IMAGE"],
        ood_status="FAIL",
        uncertainty_level="HIGH",
    )
    assert decision.status == "ABSTAIN"
    assert "LOW_INFORMATION_IMAGE" in decision.reasons
    assert "OUT_OF_DISTRIBUTION" in decision.reasons


def test_validation_reasons_take_priority() -> None:
    decision = decide_abstention(
        validation_reasons=["IMAGE_DECODE_FAILED"],
        quality_flags=["SEVERE_BLUR"],
        ood_status="PASS",
        uncertainty_level="LOW",
    )
    assert decision.status == "ABSTAIN"
    assert decision.reasons == ["IMAGE_DECODE_FAILED"]
