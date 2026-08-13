"""Explicit ACCEPT / REVIEW / ABSTAIN rules for BrainScanAI safe inference."""

from __future__ import annotations

from dataclasses import dataclass


QUALITY_WARNING_FLAGS = {
    "SEVERE_BLUR",
    "VERY_DARK_IMAGE",
    "VERY_BRIGHT_IMAGE",
    "LOW_CONTRAST_IMAGE",
    "LOW_DYNAMIC_RANGE",
    "NEAR_BLACK_DOMINANT",
    "NEAR_WHITE_DOMINANT",
}

QUALITY_HARD_FAIL_FLAGS = {
    "LOW_INFORMATION_IMAGE",
}


@dataclass(frozen=True)
class AbstentionDecision:
    status: str
    reasons: list[str]


def decide_abstention(
    *,
    validation_reasons: list[str],
    quality_flags: list[str],
    ood_status: str | None,
    uncertainty_level: str | None,
) -> AbstentionDecision:
    """Apply transparent rule-based decision logic to one inference result."""
    reasons: list[str] = []

    if validation_reasons:
        return AbstentionDecision(status="ABSTAIN", reasons=list(validation_reasons))

    hard_quality = sorted(flag for flag in quality_flags if flag in QUALITY_HARD_FAIL_FLAGS)
    warning_quality = sorted(flag for flag in quality_flags if flag in QUALITY_WARNING_FLAGS)

    if hard_quality:
        reasons.extend(hard_quality)
    if ood_status == "FAIL":
        reasons.append("OUT_OF_DISTRIBUTION")

    if hard_quality or ood_status == "FAIL":
        return AbstentionDecision(status="ABSTAIN", reasons=reasons)

    if ood_status == "BORDERLINE":
        reasons.append("BORDERLINE_OOD")
    reasons.extend(warning_quality)

    if uncertainty_level == "HIGH":
        reasons.append("HIGH_UNCERTAINTY")
    elif uncertainty_level == "MEDIUM":
        reasons.append("MEDIUM_UNCERTAINTY")

    deduplicated: list[str] = []
    for reason in reasons:
        if reason not in deduplicated:
            deduplicated.append(reason)

    if deduplicated:
        return AbstentionDecision(status="REVIEW", reasons=deduplicated)

    return AbstentionDecision(status="ACCEPT", reasons=[])
