"""Input-file and image-decoding validation for robust BrainScanAI inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageValidationResult:
    valid: bool
    image: Image.Image | None
    reasons: list[str]
    width: int
    height: int
    detected_format: str | None


def validate_image_file(image_path: str | Path) -> ImageValidationResult:
    """Validate a candidate image path and decode it into RGB safely."""
    path = Path(image_path)
    reasons: list[str] = []

    if not path.exists():
        return ImageValidationResult(
            valid=False,
            image=None,
            reasons=["INVALID_FILE"],
            width=0,
            height=0,
            detected_format=None,
        )

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return ImageValidationResult(
            valid=False,
            image=None,
            reasons=["UNSUPPORTED_FORMAT"],
            width=0,
            height=0,
            detected_format=None,
        )

    try:
        with Image.open(path) as opened_image:
            detected_format = opened_image.format
            if opened_image.width <= 0 or opened_image.height <= 0:
                reasons.append("INVALID_DIMENSIONS")
                return ImageValidationResult(
                    valid=False,
                    image=None,
                    reasons=reasons,
                    width=max(opened_image.width, 0),
                    height=max(opened_image.height, 0),
                    detected_format=detected_format,
                )

            rgb_image = opened_image.convert("RGB")
    except FileNotFoundError:
        return ImageValidationResult(False, None, ["INVALID_FILE"], 0, 0, None)
    except UnidentifiedImageError:
        return ImageValidationResult(False, None, ["IMAGE_DECODE_FAILED"], 0, 0, None)
    except OSError:
        return ImageValidationResult(False, None, ["IMAGE_DECODE_FAILED"], 0, 0, None)

    pixel_array = np.asarray(rgb_image, dtype=np.float32)
    if pixel_array.ndim != 3 or pixel_array.shape[2] != 3:
        reasons.append("CHANNEL_CONVERSION_FAILED")
    if not np.isfinite(pixel_array).all():
        reasons.append("INVALID_PIXEL_VALUES")

    return ImageValidationResult(
        valid=not reasons,
        image=rgb_image if not reasons else None,
        reasons=reasons,
        width=rgb_image.width,
        height=rgb_image.height,
        detected_format=detected_format,
    )
