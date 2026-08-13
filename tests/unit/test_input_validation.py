from __future__ import annotations

from pathlib import Path

from PIL import Image

from brainscan.robustness.input_validation import validate_image_file


def test_valid_image_is_accepted_by_decoder(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 16), color=(120, 130, 140)).save(image_path)

    result = validate_image_file(image_path)

    assert result.valid is True
    assert result.image is not None
    assert result.reasons == []


def test_corrupted_image_is_rejected(tmp_path) -> None:
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not-a-real-image")

    result = validate_image_file(image_path)

    assert result.valid is False
    assert "IMAGE_DECODE_FAILED" in result.reasons


def test_unsupported_extension_is_rejected(tmp_path) -> None:
    image_path = tmp_path / "sample.gif"
    image_path.write_bytes(b"GIF89a")

    result = validate_image_file(image_path)

    assert result.valid is False
    assert result.reasons == ["UNSUPPORTED_FORMAT"]
