from pathlib import Path

from PIL import Image
import torch

from brainscan.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)


def test_train_transform_returns_expected_dimensions():
    image = Image.new("RGB", (40, 50), color=(100, 120, 140))
    transform = build_train_transform((224, 224))

    tensor = transform(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transform_is_deterministic():
    image = Image.new("RGB", (40, 50), color=(100, 120, 140))
    transform = build_eval_transform((224, 224))

    tensor_a = transform(image)
    tensor_b = transform(image)

    assert torch.equal(tensor_a, tensor_b)


def test_normalization_constants_match_imagenet():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)
