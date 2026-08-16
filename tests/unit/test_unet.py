from __future__ import annotations

import torch

from brainscan.segmentation.models import UNet2D


def test_unet_outputs_binary_logit_map_shape() -> None:
    model = UNet2D(input_channels=4, output_channels=1, base_channels=16)
    outputs = model(torch.randn(2, 4, 240, 240))
    assert tuple(outputs.shape) == (2, 1, 240, 240)


def test_unet_handles_odd_and_even_spatial_shapes() -> None:
    model = UNet2D(input_channels=4, output_channels=1, base_channels=16)
    outputs = model(torch.randn(1, 4, 241, 239))
    assert tuple(outputs.shape) == (1, 1, 241, 239)
