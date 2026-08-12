"""Reproducibility helpers for BrainScanAI."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch generator for reproducible ordering."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process deterministically."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
