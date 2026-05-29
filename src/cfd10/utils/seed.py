"""Random-seed control for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used in the project for reproducibility.

    Seeds Python ``random``, NumPy, ``PYTHONHASHSEED`` and (if installed) PyTorch
    including CUDA and deterministic cuDNN. PyTorch is imported lazily so the
    function works in environments without it.

    Args:
        seed: The seed value to apply across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
