"""Tests for reproducibility utilities."""

import numpy as np

from cfd10.utils import record_env, set_seed


def test_seed_determinism() -> None:
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_record_env_has_python() -> None:
    info = record_env()
    assert "python_version" in info
