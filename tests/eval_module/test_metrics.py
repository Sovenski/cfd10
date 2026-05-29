"""Tests for ``cfd10.eval_module.metrics`` — event precision / recall / F1.

A hand-worked confusion case pins the exact F1, plus boundary checks for the
``zero_division=0`` convention and the relationship between the count-based and
index-based entry points.
"""

from __future__ import annotations

import math

import numpy as np

from cfd10.eval_module.metrics import event_prf, prf_from_counts


def test_hand_case_precise_f1() -> None:
    """A concrete 5-pred / 4-true case with a known precision/recall/F1.

    Truths:      [10, 20, 30, 40]
    Predictions: [11, 19, 100, 200, 41]   (tolerance = 2)

      11 <-> 10 (delta 1)   TP
      19 <-> 20 (delta 1)   TP
      41 <-> 40 (delta 1)   TP
      100, 200              far from every truth -> FP, FP
      30                    never predicted      -> FN

    So tp=3, fp=2, fn=1:
      precision = 3 / 5 = 0.6
      recall    = 3 / 4 = 0.75
      f1        = 2*0.6*0.75 / (0.6 + 0.75) = 0.9 / 1.35 = 2/3
    """
    true = np.array([10, 20, 30, 40], dtype=np.int64)
    pred = np.array([11, 19, 100, 200, 41], dtype=np.int64)
    precision, recall, f1 = event_prf(pred, true, tolerance=2)
    assert math.isclose(precision, 0.6, abs_tol=1e-12)
    assert math.isclose(recall, 0.75, abs_tol=1e-12)
    assert math.isclose(f1, 2.0 / 3.0, abs_tol=1e-12)


def test_prf_from_counts_matches_definition() -> None:
    """``prf_from_counts`` reproduces the textbook formulas."""
    precision, recall, f1 = prf_from_counts(tp=3, fp=2, fn=1)
    assert math.isclose(precision, 3 / 5, abs_tol=1e-12)
    assert math.isclose(recall, 3 / 4, abs_tol=1e-12)
    assert math.isclose(f1, 2.0 / 3.0, abs_tol=1e-12)


def test_perfect_detection() -> None:
    """Exact alignment gives precision == recall == f1 == 1."""
    idx = np.array([5, 15, 25], dtype=np.int64)
    precision, recall, f1 = event_prf(idx, idx, tolerance=0)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_zero_division_conventions() -> None:
    """Empty denominators report 0.0 rather than NaN."""
    # No predictions: precision denominator zero -> 0; recall denominator zero
    # too (no truths) -> all zero.
    assert prf_from_counts(0, 0, 0) == (0.0, 0.0, 0.0)
    # No true positives but some predictions and truths.
    precision, recall, f1 = prf_from_counts(0, 4, 3)
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)
    # Predictions but no truths -> recall denominator zero.
    empty = np.array([], dtype=np.int64)
    some = np.array([1, 2, 3], dtype=np.int64)
    p2, r2, f2 = event_prf(some, empty, tolerance=2)
    assert (p2, r2, f2) == (0.0, 0.0, 0.0)


def test_all_outputs_within_unit_interval_fuzz() -> None:
    """precision, recall, f1 stay in [0, 1] over random inputs."""
    rng = np.random.default_rng(123)
    for _ in range(50):
        pred = rng.integers(0, 150, size=int(rng.integers(0, 10))).astype(np.int64)
        true = rng.integers(0, 150, size=int(rng.integers(0, 10))).astype(np.int64)
        precision, recall, f1 = event_prf(pred, true, tolerance=int(rng.integers(0, 5)))
        for value in (precision, recall, f1):
            assert 0.0 <= value <= 1.0


def test_negative_counts_raise() -> None:
    """Negative counts are rejected."""
    try:
        prf_from_counts(-1, 0, 0)
    except ValueError:
        return
    raise AssertionError("negative counts should have raised ValueError")
