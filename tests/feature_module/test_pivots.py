"""Tests for ``cfd10.feature_module.pivots``.

These tests pin the Pine semantics of ``speculatores_v15_presets_gold.pine``:
``ta.pivothigh`` / ``ta.pivotlow`` (L491-492, called with symmetric
``baseline_lb``) and ``calc_pivot_drift`` (L142-151). They cover a hand-built
reference fixture, bounded-range / finiteness guarantees, and the warm-up
(insufficient-history) NaN / no-pivot contract.
"""

from __future__ import annotations

import math

import numpy as np

from cfd10.feature_module.pivots import pivot_drift, pivot_high, pivot_low


def _reference_pivot_high(high: np.ndarray, lb: int) -> np.ndarray:
    """Independent O(n*lb) reference for the strict-max pivot-high rule."""
    n = high.shape[0]
    out = np.zeros(n, dtype=bool)
    for i in range(lb, n - lb):
        center = high[i]
        is_pivot = True
        for j in range(i - lb, i + lb + 1):
            if j == i:
                continue
            if high[j] >= center:
                is_pivot = False
                break
        out[i] = is_pivot
    return out


def _reference_pivot_low(low: np.ndarray, lb: int) -> np.ndarray:
    """Independent O(n*lb) reference for the strict-min pivot-low rule."""
    n = low.shape[0]
    out = np.zeros(n, dtype=bool)
    for i in range(lb, n - lb):
        center = low[i]
        is_pivot = True
        for j in range(i - lb, i + lb + 1):
            if j == i:
                continue
            if low[j] <= center:
                is_pivot = False
                break
        out[i] = is_pivot
    return out


def test_pivot_high_reference_fixture() -> None:
    """Hand-built fixture: only the strict-max bar inside its window is a pivot."""
    # lb = 1: each interior bar compares with its two neighbours.
    #             idx: 0    1    2    3    4    5    6
    high = np.array([1.0, 3.0, 2.0, 2.0, 5.0, 4.0, 1.0], dtype=np.float64)
    out = pivot_high(high, lb=1)
    # idx1 (3 > 1 and 3 > 2) -> pivot. idx4 (5 > 2 and 5 > 4) -> pivot.
    # idx2 not > idx3 (2 == 2) -> NOT strict -> False. idx5 (4 > 5? no) -> False.
    # idx0 and idx6 are warm-up edges -> False.
    expected = np.array(
        [False, True, False, False, True, False, False], dtype=bool
    )
    np.testing.assert_array_equal(out, expected)


def test_pivot_high_strict_max_rejects_ties() -> None:
    """A plateau (tie at the window edge) is not a strict pivot high."""
    high = np.array([1.0, 4.0, 4.0, 1.0, 0.0], dtype=np.float64)
    out = pivot_high(high, lb=1)
    # idx1: high[1]=4 is NOT strictly greater than high[2]=4 -> False.
    # idx2: high[2]=4 is NOT strictly greater than high[1]=4 -> False.
    assert not out[1]
    assert not out[2]


def test_pivot_low_reference_fixture() -> None:
    """Hand-built fixture: only the strict-min bar inside its window is a pivot."""
    #            idx: 0    1    2    3    4    5    6
    low = np.array([5.0, 2.0, 3.0, 3.0, 1.0, 4.0, 6.0], dtype=np.float64)
    out = pivot_low(low, lb=1)
    # idx1 (2 < 5 and 2 < 3) -> pivot. idx4 (1 < 3 and 1 < 4) -> pivot.
    expected = np.array(
        [False, True, False, False, True, False, False], dtype=bool
    )
    np.testing.assert_array_equal(out, expected)


def test_pivot_high_matches_independent_reference_random() -> None:
    """Vectorized pivot_high matches the naive O(n*lb) reference on random data."""
    rng = np.random.default_rng(13)
    high = rng.normal(size=400).astype(np.float64)
    for lb in (1, 2, 5, 20):
        np.testing.assert_array_equal(
            pivot_high(high, lb), _reference_pivot_high(high, lb)
        )


def test_pivot_low_matches_independent_reference_random() -> None:
    """Vectorized pivot_low matches the naive O(n*lb) reference on random data."""
    rng = np.random.default_rng(29)
    low = rng.normal(size=400).astype(np.float64)
    for lb in (1, 2, 5, 20):
        np.testing.assert_array_equal(
            pivot_low(low, lb), _reference_pivot_low(low, lb)
        )


def test_pivot_high_output_is_bool_and_aligned() -> None:
    """Output is a bool ndarray aligned to the input length."""
    high = np.random.default_rng(0).normal(size=128).astype(np.float64)
    out = pivot_high(high, lb=3)
    assert out.dtype == np.bool_
    assert out.shape == high.shape


def test_pivot_warmup_edges_are_false() -> None:
    """The first/last ``lb`` bars can never be confirmed pivots (warm-up)."""
    rng = np.random.default_rng(5)
    high = rng.normal(size=60).astype(np.float64)
    lb = 7
    out = pivot_high(high, lb)
    assert not out[:lb].any()
    assert not out[len(out) - lb :].any()


def test_pivot_drift_reference_value() -> None:
    """``pivot_drift`` matches the explicit Pine L150 formula on a fixture.

    With lookback=3, min_pivots = max(3, 2) = 3, pivot_count = 2.
    start_val = pivots[-3] = 10.0, end_val = pivots[-1] = 14.0.
    drift = ((14 - 10) / max(|10|, 1e-9)) / 2 = (0.4) / 2 = 0.2.
    """
    pivots = np.array([8.0, 10.0, 12.0, 14.0], dtype=np.float64)
    drift = pivot_drift(pivots, lookback=3)
    assert math.isclose(drift, 0.2, abs_tol=1e-12)


def test_pivot_drift_lookback_below_two_floors_to_two() -> None:
    """``min_pivots`` floors at 2, so pivot_count is at least 1 (Pine L144,149)."""
    # lookback=1 -> min_pivots=2, pivot_count=1, uses the last two pivots.
    pivots = np.array([5.0, 4.0, 6.0], dtype=np.float64)
    drift = pivot_drift(pivots, lookback=1)
    # start_val = pivots[-2] = 4.0, end_val = 6.0 -> (6-4)/4 / 1 = 0.5.
    assert math.isclose(drift, 0.5, abs_tol=1e-12)


def test_pivot_drift_warmup_insufficient_pivots_is_nan() -> None:
    """Fewer than ``min_pivots`` confirmed pivots yields NaN (Pine ``na``)."""
    # lookback=5 -> min_pivots=5; only 4 pivots present.
    pivots = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert math.isnan(pivot_drift(pivots, lookback=5))
    # Empty array is also warm-up.
    assert math.isnan(pivot_drift(np.array([], dtype=np.float64), lookback=2))


def test_pivot_drift_zero_start_uses_epsilon_floor() -> None:
    """A zero ``start_val`` divides by the 1e-9 floor, not zero (Pine L150)."""
    pivots = np.array([0.0, 1e-3], dtype=np.float64)
    drift = pivot_drift(pivots, lookback=2)
    # (1e-3 - 0) / max(0, 1e-9) / 1 = 1e-3 / 1e-9 = 1e6.
    assert math.isclose(drift, 1e6, rel_tol=1e-9)


def test_pivot_drift_is_finite_on_real_pivots() -> None:
    """Drift is finite for any sufficiently long, strictly positive pivot series."""
    rng = np.random.default_rng(99)
    pivots = (100.0 + np.cumsum(rng.normal(size=50))).astype(np.float64)
    for lookback in (2, 3, 5, 10):
        drift = pivot_drift(pivots, lookback)
        assert np.isfinite(drift)
