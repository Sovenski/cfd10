"""Tests for ``cfd10.feature_module.sma_pir`` — the parity lynchpin.

These tests pin the Pine semantics of ``speculatores_v15_presets_gold.pine``
lines 68-126 and verify that the scalar reference implementation agrees with the
vectorized/Numba full-series implementation to 1e-9.
"""

from __future__ import annotations

import math

import numpy as np

from cfd10.feature_module import (
    agreement,
    csum_close,
    pir_for_scale,
    pir_for_scale_series,
    pir_of_series,
    sma_at,
)


def _random_close(n: int = 500, seed: int = 7) -> np.ndarray:
    """Return a strictly positive random close series (geometric random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.01, size=n)
    return (100.0 * np.exp(np.cumsum(steps))).astype(np.float64)


def test_csum_close_is_inclusive_prefix_sum() -> None:
    """``csum[i] = sum(close[0..i])`` exactly (Pine ta.cum, L87)."""
    close = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    csum = csum_close(close)
    np.testing.assert_array_equal(csum, np.array([1.0, 3.0, 6.0, 10.0]))


def test_sma_at_boundary_index_minus_one_is_zero() -> None:
    """``csum`` index ``-1`` is treated as the ``0.0`` boundary (full-window SMA)."""
    close = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    csum = csum_close(close)
    # At i = s - 1 = 1, back = 0: (csum[1] - csum[-1]) / 2 = (6 - 0) / 2 = 3.0.
    assert sma_at(csum, 2, 0, 1) == 3.0
    # Standard interior window: at i = 3, back = 0: (csum[3] - csum[1]) / 2 = 7.0.
    assert sma_at(csum, 2, 0, 3) == 7.0


def test_sma_at_out_of_range_is_nan() -> None:
    """Out-of-range lookups (index < -1) yield NaN (Pine ``na``)."""
    close = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    csum = csum_close(close)
    # i = 0, s = 2: csum[i - s] = csum[-2] -> out of range -> NaN.
    assert math.isnan(sma_at(csum, 2, 0, 0))


def test_pir_for_scale_within_unit_interval() -> None:
    """PIR-for-scale outputs always lie within [0, 1]."""
    close = _random_close()
    csum = csum_close(close)
    for i in range(len(close)):
        for s in (3, 7, 20):
            lb = max(s, 20)
            val = pir_for_scale(close, csum, s, lb, i)
            assert 0.0 <= val <= 1.0


def test_pir_of_series_within_unit_interval_and_flat_is_half() -> None:
    """``pir_of`` is in [0, 1]; a flat trailing window maps to 0.5."""
    arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    out = pir_of_series(arr, lookback=3)
    np.testing.assert_allclose(out, 0.5)

    rng = np.random.default_rng(1)
    noisy = rng.normal(size=200).astype(np.float64)
    pos = pir_of_series(noisy, lookback=10)
    assert np.all(pos >= 0.0) and np.all(pos <= 1.0)


def test_pir_of_series_matches_trailing_window_definition() -> None:
    """``pir_of`` equals the trailing min/max position over ``lookback`` bars."""
    arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float64)
    lookback = 3
    out = pir_of_series(arr, lookback)
    expected = np.empty_like(arr)
    for i in range(len(arr)):
        window = arr[max(0, i - lookback + 1) : i + 1]
        lo, hi = window.min(), window.max()
        expected[i] = (arr[i] - lo) / (hi - lo) if hi != lo else 0.5
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_strictly_increasing_series_late_bar_pir_near_one() -> None:
    """On an accelerating up-trend, late-bar ``pir_for_scale`` is near 1.0.

    ``pir_for_scale`` positions the *ratio* ``close/SMA(s)`` within its lookback
    range, not the raw price. A *linear* ramp makes that ratio mean-revert toward
    1 (the lagging SMA catches up proportionally), so its late-bar PIR is ~0, not
    1. For the PIR to approach 1 the ratio itself must be rising, which requires
    a super-linear (accelerating) up-trend that widens the gap between price and
    its trailing SMA over time.
    """
    k = np.arange(300, dtype=np.float64)
    close = (100.0 * np.exp(0.00015 * k**2)).astype(np.float64)
    assert np.all(np.diff(close) > 0)  # strictly increasing
    csum = csum_close(close)
    s, lb = 5, 30
    val = pir_for_scale(close, csum, s, lb, len(close) - 1)
    assert val > 0.99


def test_scalar_and_vectorized_pir_for_scale_agree() -> None:
    """Scalar per-bar and vectorized full-series PIR-for-scale agree to 1e-9."""
    close = _random_close(n=500, seed=42)
    csum = csum_close(close)
    for s in (2, 5, 13, 50):
        lb = max(s, 20)
        vec = pir_for_scale_series(close, csum, s, lb)
        scalar = np.array(
            [pir_for_scale(close, csum, s, lb, i) for i in range(len(close))],
            dtype=np.float64,
        )
        np.testing.assert_allclose(vec, scalar, atol=1e-9, rtol=0.0)


def test_agreement_matches_manual_loop_tiny_case() -> None:
    """``agreement`` counts match a manual loop over ``pir_for_scale``."""
    # Hand-built tiny case: short scale range, exercised at every bar.
    close = np.array(
        [10.0, 10.5, 9.8, 11.2, 12.0, 11.5, 13.0, 12.7, 14.1, 13.5, 15.0, 16.2],
        dtype=np.float64,
    )
    csum = csum_close(close)
    scale_start, scale_end, scale_step = 2, 6, 2
    pct_extreme = 0.8

    for i in range(len(close)):
        scales_high, scales_low, n, agree_high, agree_low = agreement(
            close, csum, scale_start, scale_end, scale_step, pct_extreme, i
        )
        # Manual reference loop mirroring Pine calc_agreement (L115-126).
        m_high = 0
        m_low = 0
        m_n = 0
        for s in range(scale_start, scale_end + 1, scale_step):
            lb = max(s, 20)
            pir_s = pir_for_scale(close, csum, s, lb, i)
            m_high += 1 if pir_s > pct_extreme else 0
            m_low += 1 if pir_s < (1.0 - pct_extreme) else 0
            m_n += 1
        m_n_f = max(m_n, 1)
        assert scales_high == m_high
        assert scales_low == m_low
        assert n == m_n
        assert math.isclose(agree_high, m_high / m_n_f, abs_tol=1e-12)
        assert math.isclose(agree_low, m_low / m_n_f, abs_tol=1e-12)


def test_agreement_strictly_increasing_all_high() -> None:
    """On an accelerating up-trend, late bars are 'high' on every scale.

    Uses a super-linear ramp for the same reason as
    :func:`test_strictly_increasing_series_late_bar_pir_near_one`: the agreement
    counts are driven by ``pir_for_scale`` of the ``close/SMA`` ratio, which only
    saturates high when that ratio is genuinely rising.
    """
    k = np.arange(120, dtype=np.float64)
    close = (100.0 * np.exp(0.0004 * k**2)).astype(np.float64)
    assert np.all(np.diff(close) > 0)  # strictly increasing
    csum = csum_close(close)
    scales_high, scales_low, n, agree_high, agree_low = agreement(
        close, csum, 2, 10, 2, 0.8, len(close) - 1
    )
    assert n == 5
    assert scales_high == n
    assert scales_low == 0
    assert agree_high == 1.0
    assert agree_low == 0.0
