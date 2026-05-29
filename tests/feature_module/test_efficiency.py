"""Tests for ``cfd10.feature_module.efficiency`` — Kaufman Efficiency Ratio.

These tests pin the Pine semantics of ``speculatores_v15_presets_gold.pine``
lines 468-480 (``er_path`` / ``er_net`` / ``er_val``) and verify that the scalar
reference implementation agrees with the vectorized/Numba full-series
implementation to 1e-9.
"""

from __future__ import annotations

import math

import numpy as np

from cfd10.feature_module.efficiency import (
    efficiency_ratio,
    efficiency_ratio_scalar,
)


def _random_close(n: int = 500, seed: int = 11) -> np.ndarray:
    """Return a strictly positive random close series (geometric random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.01, size=n)
    return (100.0 * np.exp(np.cumsum(steps))).astype(np.float64)


def test_reference_values_directional_hand_fixture() -> None:
    """Directional ER matches hand-computed Pine values (L468-472).

    Pine at bar ``t``: ``er_path = sum_{i=0}^{p-1} |close[t-i] - close[t-i-1]|``,
    ``er_net = close[t] - close[t-p]``, ``er_val = er_net / er_path``.
    """
    close = np.array([10.0, 11.0, 13.0, 12.0, 15.0], dtype=np.float64)
    out = efficiency_ratio(close, period=2, directional=True)

    # Warm-up: bars 0..p-1 have insufficient history -> NaN.
    assert math.isnan(out[0])
    assert math.isnan(out[1])
    # t=2: path=|13-11|+|11-10|=3 ; net=13-10=3 ; 3/3 = 1.0
    assert math.isclose(out[2], 1.0, abs_tol=1e-12)
    # t=3: path=|12-13|+|13-11|=3 ; net=12-11=1 ; 1/3
    assert math.isclose(out[3], 1.0 / 3.0, abs_tol=1e-12)
    # t=4: path=|15-12|+|12-13|=4 ; net=15-13=2 ; 2/4 = 0.5
    assert math.isclose(out[4], 0.5, abs_tol=1e-12)


def test_reference_values_absolute_hand_fixture() -> None:
    """Absolute ER uses ``abs(close[t] - close[t-p])`` for the numerator (L478)."""
    # Down-then-up series so the directional net is negative on the last bar but
    # the absolute net is positive: distinguishes the two modes.
    close = np.array([20.0, 18.0, 14.0, 16.0], dtype=np.float64)
    out_dir = efficiency_ratio(close, period=3, directional=True)
    out_abs = efficiency_ratio(close, period=3, directional=False)

    # t=3: path=|16-14|+|14-18|+|18-20| = 2+4+2 = 8
    #      net_dir = 16-20 = -4 -> -4/8 = -0.5
    #      net_abs = |16-20| = 4 ->  4/8 =  0.5
    assert math.isclose(out_dir[3], -0.5, abs_tol=1e-12)
    assert math.isclose(out_abs[3], 0.5, abs_tol=1e-12)
    # Absolute ER is the magnitude of the directional ER here.
    assert math.isclose(out_abs[3], abs(out_dir[3]), abs_tol=1e-12)


def test_warmup_bars_are_nan() -> None:
    """The first ``period`` bars lack history and are NaN; bar ``period`` is finite."""
    close = _random_close(n=50)
    for period in (1, 3, 7, 14):
        out = efficiency_ratio(close, period=period, directional=True)
        assert np.all(np.isnan(out[:period])), f"warm-up not NaN for period={period}"
        assert np.isfinite(out[period]), f"first valid bar NaN for period={period}"
        assert np.all(np.isfinite(out[period:])), f"finite tail broken (period={period})"


def test_directional_range_within_unit_interval() -> None:
    """Directional ER lies in [-1, 1] on all valid bars."""
    close = _random_close(n=400)
    for period in (2, 5, 10, 30):
        out = efficiency_ratio(close, period=period, directional=True)
        valid = out[~np.isnan(out)]
        assert np.all(valid >= -1.0 - 1e-12)
        assert np.all(valid <= 1.0 + 1e-12)


def test_absolute_range_within_unit_interval() -> None:
    """Absolute ER lies in [0, 1] on all valid bars."""
    close = _random_close(n=400, seed=99)
    for period in (2, 5, 10, 30):
        out = efficiency_ratio(close, period=period, directional=False)
        valid = out[~np.isnan(out)]
        assert np.all(valid >= 0.0 - 1e-12)
        assert np.all(valid <= 1.0 + 1e-12)


def test_zero_path_yields_zero_not_nan() -> None:
    """A flat window has ``er_path == 0`` -> ``er_val = 0.0`` (Pine L472/L479)."""
    close = np.full(10, 42.0, dtype=np.float64)
    out_dir = efficiency_ratio(close, period=4, directional=True)
    out_abs = efficiency_ratio(close, period=4, directional=False)
    # Warm-up still NaN; valid bars are exactly 0.0 (not NaN), since path == 0.
    assert np.all(np.isnan(out_dir[:4]))
    assert np.all(out_dir[4:] == 0.0)
    assert np.all(out_abs[4:] == 0.0)


def test_monotone_trend_directional_er_is_one() -> None:
    """A strictly monotone run has net == path, so directional ER == 1.0."""
    close = np.arange(1.0, 21.0, dtype=np.float64)  # +1 each bar
    out = efficiency_ratio(close, period=5, directional=True)
    np.testing.assert_allclose(out[5:], 1.0, atol=1e-12)


def test_scalar_and_vectorized_agree() -> None:
    """Scalar per-bar and vectorized full-series ER agree to 1e-9."""
    close = _random_close(n=500, seed=3)
    for directional in (True, False):
        for period in (1, 2, 7, 13, 50):
            vec = efficiency_ratio(close, period=period, directional=directional)
            scalar = np.array(
                [
                    efficiency_ratio_scalar(close, period, directional, i)
                    for i in range(len(close))
                ],
                dtype=np.float64,
            )
            # NaNs must align; finite entries must match to 1e-9.
            assert np.array_equal(np.isnan(vec), np.isnan(scalar))
            mask = ~np.isnan(vec)
            np.testing.assert_allclose(vec[mask], scalar[mask], atol=1e-9, rtol=0.0)


def test_invalid_period_raises() -> None:
    """Non-positive ``period`` is rejected with ``ValueError``."""
    close = _random_close(n=20)
    for bad in (0, -1, -5):
        try:
            efficiency_ratio(close, period=bad, directional=True)
        except ValueError:
            continue
        raise AssertionError(f"period={bad} should have raised ValueError")
