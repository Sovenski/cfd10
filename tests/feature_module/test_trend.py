"""Tests for ``cfd10.feature_module.trend`` — per-side trend slope features.

These tests pin the Pine semantics of ``speculatores_v15_presets_gold.pine``
lines 402-413 (the ``slope_val_*`` and ``linreg_norm_*`` computations):

* SMA-slope: ``slope = (sma - sma[d]) / (d * sma) * 1000`` with
  ``d = max(round(S / 4), 2)``.
* Linreg-slope: ``ta.linreg(close, S, 0) - ta.linreg(close, S, 1)`` normalized
  ``/ sma * 1000``. The offset-0-minus-offset-1 difference of a single OLS fit
  equals the regression slope, so we test against an explicit ``numpy`` polyfit.

The Pine ternary yields ``0.0`` during warm-up (``na > 0`` is falsy); the Python
feature layer instead emits ``np.nan`` for bars with insufficient history, which
is the documented divergence verified here.
"""

from __future__ import annotations

import numpy as np

from cfd10.feature_module.trend import (
    linreg_slope_norm,
    sma_slope,
    slope_delta,
)
from cfd10.feature_module.trend import _vectorized_matches_scalar


def _random_close(n: int = 400, seed: int = 11) -> np.ndarray:
    """Return a strictly positive random close series (geometric random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.01, size=n)
    return (100.0 * np.exp(np.cumsum(steps))).astype(np.float64)


def _ref_sma(close: np.ndarray, s: int) -> np.ndarray:
    """Plain trailing-SMA reference: NaN before ``s`` bars, mean of last ``s``."""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(s - 1, n):
        out[i] = close[i - s + 1 : i + 1].mean()
    return out


# --------------------------------------------------------------------------- #
# slope_delta — d = max(round(S / 4), 2), banker's-rounding-agnostic cases.    #
# --------------------------------------------------------------------------- #


def test_slope_delta_matches_pine_formula() -> None:
    """``d = max(round(S / 4), 2)`` matching Pine L288/L295."""
    # S/4 below 2 must floor to the minimum of 2.
    assert slope_delta(4) == 2  # round(1.0)=1 -> max(1, 2)=2
    assert slope_delta(8) == 2  # round(2.0)=2 -> max(2, 2)=2
    assert slope_delta(12) == 3  # round(3.0)=3
    assert slope_delta(40) == 10  # round(10.0)=10
    assert slope_delta(2) == 2  # round(0.5) -> max(.,2)=2
    assert slope_delta(1) == 2  # round(0.25)=0 -> max(0, 2)=2


# --------------------------------------------------------------------------- #
# sma_slope — reference-value test on a hand-built fixture.                    #
# --------------------------------------------------------------------------- #


def test_sma_slope_reference_value_handbuilt() -> None:
    """SMA-slope equals the hand-computed Pine formula on a tiny ramp.

    With ``S = 4`` the slope delta is ``d = max(round(1.0), 2) = 2``. We build a
    short increasing series and verify the first valid bar against an explicit
    arithmetic evaluation of ``(sma - sma[d]) / (d * sma) * 1000``.
    """
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float64)
    s = 4
    d = 2
    out = sma_slope(close, s)

    sma = _ref_sma(close, s)
    # First bar where both sma[i] and sma[i - d] are valid is i = s - 1 + d = 5.
    first_valid = s - 1 + d
    assert np.all(np.isnan(out[:first_valid]))
    for i in range(first_valid, len(close)):
        expected = (sma[i] - sma[i - d]) / (d * sma[i]) * 1000.0
        assert np.isclose(out[i], expected, atol=1e-9, rtol=0.0)

    # Spot value: i = 5, sma[5] = mean(12..15) = 13.5, sma[3] = mean(10..13) = 11.5
    expected_5 = (13.5 - 11.5) / (2 * 13.5) * 1000.0
    assert np.isclose(out[5], expected_5, atol=1e-9)


def test_sma_slope_warmup_is_nan() -> None:
    """Bars with insufficient history (``i < S - 1 + d``) are NaN."""
    close = _random_close(n=120, seed=3)
    s = 16
    d = slope_delta(s)  # max(round(4.0), 2) = 4
    out = sma_slope(close, s)
    first_valid = s - 1 + d
    assert np.all(np.isnan(out[:first_valid]))
    assert np.all(np.isfinite(out[first_valid:]))


def test_sma_slope_constant_series_is_zero_slope() -> None:
    """A flat series has zero SMA-slope on every valid bar."""
    close = np.full(60, 50.0, dtype=np.float64)
    s = 10
    out = sma_slope(close, s)
    first_valid = s - 1 + slope_delta(s)
    np.testing.assert_allclose(out[first_valid:], 0.0, atol=1e-12)


def test_sma_slope_finite_and_bounded_on_random_walk() -> None:
    """Valid SMA-slope outputs are finite and stay within a sane bound."""
    close = _random_close()
    s = 20
    out = sma_slope(close, s)
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    # A 1%-vol random walk cannot produce extreme normalized slopes; 1000 is a
    # very loose finiteness/sanity bound (per-bar drift is sub-percent).
    assert np.all(np.abs(valid) < 1000.0)


# --------------------------------------------------------------------------- #
# linreg_slope_norm — diff of consecutive linreg fits == OLS slope.           #
# --------------------------------------------------------------------------- #


def test_linreg_slope_norm_reference_value_handbuilt() -> None:
    """Normalized linreg-slope equals ``polyfit_slope / sma * 1000`` (L405-406).

    ``ta.linreg(close, S, 0) - ta.linreg(close, S, 1)`` is the difference between
    two evaluation points (offset 0 and 1) on the *same* least-squares line, which
    equals the OLS slope ``b``. We verify against an explicit ``numpy.polyfit``.
    """
    close = np.array(
        [10.0, 10.5, 9.8, 11.2, 12.0, 11.5, 13.0, 12.7, 14.1, 13.5, 15.0, 16.2],
        dtype=np.float64,
    )
    s = 5
    out = linreg_slope_norm(close, s)
    sma = _ref_sma(close, s)

    # Linreg needs a full S-bar window; valid from i = s - 1.
    assert np.all(np.isnan(out[: s - 1]))
    x = np.arange(s, dtype=np.float64)
    for i in range(s - 1, len(close)):
        window = close[i - s + 1 : i + 1]
        b = np.polyfit(x, window, 1)[0]  # OLS slope over the window
        expected = b / sma[i] * 1000.0
        assert np.isclose(out[i], expected, atol=1e-9, rtol=0.0)


def test_linreg_slope_norm_perfect_line_slope() -> None:
    """On an exact arithmetic ramp the OLS slope is the per-bar increment.

    ``close = base + step * i`` over S bars has OLS slope exactly ``step``; the
    normalized value is ``step / sma * 1000``.
    """
    step = 2.0
    base = 100.0
    n = 30
    close = base + step * np.arange(n, dtype=np.float64)
    s = 8
    out = linreg_slope_norm(close, s)
    sma = _ref_sma(close, s)
    for i in range(s - 1, n):
        expected = step / sma[i] * 1000.0
        assert np.isclose(out[i], expected, atol=1e-9, rtol=0.0)


def test_linreg_slope_norm_warmup_is_nan() -> None:
    """Bars before a full ``S``-bar window (``i < S - 1``) are NaN."""
    close = _random_close(n=100, seed=5)
    s = 24
    out = linreg_slope_norm(close, s)
    assert np.all(np.isnan(out[: s - 1]))
    assert np.all(np.isfinite(out[s - 1 :]))


def test_linreg_slope_norm_finite_on_random_walk() -> None:
    """Valid normalized linreg-slope outputs are finite on a random walk."""
    close = _random_close()
    out = linreg_slope_norm(close, 30)
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    assert np.all(np.isfinite(valid))


# --------------------------------------------------------------------------- #
# Output shape / dtype contract shared by both features.                      #
# --------------------------------------------------------------------------- #


def test_vectorized_matches_scalar_reference() -> None:
    """Numba kernels agree with the scalar (polyfit) references to 1e-9.

    Exercises several scales over a random walk, covering the warm-up NaN region
    and the valid region for both features.
    """
    close = _random_close(n=350, seed=21)
    for s in (2, 5, 13, 30, 64):
        assert _vectorized_matches_scalar(close, s, atol=1e-9)


def test_outputs_are_float_arrays_aligned_to_input() -> None:
    """Both features return float64 arrays of the input length."""
    close = _random_close(n=80)
    for fn in (sma_slope, linreg_slope_norm):
        out = fn(close, 12)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == close.shape
