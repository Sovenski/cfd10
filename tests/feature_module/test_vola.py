"""Tests for ``cfd10.feature_module.vola`` — per-side volatility features.

These tests pin the Pine semantics of ``speculatores_v15_presets_gold.pine``
lines 456-462:

* ``ATR``      -> ``ta.atr(len)``  = Wilder RMA of the true range.
* ``StdDev``   -> ``ta.stdev(close, len)`` (population / biased stdev).
* ``Intraday`` -> ``ta.sma(close > 0 ? (high - low) / close : 0.0, len)``.
* ``vola_pos`` -> ``pir_of(vola_raw, range_len)`` (trailing position-in-range).

Coverage: reference-value tests on small hand-built fixtures, bounded-range /
finiteness asserts (including one on real market data), and warm-up-is-NaN
asserts for every method.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfd10.feature_module.sma_pir import pir_of_series
from cfd10.feature_module.vola import (
    VOLA_FACTORY,
    atr_reference,
    intraday_range_reference,
    stdev_reference,
    vola_position,
    vola_raw,
)

_DATA_DIR = Path(r"C:\Users\kuben\Desktop\Projekte\cfd10\data\raw_v16")


# --------------------------------------------------------------------------- #
# Hand-built fixtures.                                                         #
# --------------------------------------------------------------------------- #


def _ohlc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a tiny strictly-positive OHLC fixture (high, low, close)."""
    high = np.array([11.0, 12.0, 11.5, 13.0, 12.5, 14.0], dtype=np.float64)
    low = np.array([9.0, 10.0, 10.5, 11.0, 11.5, 12.0], dtype=np.float64)
    close = np.array([10.0, 11.0, 11.0, 12.0, 12.0, 13.0], dtype=np.float64)
    return high, low, close


def _true_range_ref(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Reference Pine ``ta.tr(true)``: high-low on bar 0, else max of the 3 ranges."""
    n = high.shape[0]
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def _rma_ref(src: np.ndarray, length: int) -> np.ndarray:
    """Reference Pine ``ta.rma``: SMA-seeded Wilder smoothing; warm-up -> NaN."""
    n = src.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    seed = src[:length].mean()
    out[length - 1] = seed
    prev = seed
    for i in range(length, n):
        prev = (prev * (length - 1) + src[i]) / length
        out[i] = prev
    return out


# --------------------------------------------------------------------------- #
# Reference-value tests on the hand-built fixture.                             #
# --------------------------------------------------------------------------- #


def test_atr_matches_wilder_reference() -> None:
    """ATR equals an independent Wilder-RMA-of-true-range reference."""
    high, low, close = _ohlc()
    length = 3
    expected = _rma_ref(_true_range_ref(high, low, close), length)
    got = vola_raw(high, low, close, "ATR", length)
    np.testing.assert_allclose(got, expected, atol=1e-12, equal_nan=True)


def test_atr_first_defined_value_is_sma_of_true_range() -> None:
    """The first defined ATR (index len-1) is the plain mean of the first len TRs."""
    high, low, close = _ohlc()
    length = 3
    tr = _true_range_ref(high, low, close)
    got = vola_raw(high, low, close, "ATR", length)
    assert math.isclose(got[length - 1], tr[:length].mean(), abs_tol=1e-12)


def test_stdev_matches_population_std() -> None:
    """StdDev equals the trailing population (biased) std of close (Pine ta.stdev)."""
    high, low, close = _ohlc()
    length = 3
    got = vola_raw(high, low, close, "StdDev", length)
    expected = np.full_like(close, np.nan)
    for i in range(length - 1, len(close)):
        window = close[i - length + 1 : i + 1]
        expected[i] = window.std(ddof=0)  # population std, divides by N
    np.testing.assert_allclose(got, expected, atol=1e-12, equal_nan=True)


def test_intraday_matches_sma_of_range_ratio() -> None:
    """Intraday equals SMA of (high-low)/close over the window (Pine L456)."""
    high, low, close = _ohlc()
    length = 3
    ratio = np.where(close > 0.0, (high - low) / close, 0.0)
    got = vola_raw(high, low, close, "Intraday", length)
    expected = np.full_like(close, np.nan)
    for i in range(length - 1, len(close)):
        expected[i] = ratio[i - length + 1 : i + 1].mean()
    np.testing.assert_allclose(got, expected, atol=1e-12, equal_nan=True)


def test_intraday_nonpositive_close_uses_zero_ratio() -> None:
    """A non-positive close contributes a 0.0 ratio, not a division-by-zero."""
    high = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    low = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    close = np.array([1.0, 0.0, 2.0], dtype=np.float64)  # middle close is 0
    length = 1
    got = vola_raw(high, low, close, "Intraday", length)
    # length=1 -> SMA is the bar's own ratio; the zero-close bar must be 0.0.
    assert got[1] == 0.0
    assert math.isclose(got[0], (1.0 - 0.5) / 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Method dispatch (registry/factory) and case-insensitive aliasing.           #
# --------------------------------------------------------------------------- #


def test_factory_lists_three_methods() -> None:
    """The volatility-method registry exposes exactly ATR, StdDev, Intraday."""
    assert set(VOLA_FACTORY) == {"ATR", "StdDev", "Intraday"}


def test_unknown_method_raises() -> None:
    """An unknown method name raises a helpful KeyError."""
    high, low, close = _ohlc()
    with pytest.raises(KeyError):
        vola_raw(high, low, close, "NotAMethod", 3)


# --------------------------------------------------------------------------- #
# vola_position = pir_of(vola_raw, range_len).                                 #
# --------------------------------------------------------------------------- #


def test_vola_position_is_pir_of_series() -> None:
    """``vola_position`` is exactly ``pir_of_series`` of the raw vola (Pine L457)."""
    high, low, close = _ohlc()
    raw = vola_raw(high, low, close, "StdDev", 2)
    range_len = 3
    got = vola_position(raw, range_len)
    expected = pir_of_series(raw, range_len)
    np.testing.assert_allclose(got, expected, atol=1e-12, equal_nan=True)


def test_vola_position_within_unit_interval_where_defined() -> None:
    """Where defined, vola_position lies within [0, 1]."""
    high, low, close = _ohlc()
    raw = vola_raw(high, low, close, "Intraday", 2)
    pos = vola_position(raw, 3)
    finite = pos[np.isfinite(pos)]
    assert finite.size > 0
    assert np.all(finite >= 0.0) and np.all(finite <= 1.0)


# --------------------------------------------------------------------------- #
# Warm-up handling: insufficient history is NaN.                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["ATR", "StdDev", "Intraday"])
def test_warmup_bars_are_nan(method: str) -> None:
    """The first ``length - 1`` bars lack history and must be NaN for every method."""
    high, low, close = _ohlc()
    length = 4
    got = vola_raw(high, low, close, method, length)
    assert np.all(np.isnan(got[: length - 1]))
    assert np.isfinite(got[length - 1])  # first fully-warmed bar is defined
    assert got.shape == close.shape
    assert got.dtype == np.float64


def test_length_one_has_no_warmup_for_stdev_and_intraday() -> None:
    """length=1 -> StdDev is all-zero and Intraday is defined on every bar."""
    high, low, close = _ohlc()
    std = vola_raw(high, low, close, "StdDev", 1)
    np.testing.assert_allclose(std, 0.0, atol=1e-12)  # std of a single value is 0
    intra = vola_raw(high, low, close, "Intraday", 1)
    assert np.all(np.isfinite(intra))


# --------------------------------------------------------------------------- #
# Scalar reference vs vectorized Numba kernel: agree to 1e-9.                  #
# --------------------------------------------------------------------------- #


def _random_ohlc(n: int = 400, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly-positive synthetic OHLC (geometric walk + intrabar spread)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    spread = np.abs(rng.normal(0.0, 0.5, size=n)) + 0.1
    high = close + spread
    low = np.maximum(close - spread, 0.01)
    return high.astype(np.float64), low.astype(np.float64), close.astype(np.float64)


def test_atr_kernel_matches_scalar_reference() -> None:
    """Vectorized ATR agrees with the pure-Python scalar reference to 1e-9."""
    high, low, close = _random_ohlc()
    for length in (2, 5, 14, 50):
        vec = vola_raw(high, low, close, "ATR", length)
        ref = atr_reference(high, low, close, length)
        np.testing.assert_allclose(vec, ref, atol=1e-9, rtol=0.0, equal_nan=True)


def test_stdev_kernel_matches_scalar_reference() -> None:
    """Vectorized StdDev agrees with the pure-Python scalar reference to 1e-9."""
    high, low, close = _random_ohlc()
    for length in (2, 5, 14, 50):
        vec = vola_raw(high, low, close, "StdDev", length)
        ref = stdev_reference(close, length)
        np.testing.assert_allclose(vec, ref, atol=1e-9, rtol=0.0, equal_nan=True)


def test_intraday_kernel_matches_scalar_reference() -> None:
    """Vectorized Intraday agrees with the pure-Python scalar reference to 1e-9."""
    high, low, close = _random_ohlc()
    for length in (2, 5, 14, 50):
        vec = vola_raw(high, low, close, "Intraday", length)
        ref = intraday_range_reference(high, low, close, length)
        np.testing.assert_allclose(vec, ref, atol=1e-9, rtol=0.0, equal_nan=True)


# --------------------------------------------------------------------------- #
# Finiteness / bounded-range on real market data.                             #
# --------------------------------------------------------------------------- #


def _load_real_ohlc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one real market OHLC series from the read-only data directory."""
    csvs = sorted(_DATA_DIR.glob("*.csv"))
    if not csvs:
        pytest.skip(f"no market CSVs under {_DATA_DIR}")
    frame = pd.read_csv(csvs[0])
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    return high, low, close


@pytest.mark.parametrize("method", ["ATR", "StdDev", "Intraday"])
def test_real_data_finite_after_warmup_and_position_bounded(method: str) -> None:
    """On real data: vola is finite & non-negative post-warm-up; position in [0, 1]."""
    high, low, close = _load_real_ohlc()
    length = 14
    raw = vola_raw(high, low, close, method, length)
    assert raw.shape == close.shape
    warmed = raw[length:]
    assert np.all(np.isfinite(warmed))
    assert np.all(warmed >= -1e-12)  # volatility is non-negative

    pos = vola_position(raw, range_len=100)
    finite = pos[np.isfinite(pos)]
    assert finite.size > 0
    assert np.all(finite >= -1e-12) and np.all(finite <= 1.0 + 1e-12)
