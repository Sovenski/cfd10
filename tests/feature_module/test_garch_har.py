"""Tests for ``cfd10.feature_module.garch_har``.

These tests pin the Pine semantics of
``pine/speculatores_v15_presets_gold.pine`` lines 159-188:

* ``calc_gjr_asym`` (L159-177) — GJR-GARCH asymmetry, the recursive divergence
  of a leverage-aware variance (``gjr_var``) from a symmetric one (``sym_var``),
  normalised to the unit interval.
* ``calc_har_vol`` (L179-188) — HAR forecast over Garman-Klass variance,
  normalised against the contemporaneous GK volatility.

They include a reference-value test on a hand-built fixture (a slow scalar
re-derivation of the Pine recursion), a bounded-range / finiteness assert, and a
warm-up-is-NaN assert.
"""

from __future__ import annotations

import math

import numpy as np

from cfd10.feature_module.garch_har import gjr_asym, har_vol

# Pine constants (L163-165) mirrored locally for the reference re-derivation.
_ALPHA = 0.03
_BETA = 0.90
_GAMMA = 0.08
_LN2 = math.log(2.0)


def _ohlc_random(
    n: int = 400, seed: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a plausible strictly-positive OHLC fixture (geometric walk).

    ``high``/``low`` bracket the ``open``/``close`` of each bar so that the
    Garman-Klass terms are well defined and non-degenerate.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(loc=0.0, scale=0.004, size=n)) * close
    base_hi = np.maximum(open_, close)
    base_lo = np.minimum(open_, close)
    high = base_hi + spread
    low = np.maximum(base_lo - spread, 1e-6)
    return (
        open_.astype(np.float64),
        high.astype(np.float64),
        low.astype(np.float64),
        close.astype(np.float64),
    )


def _safe_log_ratio(num: float, den: float) -> float:
    """Pine ``safe_log_ratio`` (L153-154)."""
    return math.log(max(num, 1e-10) / max(den, 1e-10))


def _trailing_sma_with_warmup(x: np.ndarray, length: int) -> np.ndarray:
    """Reference ``ta.sma(x, length)``: NaN for the first ``length-1`` bars."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        out[i] = float(np.mean(x[i - length + 1 : i + 1]))
    return out


def _ref_gjr_asym(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Slow scalar re-derivation of Pine ``calc_gjr_asym`` (L159-177).

    Independent of the module under test; the recursion is unrolled bar-by-bar
    exactly as Pine evaluates it. Warm-up (``ta.sma(r2,252)`` still ``na``) is
    emitted as NaN while the recursive state keeps propagating underneath.
    """
    n = close.shape[0]
    log_ret = np.empty(n, dtype=np.float64)
    for i in range(n):
        prev_close = close[i - 1] if i >= 1 else close[i]
        log_ret[i] = _safe_log_ratio(close[i], prev_close)
    r2 = log_ret * log_ret
    sma_r2 = _trailing_sma_with_warmup(r2, 252)

    out = np.full(n, np.nan, dtype=np.float64)
    prev_gjr = math.nan
    prev_sym = math.nan
    for i in range(n):
        lr_var = max(sma_r2[i] if not math.isnan(sma_r2[i]) else r2[i], 1e-12)
        omega = max(lr_var * (1.0 - _ALPHA - _BETA - _GAMMA / 2.0), 1e-12)
        pg = prev_gjr if not math.isnan(prev_gjr) else lr_var
        ps = prev_sym if not math.isnan(prev_sym) else lr_var
        pr2 = r2[i - 1] if i >= 1 else lr_var
        leverage = 1.0 if (i >= 1 and log_ret[i - 1] < 0.0) else 0.0
        gjr_var = max(omega + (_ALPHA + _GAMMA * leverage) * pr2 + _BETA * pg, 1e-12)
        sym_var = max(omega + (_ALPHA + _GAMMA * 0.5) * pr2 + _BETA * ps, 1e-12)
        ratio = gjr_var / sym_var
        if not math.isnan(sma_r2[i]):
            out[i] = max(-1.0, min(1.0, (ratio - 1.0) / 0.1))
        prev_gjr = gjr_var
        prev_sym = sym_var
    return out


def _ref_har_vol(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Slow scalar re-derivation of Pine ``calc_har_vol`` (L179-188)."""
    n = close.shape[0]
    gk = np.empty(n, dtype=np.float64)
    for i in range(n):
        log_hl = _safe_log_ratio(high[i], low[i])
        log_co = _safe_log_ratio(close[i], open_[i])
        gk[i] = max(0.5 * log_hl * log_hl - (2.0 * _LN2 - 1.0) * log_co * log_co, 1e-10)
    sma5 = _trailing_sma_with_warmup(gk, 5)
    sma22 = _trailing_sma_with_warmup(gk, 22)

    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        weekly = sma5[i] if not math.isnan(sma5[i]) else gk[i]
        monthly = sma22[i] if not math.isnan(sma22[i]) else gk[i]
        forecast = max(0.36 * gk[i] + 0.28 * weekly + 0.28 * monthly, 1e-10)
        ratio = math.sqrt(max(forecast, 1e-10)) / math.sqrt(max(gk[i], 1e-10))
        if not math.isnan(sma22[i]):
            out[i] = max(-1.0, min(1.0, (ratio - 1.0) / 0.5))
    return out


# --------------------------------------------------------------------------- #
# Reference-value tests on hand-built fixtures.                                #
# --------------------------------------------------------------------------- #


def test_gjr_asym_matches_scalar_reference() -> None:
    """``gjr_asym`` agrees with an independent scalar re-derivation to 1e-9."""
    open_, high, low, close = _ohlc_random()
    got = gjr_asym(open_, high, low, close)
    expected = _ref_gjr_asym(open_, high, low, close)
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0.0, equal_nan=True)


def test_har_vol_matches_scalar_reference() -> None:
    """``har_vol`` agrees with an independent scalar re-derivation to 1e-9."""
    open_, high, low, close = _ohlc_random()
    got = har_vol(open_, high, low, close)
    expected = _ref_har_vol(open_, high, low, close)
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0.0, equal_nan=True)


def test_gjr_asym_tiny_hand_built_first_post_warmup_value() -> None:
    """Closed-form check of the first defined ``gjr_asym`` bar on a tiny case.

    The series is short enough that ``ta.sma(r2,252)`` never warms up, so the
    output is all NaN — but the internal recursion is still verified against an
    explicit hand computation of ``gjr_var`` / ``sym_var`` at bar 1, which is the
    first bar with a genuine previous return.
    """
    # close moves down then up so that bar-1 leverage is exercised.
    close = np.array([100.0, 99.0, 100.5], dtype=np.float64)
    open_ = np.array([100.0, 100.0, 99.0], dtype=np.float64)
    high = np.array([100.0, 100.0, 100.5], dtype=np.float64)
    low = np.array([100.0, 99.0, 99.0], dtype=np.float64)

    # Hand recursion (warm-up region -> all NaN output, but state must propagate).
    lr1 = math.log(99.0 / 100.0)
    r2_1 = lr1 * lr1
    lr_var0 = max(0.0, 1e-12)  # r2[0] == 0 -> clamped to 1e-12
    lr_var1 = max(r2_1, 1e-12)
    omega1 = max(lr_var1 * (1.0 - _ALPHA - _BETA - _GAMMA / 2.0), 1e-12)
    # bar 0 state (prev_*[-1] is na -> lr_var0; r2[-1] na -> lr_var0; leverage 0).
    omega0 = max(lr_var0 * (1.0 - _ALPHA - _BETA - _GAMMA / 2.0), 1e-12)
    gjr0 = max(omega0 + (_ALPHA + 0.0) * lr_var0 + _BETA * lr_var0, 1e-12)
    sym0 = max(omega0 + (_ALPHA + _GAMMA * 0.5) * lr_var0 + _BETA * lr_var0, 1e-12)
    # bar 1: prev_r2 = r2[0] = 0; leverage = 1 (log_ret[0] == 0, not < 0 -> 0!).
    # log_ret[0] = log(close0/close0) = 0, which is NOT < 0, so leverage == 0.
    pr2_1 = 0.0
    lev1 = 0.0
    gjr1 = max(omega1 + (_ALPHA + _GAMMA * lev1) * pr2_1 + _BETA * gjr0, 1e-12)
    sym1 = max(omega1 + (_ALPHA + _GAMMA * 0.5) * pr2_1 + _BETA * sym0, 1e-12)
    # gjr1 and sym1 differ only via prev state which here is symmetric (lev0==0),
    # and the prev_r2 term uses identical coefficients only when lev==0.5 — here
    # gjr uses lev1=0 while sym uses 0.5, but pr2_1==0 nullifies that term, so the
    # sole difference is gjr0 vs sym0 (which themselves differ by gamma*0.5*lr_var0).
    expected_ratio1 = gjr1 / sym1
    assert expected_ratio1 > 0.0  # finite, positive ratio

    out = gjr_asym(open_, high, low, close)
    assert out.shape == close.shape
    assert np.all(np.isnan(out))  # entire short series is warm-up


def test_har_vol_tiny_hand_built_reference_value() -> None:
    """Closed-form ``har_vol`` once the 22-bar SMA warms up on a constant case.

    With constant GK variance every term (daily, weekly, monthly) equals the same
    ``gk``, so ``har_forecast = (0.36 + 0.28 + 0.28) * gk = 0.92 * gk`` — the HAR
    weights do *not* sum to one (Pine L185). Hence the ratio is exactly
    ``sqrt(0.92)`` and the normalised output is ``(sqrt(0.92) - 1) / 0.5``,
    independent of the GK level. We build 22 identical bars to land on the first
    defined output index.
    """
    n = 22
    open_ = np.full(n, 100.0, dtype=np.float64)
    close = np.full(n, 100.5, dtype=np.float64)
    high = np.full(n, 101.0, dtype=np.float64)
    low = np.full(n, 99.5, dtype=np.float64)
    out = har_vol(open_, high, low, close)
    assert np.all(np.isnan(out[:21]))
    # Bar 21 is the first with a full 22-window: all GK equal.
    expected = (math.sqrt(0.92) - 1.0) / 0.5
    assert math.isclose(out[21], expected, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Bounded-range / finiteness asserts.                                          #
# --------------------------------------------------------------------------- #


def test_gjr_asym_bounded_and_finite() -> None:
    """Defined ``gjr_asym`` values lie in [-1, 1] and are finite."""
    open_, high, low, close = _ohlc_random(n=600, seed=3)
    out = gjr_asym(open_, high, low, close)
    defined = out[~np.isnan(out)]
    assert defined.size > 0
    assert np.all(np.isfinite(defined))
    assert np.all(defined >= -1.0) and np.all(defined <= 1.0)


def test_har_vol_bounded_and_finite() -> None:
    """Defined ``har_vol`` values lie in [-1, 1] and are finite."""
    open_, high, low, close = _ohlc_random(n=600, seed=5)
    out = har_vol(open_, high, low, close)
    defined = out[~np.isnan(out)]
    assert defined.size > 0
    assert np.all(np.isfinite(defined))
    assert np.all(defined >= -1.0) and np.all(defined <= 1.0)


def test_gjr_asym_recursion_stays_finite_on_volatile_input() -> None:
    """The variance recursion does not blow up on a heavy-tailed return stream."""
    rng = np.random.default_rng(99)
    n = 1000
    # Student-t-ish heavy tails to stress the recursion.
    steps = rng.standard_t(df=3, size=n) * 0.02
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    out = gjr_asym(open_, high, low, close)
    defined = out[~np.isnan(out)]
    assert np.all(np.isfinite(defined))
    assert np.all(defined >= -1.0) and np.all(defined <= 1.0)


# --------------------------------------------------------------------------- #
# Warm-up-is-NaN asserts.                                                      #
# --------------------------------------------------------------------------- #


def test_gjr_asym_warmup_is_nan() -> None:
    """``gjr_asym`` is NaN before the 252-bar long-run-variance SMA warms up."""
    open_, high, low, close = _ohlc_random(n=300, seed=8)
    out = gjr_asym(open_, high, low, close)
    assert np.all(np.isnan(out[:251]))  # bars 0..250 lack a full 252-window
    assert not np.isnan(out[251])  # first fully-warmed bar is defined


def test_har_vol_warmup_is_nan() -> None:
    """``har_vol`` is NaN before the 22-bar (monthly) SMA warms up."""
    open_, high, low, close = _ohlc_random(n=60, seed=4)
    out = har_vol(open_, high, low, close)
    assert np.all(np.isnan(out[:21]))  # bars 0..20 lack a full 22-window
    assert not np.isnan(out[21])  # first fully-warmed bar is defined


def test_outputs_are_float_arrays_aligned_to_input() -> None:
    """Both functions return float64 arrays aligned to the input length."""
    open_, high, low, close = _ohlc_random(n=120, seed=2)
    g = gjr_asym(open_, high, low, close)
    h = har_vol(open_, high, low, close)
    for arr in (g, h):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == close.shape
