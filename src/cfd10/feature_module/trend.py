"""Per-side trend-slope features — Pine parity for L402-413.

Faithful Python port of the ``slope_val_*`` / ``linreg_norm_*`` block in
``pine/speculatores_v15_presets_gold.pine`` (lines 402-413, with the slope-delta
definition at L288/L295). Two scale-parameterised features are exposed:

SMA-slope (Pine L404/L410)
    ``slope = sma > 0 ? nz(sma - sma[d]) / (d * sma) * 1000 : 0.0`` where the
    trailing simple moving average is ``ta.sma(close, S)`` and the lag is
    ``d = max(round(S / 4), 2)``.

Linreg-slope, normalized (Pine L405-406/L411-412)
    ``ta.linreg(close, S, 0) - ta.linreg(close, S, 1)`` normalized
    ``/ sma * 1000``. Pine's ``ta.linreg(src, length, offset)`` returns one point
    of a single least-squares line fit over the last ``length`` bars, evaluated
    ``offset`` bars back; the offset-0-minus-offset-1 difference is therefore the
    regression slope ``b`` itself. We compute ``b`` directly (closed-form OLS,
    identical to ``numpy.polyfit`` to machine precision) and divide by the SMA.

Warm-up convention (documented divergence from Pine)
----------------------------------------------------
Pine's ternary guard ``sma > 0 ? ... : 0.0`` emits ``0.0`` during warm-up because
``na > 0`` is falsy. The Python feature layer instead emits ``np.nan`` for bars
with insufficient history (the correct ML sentinel; the Pine ``0.0`` is a
plotting artifact). Concretely:

* :func:`sma_slope` is ``NaN`` for ``i < S - 1 + d`` (needs both ``sma[i]`` and
  ``sma[i - d]``), then valid; a valid bar with a non-positive SMA still maps to
  ``0.0`` exactly as Pine.
* :func:`linreg_slope_norm` is ``NaN`` for ``i < S - 1`` (needs a full ``S``-bar
  regression window), then valid.

All outputs are ``float64`` ``np.ndarray`` aligned to the input length. Each
public feature has a scalar reference (the normative spec, ``numpy.polyfit`` for
linreg) and a vectorized Numba counterpart; :mod:`tests.feature_module.test_trend`
plus :func:`_vectorized_matches_scalar` pin them to agree to ``1e-9``.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Callable
from numba import njit
from numpy.typing import NDArray
from typing import TypeVar

from cfd10.feature_module.sma_pir import csum_close, sma_at
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "slope_delta",
    "sma_slope",
    "linreg_slope_norm",
    "TREND_FACTORY",
    "register_trend",
    "get_trend_fn",
    "TrendFn",
]

_NAN: float = float("nan")


def slope_delta(s: int) -> int:
    """Return the SMA-slope lag ``d = max(round(S / 4), 2)`` (Pine L288/L295).

    Pine ``math.round`` rounds half away from zero; for the positive integer
    quotients produced by ``S / 4`` this coincides with :func:`round`'s result on
    non-half values, and the ``max(., 2)`` floor dominates the only half case
    (``S / 4 == 0.5`` at ``S == 2``). We therefore use the away-from-zero rule
    explicitly to stay faithful regardless of ``S``.

    Args:
        s: SMA window length (scale), must be a positive integer.

    Returns:
        The integer lag ``d`` (at least 2).
    """
    if s <= 0:
        raise ValueError(f"slope_delta: S must be positive, got {s}")
    rounded = int(np.floor(s / 4.0 + 0.5))  # round-half-away-from-zero (S > 0)
    return rounded if rounded > 2 else 2


# --------------------------------------------------------------------------- #
# SMA series (trailing ta.sma) via the shared cumulative-sum helper.          #
# --------------------------------------------------------------------------- #


def _sma_series(close: NDArray[np.float64], s: int) -> NDArray[np.float64]:
    """Full trailing-SMA series (Pine ``ta.sma(close, S)``), warm-up = NaN.

    Reuses the tested :func:`cfd10.feature_module.sma_pir.sma_at` semantics
    (cumulative-sum subtraction with the ``-1`` boundary rule) so that the SMA
    here is bit-identical to the parity lynchpin. ``out[i]`` is the mean of the
    ``s`` closes ending at bar ``i`` for ``i >= s - 1`` and ``NaN`` before.

    Args:
        close: 1-D close-price array.
        s: SMA window length.

    Returns:
        ``float64`` array of length ``len(close)``.
    """
    csum = csum_close(close)
    n = close.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    for i in range(s - 1, n):
        out[i] = sma_at(csum, s, 0, i)
    return out


# --------------------------------------------------------------------------- #
# Numba kernels (module scope so the JIT cache is shared across calls).        #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _sma_slope_kernel(
    sma: NDArray[np.float64], d: int, n: int
) -> NDArray[np.float64]:
    """Vectorized SMA-slope from a precomputed SMA series (Pine L404).

    ``out[i] = (sma[i] - sma[i - d]) / (d * sma[i]) * 1000`` once both ``sma[i]``
    and ``sma[i - d]`` are valid; ``NaN`` during warm-up; ``0.0`` on a valid bar
    whose SMA is non-positive (mirroring the Pine ternary's false branch).
    """
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        cur = sma[i]
        if np.isnan(cur):
            continue
        prev = sma[i - d] if i - d >= 0 else np.nan
        if np.isnan(prev):
            continue  # warm-up: sma[i - d] not yet defined
        if cur > 0.0:
            out[i] = (cur - prev) / (d * cur) * 1000.0
        else:
            out[i] = 0.0  # Pine false branch of ``sma > 0 ? ... : 0.0``
    return out


@njit(cache=True, fastmath=False)
def _ols_slope(window: NDArray[np.float64], s: int) -> float:
    """Closed-form OLS slope over ``s`` points ``x = 0..s-1`` vs ``window``.

    Numerically identical to ``numpy.polyfit(x, window, 1)[0]`` to machine
    precision. Equals Pine's ``ta.linreg(.,S,0) - ta.linreg(.,S,1)`` (the slope of
    the single least-squares line over the last ``S`` bars).
    """
    # x = 0..s-1: mean = (s-1)/2, Sxx = sum (x-mean)^2 = s(s^2-1)/12.
    mean_x = (s - 1) / 2.0
    mean_y = 0.0
    for k in range(s):
        mean_y += window[k]
    mean_y /= s
    sxy = 0.0
    for k in range(s):
        sxy += (k - mean_x) * (window[k] - mean_y)
    sxx = s * (s * s - 1.0) / 12.0
    return sxy / sxx


@njit(cache=True, fastmath=False)
def _linreg_slope_norm_kernel(
    close: NDArray[np.float64], sma: NDArray[np.float64], s: int, n: int
) -> NDArray[np.float64]:
    """Vectorized normalized linreg-slope (Pine L405-406).

    For ``i >= s - 1`` computes the OLS slope of ``close[i-s+1 .. i]`` and divides
    by ``sma[i]`` (``* 1000``); ``NaN`` before a full window; ``0.0`` on a valid
    bar whose SMA is non-positive (Pine ternary false branch).
    """
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(s - 1, n):
        cur = sma[i]
        if np.isnan(cur):
            continue
        b = _ols_slope(close[i - s + 1 : i + 1], s)
        if cur > 0.0:
            out[i] = b / cur * 1000.0
        else:
            out[i] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Scalar reference implementations (normative spec; polyfit for linreg).       #
# --------------------------------------------------------------------------- #


def _sma_slope_scalar(close: NDArray[np.float64], s: int) -> NDArray[np.float64]:
    """Scalar reference for :func:`sma_slope` (pure NumPy, no JIT)."""
    sma = _sma_series(close, s)
    d = slope_delta(s)
    n = close.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    for i in range(n):
        cur = sma[i]
        if np.isnan(cur) or i - d < 0 or np.isnan(sma[i - d]):
            continue
        out[i] = (cur - sma[i - d]) / (d * cur) * 1000.0 if cur > 0.0 else 0.0
    return out


def _linreg_slope_norm_scalar(
    close: NDArray[np.float64], s: int
) -> NDArray[np.float64]:
    """Scalar reference for :func:`linreg_slope_norm` using ``numpy.polyfit``."""
    sma = _sma_series(close, s)
    n = close.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    x = np.arange(s, dtype=np.float64)
    for i in range(s - 1, n):
        cur = sma[i]
        if np.isnan(cur):
            continue
        b = float(np.polyfit(x, close[i - s + 1 : i + 1], 1)[0])
        out[i] = b / cur * 1000.0 if cur > 0.0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# Public thin wrappers (validate inputs, then dispatch to the JIT kernels).    #
# --------------------------------------------------------------------------- #


def _validate(close: NDArray[np.float64], s: int) -> NDArray[np.float64]:
    """Coerce ``close`` to contiguous float64 and validate the scale ``s``."""
    if s <= 0:
        raise ValueError(f"S must be positive, got {s}")
    arr = np.ascontiguousarray(close, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"close must be 1-D, got shape {arr.shape}")
    return arr


def sma_slope(close: NDArray[np.float64], s: int) -> NDArray[np.float64]:
    """SMA-slope feature (Pine L404/L410).

    ``slope[i] = (sma[i] - sma[i - d]) / (d * sma[i]) * 1000`` with
    ``sma = ta.sma(close, S)`` and ``d = max(round(S / 4), 2)`` (:func:`slope_delta`).

    Args:
        close: 1-D close-price array.
        s: SMA window length (scale), positive.

    Returns:
        ``float64`` array of length ``len(close)``; ``NaN`` for warm-up bars
        ``i < S - 1 + d``; ``0.0`` on a valid bar whose SMA is non-positive.
    """
    arr = _validate(close, s)
    sma = _sma_series(arr, s)
    d = slope_delta(s)
    return _sma_slope_kernel(sma, d, arr.shape[0])


def linreg_slope_norm(close: NDArray[np.float64], s: int) -> NDArray[np.float64]:
    """Normalized linreg-slope feature (Pine L405-406/L411-412).

    Equals ``(ta.linreg(close, S, 0) - ta.linreg(close, S, 1)) / sma * 1000`` —
    i.e. the OLS slope of the last ``S`` closes divided by ``ta.sma(close, S)``.

    Args:
        close: 1-D close-price array.
        s: Regression / SMA window length (scale), positive.

    Returns:
        ``float64`` array of length ``len(close)``; ``NaN`` for warm-up bars
        ``i < S - 1``; ``0.0`` on a valid bar whose SMA is non-positive.
    """
    arr = _validate(close, s)
    sma = _sma_series(arr, s)
    return _linreg_slope_norm_kernel(arr, sma, s, arr.shape[0])


def _vectorized_matches_scalar(
    close: NDArray[np.float64], s: int, atol: float = 1e-9
) -> bool:
    """Return True iff both vectorized features match their scalar references.

    Internal parity guard exercised by the test-suite (NaNs compared by position,
    finite values to ``atol``). Kept here so the scalar/vectorized contract lives
    beside the implementations it constrains.

    Args:
        close: 1-D close-price array.
        s: Scale to compare at.
        atol: Absolute tolerance for finite-value agreement.

    Returns:
        ``True`` when every bar agrees (NaN-position and value) for both features.
    """
    arr = _validate(close, s)
    for fast, ref in (
        (sma_slope(arr, s), _sma_slope_scalar(arr, s)),
        (linreg_slope_norm(arr, s), _linreg_slope_norm_scalar(arr, s)),
    ):
        if not np.array_equal(np.isnan(fast), np.isnan(ref)):
            return False
        mask = ~np.isnan(fast)
        if not np.allclose(fast[mask], ref[mask], atol=atol, rtol=0.0):
            return False
    return True


# --------------------------------------------------------------------------- #
# Registry / factory for the per-scale trend feature functions.               #
# --------------------------------------------------------------------------- #

# A trend feature: maps ``(close, S)`` to a full-length float64 series.
TrendFn = Callable[[NDArray[np.float64], int], NDArray[np.float64]]

TREND_FACTORY: dict[str, TrendFn] = {}

_F = TypeVar("_F", bound=TrendFn)


def register_trend(name: str) -> Callable[[_F], _F]:
    """Register a trend feature function under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator recording the function in :data:`TREND_FACTORY` and returning
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: _F) -> _F:
        if name in TREND_FACTORY:
            raise ValueError(f"register_trend: duplicate registration for {name!r}")
        TREND_FACTORY[name] = fn
        logger.debug("register_trend: registered %s", name)
        return fn

    return decorator


def get_trend_fn(name: str) -> TrendFn:
    """Resolve a registered trend feature function by name.

    Args:
        name: Registry key used with :func:`register_trend`.

    Returns:
        The registered callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return TREND_FACTORY[name]
    except KeyError as exc:
        known = ", ".join(sorted(TREND_FACTORY)) or "<empty>"
        raise KeyError(
            f"get_trend_fn: unknown trend function {name!r}; known: {known}"
        ) from exc


register_trend("sma_slope")(sma_slope)
register_trend("linreg_slope_norm")(linreg_slope_norm)
