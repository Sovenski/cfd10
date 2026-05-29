"""Per-side volatility features — faithful port of the Pine volatility block.

This module ports ``pine/speculatores_v15_presets_gold.pine`` lines 456-462,
which compute a *raw* volatility series per side and then its trailing
position-in-range::

    vola_raw = vola_method == "ATR"      ? ta.atr(len)
             : vola_method == "Intraday" ? ta.sma(close > 0 ? (high - low) / close : 0.0, len)
             :                             ta.stdev(close, len)          // default
    vola_pos = pir_of(vola_raw, range_len)

Pine built-in semantics matched here
------------------------------------
* ``ta.atr(len)`` == ``ta.rma(ta.tr(true), len)``. ``ta.tr(true)`` is the true
  range with the first-bar special case ``high - low`` (because ``close[1]`` is
  ``na``); ``ta.rma`` is **Wilder's** smoothing seeded by the simple mean of the
  first ``len`` true-range values. The Wilder recurrence is
  ``rma[i] = (rma[i-1] * (len - 1) + tr[i]) / len``.
* ``ta.stdev(close, len)`` is the **population** (biased, divide-by-``N``)
  standard deviation over the trailing ``len``-bar window — Pine's default
  ``biased = true``.
* ``ta.sma(x, len)`` is the simple moving average of the per-bar intraday range
  ratio ``(high - low) / close`` (or ``0.0`` when ``close <= 0``).

Warm-up convention
------------------
The first ``len - 1`` bars lack a full trailing window and are returned as
``np.nan`` (Pine ``na`` propagation). The first fully-warmed bar is at index
``len - 1``. ``vola_position`` propagates those ``NaN``s through
:func:`cfd10.feature_module.sma_pir.pir_of_series`.

Each Numba-vectorized kernel has a pure-Python scalar reference
(``*_reference``); :mod:`tests.feature_module.test_vola` pins them to agree to
``1e-9`` and pins the kernels to independent hand-computed values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cfd10.feature_module.sma_pir import pir_of_series
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "VOLA_METHODS",
    "VOLA_FACTORY",
    "register_vola_method",
    "get_vola_method",
    "true_range",
    "atr",
    "stdev",
    "intraday_range",
    "vola_raw",
    "vola_position",
    "atr_reference",
    "stdev_reference",
    "intraday_range_reference",
]

# A volatility-method kernel: (high, low, close, length) -> raw volatility series.
VolaMethodFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int],
    NDArray[np.float64],
]

# Canonical Pine method names (L456/L460). Order documents the Pine ternary.
VOLA_METHODS: tuple[str, ...] = ("ATR", "StdDev", "Intraday")

VOLA_FACTORY: dict[str, VolaMethodFn] = {}


def register_vola_method(name: str) -> Callable[[VolaMethodFn], VolaMethodFn]:
    """Register a volatility-method kernel under its canonical Pine ``name``.

    Args:
        name: Canonical method name (e.g. ``"ATR"``).

    Returns:
        A decorator recording the function in :data:`VOLA_FACTORY` and returning
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: VolaMethodFn) -> VolaMethodFn:
        if name in VOLA_FACTORY:
            raise ValueError(f"register_vola_method: duplicate registration for {name!r}")
        VOLA_FACTORY[name] = fn
        logger.debug("register_vola_method: registered %s", name)
        return fn

    return decorator


def get_vola_method(name: str) -> VolaMethodFn:
    """Resolve a volatility-method kernel by name (case-insensitive).

    Args:
        name: Method name; matched case-insensitively against
            :data:`VOLA_METHODS` so ``"atr"`` resolves to ``"ATR"``.

    Returns:
        The registered kernel callable.

    Raises:
        KeyError: If ``name`` does not match a known method.
    """
    if name in VOLA_FACTORY:
        return VOLA_FACTORY[name]
    lowered = name.casefold()
    for canonical in VOLA_FACTORY:
        if canonical.casefold() == lowered:
            return VOLA_FACTORY[canonical]
    known = ", ".join(VOLA_FACTORY) or "<empty>"
    raise KeyError(f"get_vola_method: unknown volatility method {name!r}; known: {known}")


# --------------------------------------------------------------------------- #
# Numba kernels (module scope so the JIT cache is shared across calls).        #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pine ``ta.tr(true)``: ``high - low`` on bar 0, else max of the three ranges."""
    n = high.shape[0]
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close = close[i - 1]
        hl = high[i] - low[i]
        hc = abs(high[i] - prev_close)
        lc = abs(low[i] - prev_close)
        m = hl
        if hc > m:
            m = hc
        if lc > m:
            m = lc
        tr[i] = m
    return tr


@njit(cache=True, fastmath=False)
def _rma(src: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    """Pine ``ta.rma``: SMA-seeded Wilder smoothing; warm-up (< len-1) is NaN."""
    n = src.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    acc = 0.0
    for k in range(length):
        acc += src[k]
    prev = acc / length
    out[length - 1] = prev
    for i in range(length, n):
        prev = (prev * (length - 1) + src[i]) / length
        out[i] = prev
    return out


@njit(cache=True, fastmath=False)
def _atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """Pine ``ta.atr(len)`` == ``ta.rma(ta.tr(true), len)``."""
    return _rma(_true_range(high, low, close), length)


@njit(cache=True, fastmath=False)
def _stdev(close: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    """Pine ``ta.stdev(close, len)``: trailing population (biased) std; warm-up NaN."""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    for i in range(length - 1, n):
        mean = 0.0
        for j in range(i - length + 1, i + 1):
            mean += close[j]
        mean /= length
        var = 0.0
        for j in range(i - length + 1, i + 1):
            d = close[j] - mean
            var += d * d
        out[i] = np.sqrt(var / length)
    return out


@njit(cache=True, fastmath=False)
def _intraday_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """Pine ``ta.sma(close > 0 ? (high - low) / close : 0.0, len)``; warm-up NaN."""
    n = high.shape[0]
    ratio = np.empty(n, dtype=np.float64)
    for i in range(n):
        ratio[i] = (high[i] - low[i]) / close[i] if close[i] > 0.0 else 0.0
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(i - length + 1, i + 1):
            acc += ratio[j]
        out[i] = acc / length
    return out


# --------------------------------------------------------------------------- #
# Public method kernels (validate, contiguous-cast, dispatch to JIT).          #
# --------------------------------------------------------------------------- #


def _as_f64(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``arr`` as a contiguous float64 array (no copy when already so)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """True range series (Pine ``ta.tr(true)``).

    Args:
        high: 1-D high-price array.
        low: 1-D low-price array.
        close: 1-D close-price array.

    Returns:
        Float64 true-range array aligned to the inputs; bar 0 is ``high - low``.
    """
    return _true_range(_as_f64(high), _as_f64(low), _as_f64(close))


@register_vola_method("ATR")
def atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """Average true range (Pine ``ta.atr(len)`` = Wilder RMA of true range).

    Args:
        high: 1-D high-price array.
        low: 1-D low-price array.
        close: 1-D close-price array.
        length: Averaging window; the first ``length - 1`` bars are ``NaN``.

    Returns:
        Float64 ATR array aligned to the inputs.
    """
    return _atr(_as_f64(high), _as_f64(low), _as_f64(close), int(length))


@register_vola_method("StdDev")
def stdev(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """Trailing population standard deviation of ``close`` (Pine ``ta.stdev``).

    The ``high``/``low`` arguments are accepted for a uniform method signature
    but unused, matching Pine ``ta.stdev(close, len)``.

    Args:
        high: Unused (kept for signature uniformity).
        low: Unused (kept for signature uniformity).
        close: 1-D close-price array.
        length: Window length; the first ``length - 1`` bars are ``NaN``.

    Returns:
        Float64 standard-deviation array aligned to ``close``.
    """
    del high, low  # Pine ta.stdev uses close only.
    return _stdev(_as_f64(close), int(length))


@register_vola_method("Intraday")
def intraday_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """SMA of the intraday range ratio ``(high - low) / close`` (Pine L456).

    Args:
        high: 1-D high-price array.
        low: 1-D low-price array.
        close: 1-D close-price array (non-positive closes contribute a 0.0 ratio).
        length: SMA window; the first ``length - 1`` bars are ``NaN``.

    Returns:
        Float64 intraday-range array aligned to the inputs.
    """
    return _intraday_range(_as_f64(high), _as_f64(low), _as_f64(close), int(length))


# --------------------------------------------------------------------------- #
# Top-level dispatchers (Pine L456-457 / L460-461).                            #
# --------------------------------------------------------------------------- #


def vola_raw(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    method: str,
    length: int,
) -> NDArray[np.float64]:
    """Raw per-side volatility (Pine L456/L460).

    Dispatches on ``method`` to one of the registered kernels.

    Args:
        high: 1-D high-price array.
        low: 1-D low-price array.
        close: 1-D close-price array.
        method: One of :data:`VOLA_METHODS` (``"ATR"``, ``"StdDev"``,
            ``"Intraday"``); matched case-insensitively.
        length: Calculation window; the first ``length - 1`` bars are ``NaN``.

    Returns:
        Float64 raw-volatility array aligned to the inputs.

    Raises:
        KeyError: If ``method`` is not a known volatility method.
        ValueError: If the input arrays differ in length.
    """
    high_a, low_a, close_a = _as_f64(high), _as_f64(low), _as_f64(close)
    if not (high_a.shape == low_a.shape == close_a.shape):
        raise ValueError(
            "vola_raw: high, low, close must share a shape; got "
            f"{high_a.shape}, {low_a.shape}, {close_a.shape}"
        )
    kernel = get_vola_method(method)
    return kernel(high_a, low_a, close_a, int(length))


def vola_position(
    vola_raw_series: NDArray[np.float64],
    range_len: int,
) -> NDArray[np.float64]:
    """Trailing position-in-range of the raw volatility (Pine ``pir_of``, L457).

    Args:
        vola_raw_series: Raw volatility series from :func:`vola_raw`.
        range_len: Trailing window length passed to
            :func:`cfd10.feature_module.sma_pir.pir_of_series`.

    Returns:
        Float64 position array in ``[0, 1]`` where defined (``0.5`` on a flat
        window); ``NaN`` propagates from warm-up bars of ``vola_raw_series``.
    """
    return pir_of_series(_as_f64(vola_raw_series), int(range_len))


# --------------------------------------------------------------------------- #
# Pure-Python scalar references (pinned to the kernels at 1e-9 in tests).      #
# --------------------------------------------------------------------------- #


def atr_reference(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """NumPy scalar reference for :func:`atr` (Wilder RMA of true range)."""
    high_a, low_a, close_a = _as_f64(high), _as_f64(low), _as_f64(close)
    n = high_a.shape[0]
    tr = np.empty(n, dtype=np.float64)
    if n:
        tr[0] = high_a[0] - low_a[0]
        for i in range(1, n):
            pc = close_a[i - 1]
            tr[i] = max(high_a[i] - low_a[i], abs(high_a[i] - pc), abs(low_a[i] - pc))
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    prev = float(tr[:length].mean())
    out[length - 1] = prev
    for i in range(length, n):
        prev = (prev * (length - 1) + tr[i]) / length
        out[i] = prev
    return out


def stdev_reference(
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """NumPy scalar reference for :func:`stdev` (population std over a window)."""
    close_a = _as_f64(close)
    n = close_a.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    for i in range(length - 1, n):
        window = close_a[i - length + 1 : i + 1]
        out[i] = float(window.std(ddof=0))
    return out


def intraday_range_reference(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    length: int,
) -> NDArray[np.float64]:
    """NumPy scalar reference for :func:`intraday_range` (SMA of range ratio)."""
    high_a, low_a, close_a = _as_f64(high), _as_f64(low), _as_f64(close)
    n = high_a.shape[0]
    ratio = np.where(close_a > 0.0, (high_a - low_a) / np.where(close_a > 0.0, close_a, 1.0), 0.0)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out
    for i in range(length - 1, n):
        out[i] = float(ratio[i - length + 1 : i + 1].mean())
    return out
