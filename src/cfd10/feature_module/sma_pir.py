"""SMA-based Position-In-Range (PIR) primitives — the cfd10 parity lynchpin.

This module is a faithful Python port of the parity shim in
``pine/speculatores_v15_presets_gold.pine`` lines 68-126. Pine's stateful
``ta.*`` built-ins share one history buffer per call site, so calling them inside
the variable-scale loop silently corrupts state; the Pine source works around
this with a single stateless ``ta.cum(close)`` and reconstructs SMA/ratio history
by plain arithmetic on historical access. We mirror that exactly.

Indexing convention (matching Pine ``series[back]``)
----------------------------------------------------
Pine evaluates everything at the *current* bar ``i`` and addresses history with a
non-negative offset: ``series[0]`` is bar ``i`` and ``series[back]`` is bar
``i - back``. All helpers here take an explicit absolute bar index ``i`` and an
offset ``back``; an array lookup at ``i - back`` reproduces ``series[back]``.

Cumulative-sum boundary
-----------------------
``sma_at`` evaluates ``(csum[i-back] - csum[i-back-s]) / s``. The lower index can
fall to ``-1`` at the very start of the series; per the task contract that single
boundary index is treated as ``0.0`` (yielding the full inclusive-prefix mean for
the first complete window), while any index ``< -1`` is out of range and yields
``NaN`` — the Pine ``na`` propagation.

Each public scalar function has a vectorized (Numba-JIT) full-series counterpart;
:mod:`tests.feature_module.test_sma_pir` pins them to agree to ``1e-9``.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "csum_close",
    "sma_at",
    "pir_for_scale",
    "pir_of_series",
    "agreement",
    "pir_for_scale_series",
    "AgreementResult",
]

# Public alias for the agreement return tuple: (scales_high, scales_low, n,
# agree_high, agree_low). Kept as a plain tuple for cheap interop with the
# vectorized layer and to mirror Pine's positional return on L126.
AgreementResult = tuple[int, int, int, float, float]

_NAN: float = float("nan")


def csum_close(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the inclusive prefix sum of ``close`` (Pine ``ta.cum``, L87).

    Args:
        close: 1-D array of close prices.

    Returns:
        Array ``csum`` with ``csum[i] = sum(close[0..i])`` and the same dtype
        contract as a ``float64`` cumulative sum.
    """
    arr = np.ascontiguousarray(close, dtype=np.float64)
    return np.cumsum(arr)


# --------------------------------------------------------------------------- #
# Numba kernels (module scope so the JIT cache is shared across calls).        #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _csum_lookup(csum: NDArray[np.float64], idx: int) -> float:
    """Return ``csum[idx]`` with the Pine boundary rule applied.

    Index ``-1`` is the ``0.0`` boundary; any index ``< -1`` or ``>= n`` is out
    of range and returns ``NaN``.
    """
    if idx == -1:
        return 0.0
    n = csum.shape[0]
    if idx < -1 or idx >= n:
        return np.nan
    return csum[idx]


@njit(cache=True, fastmath=False)
def _sma_at(csum: NDArray[np.float64], s: int, back: int, i: int) -> float:
    """Scalar Pine ``sma_at`` (L90-93): ``(csum[i-back] - csum[i-back-s]) / s``."""
    a = _csum_lookup(csum, i - back)
    b = _csum_lookup(csum, i - back - s)
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return (a - b) / s


@njit(cache=True, fastmath=False)
def _pir_for_scale(
    close: NDArray[np.float64],
    csum: NDArray[np.float64],
    s: int,
    lb: int,
    i: int,
) -> float:
    """Scalar Pine ``pir_for_scale`` (L98-113).

    Scans ``close / sma_at(s, back)`` over ``back`` in ``0..lb-1`` (bars ``i`` down
    to ``i-lb+1``) and returns the position of the current ratio within the
    [min, max] of that ratio history. Returns ``0.5`` when the window is flat or
    the current SMA is non-positive.
    """
    sma_now = _sma_at(csum, s, 0, i)
    val_now = close[i] / sma_now if sma_now > 0.0 else 1.0
    result = 0.5
    if not np.isnan(val_now):
        lo = val_now
        hi = val_now
        for back in range(1, lb):
            sma_b = _sma_at(csum, s, back, i)
            j = i - back
            c_b = close[j] if j >= 0 else np.nan
            r_b = c_b / sma_b if sma_b > 0.0 else 1.0
            if not np.isnan(r_b):
                if r_b < lo:
                    lo = r_b
                if r_b > hi:
                    hi = r_b
        result = (val_now - lo) / (hi - lo) if hi != lo else 0.5
    return result


@njit(cache=True, fastmath=False)
def _pir_for_scale_series(
    close: NDArray[np.float64],
    csum: NDArray[np.float64],
    s: int,
    lb: int,
) -> NDArray[np.float64]:
    """Vectorized full-series ``pir_for_scale`` for one scale ``s``.

    Returns an array ``out`` with ``out[i] = pir_for_scale(s, lb, i)`` for every
    bar ``i``. Computed bar-by-bar in a JIT loop (the inner ratio scan has a data
    dependency on ``i - back``, so this is the natural shape) and is the
    speed-oriented counterpart unit-tested against :func:`pir_for_scale`.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _pir_for_scale(close, csum, s, lb, i)
    return out


@njit(cache=True, fastmath=False)
def _pir_of_series(arr: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Vectorized Pine ``pir_of`` (L68-71) over a trailing ``lookback`` window.

    ``out[i]`` is the position of ``arr[i]`` within the [min, max] of the window
    ``arr[max(0, i-lookback+1) .. i]`` (inclusive of the current bar, matching
    Pine ``ta.lowest``/``ta.highest``). Flat windows map to ``0.5``.
    """
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i - lookback + 1
        if start < 0:
            start = 0
        lo = arr[i]
        hi = arr[i]
        for j in range(start, i + 1):
            v = arr[j]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        out[i] = (arr[i] - lo) / (hi - lo) if hi != lo else 0.5
    return out


@njit(cache=True, fastmath=False)
def _agreement(
    close: NDArray[np.float64],
    csum: NDArray[np.float64],
    scale_start: int,
    scale_end: int,
    scale_step: int,
    pct_extreme: float,
    i: int,
) -> tuple[int, int, int, float, float]:
    """Scalar Pine ``calc_agreement`` (L115-126) at bar ``i``.

    Iterates ``s`` over ``range(scale_start, scale_end + 1, scale_step)`` with
    ``lb = max(s, 20)``, counting scales whose ``pir_for_scale`` exceeds
    ``pct_extreme`` (high) or falls below ``1 - pct_extreme`` (low).
    """
    n_scales = 0
    scales_high = 0
    scales_low = 0
    s = scale_start
    while s <= scale_end:
        lb = s if s > 20 else 20
        pir_s = _pir_for_scale(close, csum, s, lb, i)
        if pir_s > pct_extreme:
            scales_high += 1
        if pir_s < (1.0 - pct_extreme):
            scales_low += 1
        n_scales += 1
        s += scale_step
    n_scales_f = n_scales if n_scales > 1 else 1
    return (
        scales_high,
        scales_low,
        n_scales,
        scales_high / n_scales_f,
        scales_low / n_scales_f,
    )


# --------------------------------------------------------------------------- #
# Public thin wrappers (validate inputs, then dispatch to the JIT kernels).    #
# --------------------------------------------------------------------------- #


def sma_at(csum: NDArray[np.float64], s: int, back: int, i: int) -> float:
    """Mean of the ``s`` closes ending at bar ``i - back`` (Pine ``sma_at``, L90).

    Args:
        csum: Inclusive prefix sum from :func:`csum_close`.
        s: SMA window length (number of closes), must be positive.
        back: Historical offset; bar evaluated is ``i - back``.
        i: Absolute current bar index.

    Returns:
        ``(csum[i-back] - csum[i-back-s]) / s``, with the ``-1`` boundary treated
        as ``0.0`` and out-of-range lookups yielding ``NaN``.
    """
    return float(_sma_at(np.ascontiguousarray(csum, dtype=np.float64), s, back, i))


def pir_for_scale(
    close: NDArray[np.float64],
    csum: NDArray[np.float64],
    s: int,
    lb: int,
    i: int,
) -> float:
    """Position of the current ``close/SMA(s)`` ratio in its ``lb``-bar range.

    Faithful port of Pine ``pir_for_scale`` (L98-113). See module docstring.

    Args:
        close: 1-D close-price array.
        csum: Inclusive prefix sum from :func:`csum_close`.
        s: SMA scale (window length).
        lb: Ratio-history lookback (number of bars scanned, including ``i``).
        i: Absolute current bar index.

    Returns:
        A value in ``[0, 1]``; ``0.5`` when the ratio window is flat or the
        current SMA is non-positive.
    """
    close_c = np.ascontiguousarray(close, dtype=np.float64)
    csum_c = np.ascontiguousarray(csum, dtype=np.float64)
    return float(_pir_for_scale(close_c, csum_c, s, lb, i))


def pir_for_scale_series(
    close: NDArray[np.float64],
    csum: NDArray[np.float64],
    s: int,
    lb: int,
) -> NDArray[np.float64]:
    """Vectorized full-series :func:`pir_for_scale` for one scale ``s``.

    Args:
        close: 1-D close-price array.
        csum: Inclusive prefix sum from :func:`csum_close`.
        s: SMA scale (window length).
        lb: Ratio-history lookback.

    Returns:
        Array ``out`` with ``out[i] = pir_for_scale(close, csum, s, lb, i)``.
    """
    close_c = np.ascontiguousarray(close, dtype=np.float64)
    csum_c = np.ascontiguousarray(csum, dtype=np.float64)
    return _pir_for_scale_series(close_c, csum_c, s, lb)


def pir_of_series(arr: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Trailing-window position-in-range of ``arr`` (Pine ``pir_of``, L68-71).

    Args:
        arr: 1-D series.
        lookback: Trailing window length (bars, including the current bar).

    Returns:
        Array of positions in ``[0, 1]``; ``0.5`` where the window is flat.
    """
    return _pir_of_series(np.ascontiguousarray(arr, dtype=np.float64), lookback)


def agreement(
    close: NDArray[np.float64],
    csum: NDArray[np.float64],
    scale_start: int,
    scale_end: int,
    scale_step: int,
    pct_extreme: float,
    i: int,
) -> AgreementResult:
    """Multi-scale agreement counts at bar ``i`` (Pine ``calc_agreement``, L115).

    Args:
        close: 1-D close-price array.
        csum: Inclusive prefix sum from :func:`csum_close`.
        scale_start: First SMA scale (inclusive).
        scale_end: Last SMA scale (inclusive).
        scale_step: Stride between scales.
        pct_extreme: High threshold; the low threshold is ``1 - pct_extreme``.
        i: Absolute current bar index.

    Returns:
        ``(scales_high, scales_low, n, agree_high, agree_low)`` where the agree
        fractions divide the counts by ``max(n, 1)``.
    """
    close_c = np.ascontiguousarray(close, dtype=np.float64)
    csum_c = np.ascontiguousarray(csum, dtype=np.float64)
    high, low, n, ah, al = _agreement(
        close_c, csum_c, scale_start, scale_end, scale_step, pct_extreme, i
    )
    return int(high), int(low), int(n), float(ah), float(al)
