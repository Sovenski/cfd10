"""Kaufman Efficiency Ratio (ER) — faithful port of the Pine ER gate.

This module ports ``pine/speculatores_v15_presets_gold.pine`` lines 468-480, the
per-side Efficiency Ratio that feeds the ER gate (``er_gate_ok_*``). The Pine
source computes, at the current bar ``t`` for a window ``p`` (``er_period``):

* ``er_path = Sigma_{i=0}^{p-1} |close[i] - close[i+1]|`` — the total path length,
  i.e. the sum of ``p`` consecutive one-bar absolute moves ending at the most
  recent move ``|close[t] - close[t-1]|`` (Pine ``close[i]`` is bar ``t - i``).
* ``er_net = close - close[p]`` when ``er_directional`` else ``|close - close[p]|``
  — the net displacement over the same window.
* ``er_val = er_path > 0 ? er_net / er_path : 0.0`` — the ratio; a flat window
  (zero path) maps to ``0.0``, not ``NaN``.

Indexing convention (matching Pine ``series[back]``)
----------------------------------------------------
Pine evaluates at the current bar ``t`` and addresses history with a
non-negative offset: ``series[0]`` is bar ``t`` and ``series[back]`` is bar
``t - back``. The oldest path term (``i = p - 1``) reaches ``close[t - p]``, so a
full window needs bars ``t - p .. t``: the first valid bar is ``t = p`` and bars
``0 .. p - 1`` are warm-up (returned as ``NaN``, mirroring Pine ``na``
propagation through the path sum).

Range
-----
By the triangle inequality ``|er_net| <= er_path``, so directional ER lies in
``[-1, 1]`` and absolute ER in ``[0, 1]`` on every valid bar.

Implementation
--------------
:func:`efficiency_ratio_scalar` is the literal Pine loop at a single bar (the
parity reference). :func:`efficiency_ratio` is the vectorized full-series
counterpart: it forms the absolute one-bar-move series once, takes its inclusive
prefix sum, and extracts each window's path in O(1). :mod:`tests.feature_module.
test_efficiency` pins the two to agree to ``1e-9``.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "efficiency_ratio",
    "efficiency_ratio_scalar",
]

_NAN: float = float("nan")


def _validate_period(period: int) -> None:
    """Raise ``ValueError`` if ``period`` is not a positive integer.

    Args:
        period: Efficiency-ratio window length (number of one-bar moves).

    Raises:
        ValueError: If ``period`` is not strictly positive.
    """
    if period < 1:
        raise ValueError(f"efficiency_ratio: period must be >= 1, got {period}")


# --------------------------------------------------------------------------- #
# Numba kernels (module scope so the JIT cache is shared across calls).        #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _efficiency_ratio_scalar(
    close: NDArray[np.float64],
    period: int,
    directional: bool,
    i: int,
) -> float:
    """Scalar Pine ER at bar ``i`` (L468-480), the parity reference.

    Returns ``NaN`` when ``i < period`` (insufficient history for the full path
    window) and otherwise the Pine ``er_val``: ``er_net / er_path`` with a zero
    path mapped to ``0.0``.
    """
    n = close.shape[0]
    if i < 0 or i >= n:
        return np.nan
    if i < period:
        return np.nan
    # er_path = sum_{k=0}^{period-1} |close[i - k] - close[i - k - 1]|.
    path = 0.0
    for k in range(period):
        path += abs(close[i - k] - close[i - k - 1])
    net = close[i] - close[i - period]
    if not directional:
        net = abs(net)
    if path > 0.0:
        return net / path
    return 0.0


@njit(cache=True, fastmath=False)
def _efficiency_ratio_series(
    close: NDArray[np.float64],
    abs_move_csum: NDArray[np.float64],
    period: int,
    directional: bool,
) -> NDArray[np.float64]:
    """Vectorized full-series ER.

    ``abs_move_csum`` is the inclusive prefix sum of the absolute one-bar-move
    series ``m`` where ``m[j] = |close[j] - close[j-1]|`` for ``j >= 1`` and
    ``m[0] = 0``. The path over the window ending at bar ``i`` is the sum of
    ``m[i - period + 1 .. i]`` = ``abs_move_csum[i] - abs_move_csum[i - period]``.

    Returns an array ``out`` with ``out[i] = efficiency_ratio_scalar(close,
    period, directional, i)`` for every bar ``i``.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < period:
            out[i] = np.nan
            continue
        path = abs_move_csum[i] - abs_move_csum[i - period]
        net = close[i] - close[i - period]
        if not directional:
            net = abs(net)
        out[i] = net / path if path > 0.0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# Public thin wrappers (validate inputs, then dispatch to the JIT kernels).    #
# --------------------------------------------------------------------------- #


def efficiency_ratio_scalar(
    close: NDArray[np.float64],
    period: int,
    directional: bool,
    i: int,
) -> float:
    """Kaufman Efficiency Ratio at a single bar ``i`` (Pine L468-480).

    This is the literal Pine loop and serves as the parity reference for
    :func:`efficiency_ratio`.

    Args:
        close: 1-D close-price array.
        period: Window length ``er_period`` (number of one-bar moves), ``>= 1``.
        directional: If ``True`` use the signed net displacement
            ``close[i] - close[i - period]`` (range ``[-1, 1]``); if ``False`` use
            its absolute value (range ``[0, 1]``).
        i: Absolute current bar index.

    Returns:
        The Pine ``er_val`` at bar ``i``: ``er_net / er_path`` with a zero path
        mapped to ``0.0``, or ``NaN`` when ``i < period`` (warm-up) or ``i`` is
        out of range.

    Raises:
        ValueError: If ``period`` is not strictly positive.
    """
    _validate_period(period)
    close_c = np.ascontiguousarray(close, dtype=np.float64)
    return float(_efficiency_ratio_scalar(close_c, period, directional, i))


def efficiency_ratio(
    close: NDArray[np.float64],
    period: int,
    directional: bool,
) -> NDArray[np.float64]:
    """Vectorized full-series Kaufman Efficiency Ratio (Pine L468-480).

    Args:
        close: 1-D close-price array.
        period: Window length ``er_period`` (number of one-bar moves), ``>= 1``.
        directional: If ``True`` use the signed net displacement (range
            ``[-1, 1]``); if ``False`` use its absolute value (range ``[0, 1]``).

    Returns:
        A ``float64`` array aligned to ``close`` with the per-bar ``er_val``. The
        first ``period`` bars are warm-up and set to ``NaN``; a zero-path window
        yields ``0.0``.

    Raises:
        ValueError: If ``period`` is not strictly positive.
    """
    _validate_period(period)
    close_c = np.ascontiguousarray(close, dtype=np.float64)
    # Absolute one-bar-move series m[j] = |close[j] - close[j-1]|, m[0] = 0.
    moves = np.empty_like(close_c)
    moves[0] = 0.0
    np.abs(np.diff(close_c), out=moves[1:])
    abs_move_csum = np.cumsum(moves)
    return _efficiency_ratio_series(close_c, abs_move_csum, period, directional)
