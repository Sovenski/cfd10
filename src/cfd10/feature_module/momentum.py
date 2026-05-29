"""Per-side momentum and momentum-velocity features (Pine parity, L443-453).

Faithful Python port of the momentum block in
``pine/speculatores_v15_presets_gold.pine`` lines 443-453, generalised over a
single lookback ``L`` (the Pine source instantiates it twice as
``mom_lookback_high`` / ``mom_lookback_low``):

.. code-block:: text

    price_ret = nz((close - close[L]) / close[L])                      // L443/449
    vol_ret   = nz((volume - volume[L]) / math.max(nz(volume[L]), 1))  // L444/450
    mom_diverge     = price_ret * vol_ret                              // L445/451
    momentum_velocity = nz(price_ret - price_ret[1])                   // L446/452

Indexing convention (matching Pine ``series[back]``)
----------------------------------------------------
Pine evaluates at the current bar ``i`` and addresses history with a
non-negative offset, so ``close[L]`` is ``close_array[i - L]``. A bar ``i`` is a
*warm-up* bar when ``i < L`` (no ``close[L]`` exists).

Warm-up contract (deliberate divergence from Pine ``nz``)
---------------------------------------------------------
Pine wraps each formula in ``nz(...)``, which turns the ``na`` produced on
insufficient history into ``0.0``. This feature layer instead surfaces warm-up
bars as ``NaN`` so the assembly step can decide where to re-apply ``nz``; this
mirrors the NaN-on-out-of-range policy already used in
:mod:`cfd10.feature_module.sma_pir`. Concretely:

* :func:`price_return` and :func:`mom_divergence` are ``NaN`` for ``i < L``.
* :func:`mom_velocity` needs two consecutive valid returns, so it is ``NaN`` for
  ``i < L + 1``.

The ``math.max(nz(volume[L]), 1)`` denominator clamp is reproduced exactly on
the valid (non-warm-up) bars: a genuine zero historical volume yields a
denominator of ``1`` rather than a division by zero.

Each public vectorized function has a scalar reference counterpart
(``_*_scalar``); :mod:`tests.feature_module.test_momentum` pins them to agree to
``1e-9``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "price_return",
    "mom_divergence",
    "mom_velocity",
]

_NAN: float = float("nan")


def _as_f64(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``arr`` as a contiguous 1-D ``float64`` array.

    Args:
        arr: Input series.

    Returns:
        A C-contiguous ``float64`` view/copy of ``arr``.

    Raises:
        ValueError: If ``arr`` is not one-dimensional.
    """
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {out.shape}")
    return out


def _check_lookback(length: int, lookback: int) -> None:
    """Validate the lookback against the series length.

    Args:
        length: Number of bars in the series.
        lookback: Requested lookback ``L``.

    Raises:
        ValueError: If ``lookback`` is not a positive integer.
    """
    if lookback < 1:
        raise ValueError(f"lookback L must be a positive integer, got {lookback}")
    if lookback >= length:
        logger.warning(
            "lookback L=%d >= series length %d: output is all-NaN (pure warm-up)",
            lookback,
            length,
        )


# --------------------------------------------------------------------------- #
# Vectorized implementations (public API).                                     #
# --------------------------------------------------------------------------- #


def price_return(close: NDArray[np.float64], L: int) -> NDArray[np.float64]:
    """Trailing price return ``(close - close[L]) / close[L]`` (Pine L443/449).

    Args:
        close: 1-D close-price array.
        L: Lookback in bars (``mom_lookback`` in Pine); must be positive.

    Returns:
        ``float64`` array aligned to ``close``; the first ``L`` bars are ``NaN``
        (warm-up) and bar ``i >= L`` holds ``(close[i] - close[i-L]) / close[i-L]``.
    """
    c = _as_f64(close)
    n = c.shape[0]
    _check_lookback(n, L)

    out = np.full(n, _NAN, dtype=np.float64)
    if L < n:
        cur = c[L:]
        back = c[:-L]
        out[L:] = (cur - back) / back
    return out


def mom_divergence(
    close: NDArray[np.float64], volume: NDArray[np.float64], L: int
) -> NDArray[np.float64]:
    """Momentum divergence ``price_ret * vol_ret`` (Pine L444-445 / L450-451).

    ``vol_ret = (volume - volume[L]) / max(volume[L], 1)``. The denominator clamp
    to ``1`` reproduces Pine ``math.max(nz(volume[L]), 1)`` and guards against a
    zero historical volume.

    Args:
        close: 1-D close-price array.
        volume: 1-D volume array, same length as ``close``.
        L: Lookback in bars; must be positive.

    Returns:
        ``float64`` array aligned to the inputs; the first ``L`` bars are ``NaN``.

    Raises:
        ValueError: If ``close`` and ``volume`` differ in length.
    """
    c = _as_f64(close)
    v = _as_f64(volume)
    if c.shape[0] != v.shape[0]:
        raise ValueError(
            f"close and volume length mismatch: {c.shape[0]} vs {v.shape[0]}"
        )
    n = c.shape[0]
    _check_lookback(n, L)

    out = np.full(n, _NAN, dtype=np.float64)
    if L < n:
        price_ret = (c[L:] - c[:-L]) / c[:-L]
        vol_back = v[:-L]
        denom = np.maximum(vol_back, 1.0)
        vol_ret = (v[L:] - vol_back) / denom
        out[L:] = price_ret * vol_ret
    return out


def mom_velocity(close: NDArray[np.float64], L: int) -> NDArray[np.float64]:
    """Momentum velocity ``price_ret - price_ret[1]`` (Pine L446/452).

    This is the bar-over-bar change of :func:`price_return`. Because it consumes
    two consecutive valid returns, the first ``L + 1`` bars are ``NaN``.

    Args:
        close: 1-D close-price array.
        L: Lookback in bars; must be positive.

    Returns:
        ``float64`` array aligned to ``close``; bar ``i >= L + 1`` holds
        ``price_return[i] - price_return[i-1]``.
    """
    price_ret = price_return(close, L)
    out = np.full(price_ret.shape[0], _NAN, dtype=np.float64)
    # np.diff would propagate the warm-up NaN at index L into index L (NaN - NaN),
    # which is exactly the desired warm-up extension to L + 1.
    out[1:] = price_ret[1:] - price_ret[:-1]
    return out


# --------------------------------------------------------------------------- #
# Scalar references (parity anchors; pinned to the vectorized API to 1e-9).    #
# --------------------------------------------------------------------------- #


def _price_return_scalar(close: NDArray[np.float64], L: int, i: int) -> float:
    """Scalar Pine ``price_ret`` at bar ``i`` (warm-up -> ``NaN``)."""
    if i < L:
        return _NAN
    back = float(close[i - L])
    return (float(close[i]) - back) / back


def _mom_divergence_scalar(
    close: NDArray[np.float64], volume: NDArray[np.float64], L: int, i: int
) -> float:
    """Scalar Pine ``mom_diverge`` at bar ``i`` (warm-up -> ``NaN``)."""
    if i < L:
        return _NAN
    c_back = float(close[i - L])
    price_ret = (float(close[i]) - c_back) / c_back
    v_back = float(volume[i - L])
    vol_ret = (float(volume[i]) - v_back) / max(v_back, 1.0)
    return price_ret * vol_ret


def _mom_velocity_scalar(close: NDArray[np.float64], L: int, i: int) -> float:
    """Scalar Pine ``momentum_velocity`` at bar ``i`` (warm-up -> ``NaN``)."""
    if i < L + 1:
        return _NAN
    return _price_return_scalar(close, L, i) - _price_return_scalar(close, L, i - 1)
