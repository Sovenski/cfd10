"""Overextension / volatility-regime features — classic top "tells".

This module supplies the SHORT/TOP specialist with a block of *dimensionless,
bounded* features that quantify how stretched, overbought, and how far into a
volatility regime a price series has run. Unlike the symmetric structural oracle
(which marks sharp V-bottoms cleanly but rounded distribution-tops poorly), these
features describe tops on their own terms: persistent extension above the long
mean, shallow drawdown from the running high, and an elevated / accelerating
volatility regime.

Features
--------
``dist_above_sma_z(close, length, z_win)``
    ``tanh( (close - SMA(close, length)) / rolling_std(close, z_win) )`` — a
    bounded ``(-1, 1)`` measure of how many trailing standard deviations the
    close sits above its long mean (overextension; high near tops).
``drawdown_from_high(close, lookback)``
    ``(close - rolling_max(close, lookback)) / rolling_max(close, lookback)`` —
    always ``<= 0``; near ``0`` at a fresh high (i.e. at tops), strongly negative
    after a sell-off.
``realized_vol_pct(close, win, range_len)``
    Trailing position-in-range over ``range_len`` of the rolling realized
    volatility (population std of log returns over ``win``); a ``[0, 1]``
    vol-regime percentile (reuses :func:`cfd10.feature_module.sma_pir.pir_of_series`).
``up_streak_norm(close, cap)``
    ``min(consecutive up bars, cap) / cap`` — a ``[0, 1]`` overbought-persistence
    count (how many bars in a row closed up, saturating at ``cap``).
``vol_of_vol(close, win)``
    Position-in-range over ``win`` of the rolling std (window ``win``) of the
    rolling realized volatility (window ``win``) — a ``[0, 1]`` "is volatility
    itself unstable" regime tell.

Warm-up convention
------------------
Every feature mirrors the NaN-on-insufficient-history policy used across the
feature layer (see :mod:`cfd10.feature_module.sma_pir`): bars without a full
defining window are ``np.nan``. The first defined bar is documented per function.
``pir_of_series`` then propagates those leading ``NaN``s.

Scalar / vectorized parity
--------------------------
Each window statistic has a pure-Python scalar reference (``_*_scalar``) pinned
to the vectorized public function to ``1e-9`` in
:mod:`tests.feature_module.test_overextension`. The vectorized paths are plain
NumPy (sliding-window reductions); no Numba kernel is needed at these window
sizes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from cfd10.feature_module.sma_pir import pir_of_series
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "dist_above_sma_z",
    "drawdown_from_high",
    "realized_vol_pct",
    "up_streak_norm",
    "vol_of_vol",
    "OVEREXTENSION_FACTORY",
    "register_overextension",
    "get_overextension_fn",
    "OverextensionFn",
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


def _validate_window(name: str, window: int) -> None:
    """Raise ``ValueError`` if ``window`` is not a positive integer.

    Args:
        name: Parameter name, used in the error message.
        window: Requested window length.

    Raises:
        ValueError: If ``window`` is not strictly positive.
    """
    if window < 1:
        raise ValueError(f"{name} must be a positive integer, got {window}")


def _rolling_window_sums(
    arr: NDArray[np.float64], window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trailing window sums of ``arr`` and of its per-bar ``NaN`` mask.

    Returns ``(win_sum, nan_count)`` where, for ``i >= window - 1``, ``win_sum[i]``
    is the sum of ``arr[i - window + 1 .. i]`` with any ``NaN`` element treated as
    ``0`` and ``nan_count[i]`` is how many of those bars are ``NaN``; warm-up bars
    (``i < window - 1``) are ``NaN`` in both. Substituting ``NaN`` with ``0`` keeps
    the inclusive-prefix-sum arithmetic finite so a leading warm-up region in
    ``arr`` (e.g. the realized-vol band feeding :func:`vol_of_vol`) cannot poison
    every downstream bar via :func:`numpy.cumsum`; callers use ``nan_count`` to
    re-mask windows that actually contain a ``NaN``.
    """
    n = arr.shape[0]
    win_sum = np.full(n, _NAN, dtype=np.float64)
    nan_count = np.full(n, _NAN, dtype=np.float64)
    if window > n:
        return win_sum, nan_count
    is_nan = np.isnan(arr)
    filled = np.where(is_nan, 0.0, arr)
    csum = np.cumsum(filled)
    cnan = np.cumsum(is_nan.astype(np.float64))
    win_sum[window - 1] = csum[window - 1]
    nan_count[window - 1] = cnan[window - 1]
    if n > window:
        win_sum[window:] = csum[window:] - csum[:-window]
        nan_count[window:] = cnan[window:] - cnan[:-window]
    return win_sum, nan_count


def _rolling_mean(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Trailing simple moving average; the first ``window - 1`` bars are ``NaN``.

    ``out[i]`` is the mean of ``arr[i - window + 1 .. i]`` for ``i >= window - 1``,
    or ``NaN`` if that window contains any ``NaN`` (matching the scalar
    ``arr[i-w+1:i+1].mean()`` reference and propagating an upstream warm-up band).
    Computed from an inclusive prefix sum in O(n); for an all-finite window this is
    numerically identical to the per-window mean.
    """
    n = arr.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    if window > n:
        return out
    win_sum, nan_count = _rolling_window_sums(arr, window)
    clean = (nan_count == 0.0)  # NaN warm-up bars compare False -> stay NaN.
    out[clean] = win_sum[clean] / window
    return out


def _rolling_std(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Trailing population (biased, divide-by-``N``) std; warm-up bars are ``NaN``.

    ``out[i]`` is the population standard deviation of ``arr[i - window + 1 .. i]``
    for ``i >= window - 1``. Uses ``E[x^2] - E[x]^2`` via prefix sums, with a
    ``max(., 0)`` clamp so floating-point cancellation cannot yield a tiny
    negative variance (and hence a ``NaN`` square root).
    """
    n = arr.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    if window > n:
        return out
    mean = _rolling_mean(arr, window)
    mean_sq = _rolling_mean(arr * arr, window)
    var = mean_sq - mean * mean
    valid = ~np.isnan(var)
    var[valid] = np.maximum(var[valid], 0.0)
    out[valid] = np.sqrt(var[valid])
    return out


def _rolling_max(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Trailing running maximum; the first ``window - 1`` bars are ``NaN``.

    ``out[i]`` is ``max(arr[i - window + 1 .. i])`` for ``i >= window - 1``
    (inclusive of the current bar, matching Pine ``ta.highest``).
    """
    n = arr.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    if window > n:
        return out
    for i in range(window - 1, n):
        out[i] = arr[i - window + 1 : i + 1].max()
    return out


def _log_returns(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """One-bar log returns ``ln(close[i] / close[i - 1])``; ``out[0]`` is ``NaN``.

    Non-positive closes (degenerate inputs) propagate ``NaN`` rather than raising,
    keeping the feature robust on pathological synthetic series.
    """
    n = close.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    if n < 2:
        return out
    prev = close[:-1]
    cur = close[1:]
    safe = (prev > 0.0) & (cur > 0.0)
    ratio = np.where(safe, cur / np.where(safe, prev, 1.0), _NAN)
    out[1:] = np.where(safe, np.log(ratio), _NAN)
    return out


def _realized_vol(close: NDArray[np.float64], win: int) -> NDArray[np.float64]:
    """Rolling realized volatility: population std of log returns over ``win``.

    The first log return is at index 1, so the first defined volatility bar is at
    index ``win`` (a full window of ``win`` returns covers bars ``1 .. win``).
    Earlier bars are ``NaN``.
    """
    rets = _log_returns(close)
    n = close.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)
    if n < win + 1:
        return out
    # Returns live at indices 1..n-1; std over a trailing win-window of returns.
    rets_valid = rets[1:]
    vol = _rolling_std(rets_valid, win)
    out[1:] = vol
    return out


# --------------------------------------------------------------------------- #
# Public features.                                                            #
# --------------------------------------------------------------------------- #


def dist_above_sma_z(
    close: NDArray[np.float64], length: int, z_win: int
) -> NDArray[np.float64]:
    """Squashed z-distance of ``close`` above its long SMA (overextension).

    Computes ``z[i] = (close[i] - SMA(close, length)[i]) / rolling_std(close,
    z_win)[i]`` and returns ``tanh(z)``, a bounded ``(-1, 1)`` measure of how many
    trailing standard deviations the close sits above its long mean. High (toward
    ``+1``) when the price is stretched well above the mean — a classic top tell.

    Args:
        close: 1-D close-price array.
        length: SMA window for the long mean; positive.
        z_win: Trailing window for the normalising standard deviation; positive.

    Returns:
        ``float64`` array aligned to ``close`` in ``(-1, 1)``. Warm-up bars
        (``i < max(length, z_win) - 1``) are ``NaN``. A bar whose ``z_win`` window
        is perfectly flat (zero std) maps to ``0.0`` (no overextension signal).
    """
    c = _as_f64(close)
    _validate_window("length", length)
    _validate_window("z_win", z_win)
    n = c.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)

    sma = _rolling_mean(c, length)
    std = _rolling_std(c, z_win)
    defined = ~np.isnan(sma) & ~np.isnan(std)
    nz_std = defined & (std > 0.0)
    z = np.zeros(n, dtype=np.float64)
    z[nz_std] = (c[nz_std] - sma[nz_std]) / std[nz_std]
    out[defined] = np.tanh(z[defined])  # flat-window bars keep z = 0 -> tanh 0.
    return out


def drawdown_from_high(
    close: NDArray[np.float64], lookback: int
) -> NDArray[np.float64]:
    """Drawdown of ``close`` from its trailing running high (``<= 0``).

    ``out[i] = (close[i] - H[i]) / H[i]`` where ``H = rolling_max(close,
    lookback)``. The value is ``0`` at a fresh high (the running max equals the
    current close — i.e. *at tops*) and increasingly negative after a decline.

    Args:
        close: 1-D close-price array (assumed positive, as for prices).
        lookback: Trailing window for the running high; positive.

    Returns:
        ``float64`` array aligned to ``close``, ``<= 0`` where defined. Warm-up
        bars (``i < lookback - 1``) are ``NaN``. A non-positive running high maps
        to ``0.0`` (degenerate guard against division by zero).
    """
    c = _as_f64(close)
    _validate_window("lookback", lookback)
    n = c.shape[0]
    out = np.full(n, _NAN, dtype=np.float64)

    roll_max = _rolling_max(c, lookback)
    defined = ~np.isnan(roll_max)
    pos = defined & (roll_max > 0.0)
    out[defined] = 0.0  # non-positive high (degenerate) -> 0.0.
    out[pos] = (c[pos] - roll_max[pos]) / roll_max[pos]
    return out


def realized_vol_pct(
    close: NDArray[np.float64], win: int, range_len: int
) -> NDArray[np.float64]:
    """Trailing position-in-range of rolling realized volatility (vol regime).

    Realized volatility is the population std of one-bar log returns over ``win``;
    it is then mapped through
    :func:`cfd10.feature_module.sma_pir.pir_of_series` over ``range_len`` to a
    ``[0, 1]`` percentile of the recent vol distribution. High (toward ``1``) when
    the current regime is among the most volatile of the trailing window.

    Args:
        close: 1-D close-price array.
        win: Window for the rolling realized volatility; positive.
        range_len: Trailing window for the position-in-range; positive.

    Returns:
        ``float64`` array in ``[0, 1]`` where defined (``0.5`` on a flat vol
        window), aligned to ``close``. The realized-vol warm-up (``i < win``)
        propagates as ``NaN`` through the position-in-range.
    """
    c = _as_f64(close)
    _validate_window("win", win)
    _validate_window("range_len", range_len)
    vol = _realized_vol(c, win)
    return pir_of_series(vol, int(range_len))


def up_streak_norm(close: NDArray[np.float64], cap: int) -> NDArray[np.float64]:
    """Normalised consecutive-up-bar streak (overbought persistence), in ``[0, 1]``.

    Counts how many consecutive bars closed strictly higher than the previous bar
    (the run ending at bar ``i``), saturates that count at ``cap``, and divides by
    ``cap``. ``0`` after a down/flat bar; ``1`` once the up-run reaches ``cap``
    bars.

    Args:
        close: 1-D close-price array.
        cap: Saturation count (maximum streak that maps to ``1.0``); positive.

    Returns:
        ``float64`` array in ``[0, 1]`` aligned to ``close``. Bar ``0`` has no
        predecessor and is defined as ``0.0`` (no up-move yet); there is no
        multi-bar warm-up.
    """
    c = _as_f64(close)
    _validate_window("cap", cap)
    n = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    streak = 0
    for i in range(1, n):
        if c[i] > c[i - 1]:
            streak += 1
        else:
            streak = 0
        capped = streak if streak < cap else cap
        out[i] = capped / cap
    return out


def vol_of_vol(close: NDArray[np.float64], win: int) -> NDArray[np.float64]:
    """Position-in-range of the volatility-of-volatility (regime instability).

    Takes the rolling realized volatility (window ``win``), then its rolling
    population std over ``win`` (the "vol of vol"), and maps that through
    :func:`cfd10.feature_module.sma_pir.pir_of_series` over ``win`` to a ``[0, 1]``
    percentile. High (toward ``1``) when volatility itself is unusually unstable —
    a regime tell that often accompanies tops.

    Args:
        close: 1-D close-price array.
        win: Shared window for the realized vol, its rolling std, and the
            position-in-range; positive.

    Returns:
        ``float64`` array in ``[0, 1]`` where defined (``0.5`` on a flat window),
        aligned to ``close``. The compounded warm-up of the two nested windows
        propagates as ``NaN``.
    """
    c = _as_f64(close)
    _validate_window("win", win)
    vol = _realized_vol(c, win)
    vov = _rolling_std(vol, win)
    return pir_of_series(vov, int(win))


# --------------------------------------------------------------------------- #
# Scalar references (parity anchors; pinned to the vectorized API to 1e-9).    #
# --------------------------------------------------------------------------- #


def _rolling_mean_scalar(arr: NDArray[np.float64], window: int, i: int) -> float:
    """Scalar trailing mean at bar ``i`` (warm-up -> ``NaN``)."""
    if i < window - 1:
        return _NAN
    return float(arr[i - window + 1 : i + 1].mean())


def _rolling_std_scalar(arr: NDArray[np.float64], window: int, i: int) -> float:
    """Scalar trailing population std at bar ``i`` (warm-up -> ``NaN``)."""
    if i < window - 1:
        return _NAN
    return float(arr[i - window + 1 : i + 1].std(ddof=0))


def _dist_above_sma_z_scalar(
    close: NDArray[np.float64], length: int, z_win: int, i: int
) -> float:
    """Scalar :func:`dist_above_sma_z` at bar ``i`` (warm-up -> ``NaN``)."""
    sma = _rolling_mean_scalar(close, length, i)
    std = _rolling_std_scalar(close, z_win, i)
    if np.isnan(sma) or np.isnan(std):
        return _NAN
    if std <= 0.0:
        return 0.0
    return float(np.tanh((float(close[i]) - sma) / std))


def _drawdown_from_high_scalar(
    close: NDArray[np.float64], lookback: int, i: int
) -> float:
    """Scalar :func:`drawdown_from_high` at bar ``i`` (warm-up -> ``NaN``)."""
    if i < lookback - 1:
        return _NAN
    high = float(close[i - lookback + 1 : i + 1].max())
    if high <= 0.0:
        return 0.0
    return (float(close[i]) - high) / high


def _up_streak_norm_scalar(close: NDArray[np.float64], cap: int, i: int) -> float:
    """Scalar :func:`up_streak_norm` at bar ``i`` (no warm-up; bar 0 is ``0.0``)."""
    streak = 0
    for k in range(1, i + 1):
        if close[k] > close[k - 1]:
            streak += 1
        else:
            streak = 0
    capped = streak if streak < cap else cap
    return capped / cap


# --------------------------------------------------------------------------- #
# Registry / factory for the per-series overextension feature functions.       #
# --------------------------------------------------------------------------- #

# An overextension feature: maps ``close`` plus its scalar window parameters to a
# full-length float64 series. Arity differs per function (e.g. one window vs two),
# so the parameter pack is left loose; callers supply the documented arguments.
OverextensionFn = Callable[..., NDArray[np.float64]]

OVEREXTENSION_FACTORY: dict[str, OverextensionFn] = {}

_F = TypeVar("_F", bound=OverextensionFn)


def register_overextension(name: str) -> Callable[[_F], _F]:
    """Register an overextension feature function under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator recording the function in :data:`OVEREXTENSION_FACTORY` and
        returning it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: _F) -> _F:
        if name in OVEREXTENSION_FACTORY:
            raise ValueError(
                f"register_overextension: duplicate registration for {name!r}"
            )
        OVEREXTENSION_FACTORY[name] = fn
        logger.debug("register_overextension: registered %s", name)
        return fn

    return decorator


def get_overextension_fn(name: str) -> OverextensionFn:
    """Resolve a registered overextension feature function by name.

    Args:
        name: Registry key used with :func:`register_overextension`.

    Returns:
        The registered callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return OVEREXTENSION_FACTORY[name]
    except KeyError as exc:
        known = ", ".join(sorted(OVEREXTENSION_FACTORY)) or "<empty>"
        raise KeyError(
            f"get_overextension_fn: unknown overextension function {name!r}; "
            f"known: {known}"
        ) from exc


register_overextension("dist_above_sma_z")(dist_above_sma_z)
register_overextension("drawdown_from_high")(drawdown_from_high)
register_overextension("realized_vol_pct")(realized_vol_pct)
register_overextension("up_streak_norm")(up_streak_norm)
register_overextension("vol_of_vol")(vol_of_vol)
