"""GJR-GARCH asymmetry and HAR / Garman-Klass volatility features.

Faithful Python port of the two volatility indicators in
``pine/speculatores_v15_presets_gold.pine``:

* ``calc_gjr_asym`` (lines 159-177) — a GJR-GARCH(1,1) recursion that runs two
  parallel variance processes, an asymmetric (leverage-aware) ``gjr_var`` and a
  symmetric ``sym_var``, on squared log-returns, and reports how far the
  asymmetric variance has diverged from the symmetric one.
* ``calc_har_vol`` (lines 179-188) — a Heterogeneous Auto-Regressive forecast
  built from daily / weekly / monthly Garman-Klass variance, normalised against
  the contemporaneous Garman-Klass volatility.

Pine recursion semantics
-------------------------
Pine evaluates one bar at a time and addresses history with ``series[back]``
(``series[1]`` is the previous bar). The state variables are declared with
``var float ... = na`` and reassigned every bar, so ``gjr_var[1]`` on the first
bar refers to the bar *before* the series start and is ``na``; ``nz(x, y)``
substitutes ``y`` there. We reproduce this exactly: the recursion carries the
previous ``gjr_var`` / ``sym_var`` forward, and on bar 0 (and only bar 0) the
previous values fall back to the long-run variance ``lr_var``.

Warm-up
-------
The normalisations divide by reference statistics that depend on a trailing
``ta.sma``:

* ``gjr_asym`` needs ``ta.sma(r2, 252)`` (the long-run variance estimate), which
  Pine returns as ``na`` until 252 bars exist. The recursive state still
  propagates underneath (so the first defined value matches Pine exactly), but
  the *output* for bars ``0..250`` is emitted as ``NaN``.
* ``har_vol`` needs ``ta.sma(gk, 22)`` (the monthly term); its output is ``NaN``
  for bars ``0..20``.

Both functions return ``float64`` arrays aligned to the input length. A scalar
reference and a Numba-vectorized kernel are kept side by side and pinned to
agree to ``1e-9`` in :mod:`tests.feature_module.test_garch_har`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "gjr_asym",
    "har_vol",
    "GJR_ALPHA",
    "GJR_BETA",
    "GJR_GAMMA",
    "VOL_FEATURE_FACTORY",
    "register_vol_feature",
    "get_vol_feature_fn",
    "VolFeatureFn",
]

# --------------------------------------------------------------------------- #
# Pine constants (L163-165, L162/L184 window lengths, L154 epsilons).          #
# --------------------------------------------------------------------------- #
GJR_ALPHA: Final[float] = 0.03
GJR_BETA: Final[float] = 0.90
GJR_GAMMA: Final[float] = 0.08

# ``ta.sma`` window lengths used by the two indicators.
_LR_VAR_WINDOW: Final[int] = 252  # long-run variance SMA for GJR (L162).
_HAR_WEEKLY_WINDOW: Final[int] = 5  # weekly GK term (L183).
_HAR_MONTHLY_WINDOW: Final[int] = 22  # monthly GK term (L184).

# Pine ``safe_log_ratio`` floor (L154) and the per-indicator variance floors.
_LOG_FLOOR: Final[float] = 1e-10
_GJR_VAR_FLOOR: Final[float] = 1e-12
_GK_VAR_FLOOR: Final[float] = 1e-10

# Normalisation denominators (L176 / L187) and the GK leverage-of-2 constant.
_GJR_NORM_DEN: Final[float] = 0.1
_HAR_NORM_DEN: Final[float] = 0.5
_LN2: Final[float] = float(np.log(2.0))

# HAR forecast weights (L185).
_HAR_W_DAILY: Final[float] = 0.36
_HAR_W_WEEKLY: Final[float] = 0.28
_HAR_W_MONTHLY: Final[float] = 0.28


# --------------------------------------------------------------------------- #
# Numba kernels (module scope so the JIT cache is shared across calls).        #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _safe_log_ratio(num: float, den: float) -> float:
    """Pine ``safe_log_ratio`` (L153-154): ``log(max(num,e)/max(den,e))``."""
    return np.log(max(num, _LOG_FLOOR) / max(den, _LOG_FLOOR))


@njit(cache=True, fastmath=False)
def _clamp_unit(val: float) -> float:
    """Pine ``clamp_unit`` (L156-157): clamp to ``[-1, 1]``."""
    return max(-1.0, min(1.0, val))


@njit(cache=True, fastmath=False)
def _trailing_sma_warmup(x: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    """Faithful ``ta.sma(x, length)``: NaN for the first ``length-1`` bars.

    Uses a running-sum sweep so the cost is linear in the series length. The
    first ``length-1`` positions are ``na`` (NaN) exactly as the Pine built-in
    returns during warm-up; from index ``length-1`` onward the value is the mean
    of the trailing ``length`` samples.
    """
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n == 0:
        return out
    running = 0.0
    for i in range(n):
        running += x[i]
        if i >= length:
            running -= x[i - length]
        if i >= length - 1:
            out[i] = running / length
    return out


@njit(cache=True, fastmath=False)
def _log_returns(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bar log-returns (L160): ``safe_log_ratio(close, nz(close[1], close))``.

    On bar 0 the previous close is ``na`` and ``nz`` substitutes the current
    close, giving a zero return.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        prev_close = close[i - 1] if i >= 1 else close[i]
        out[i] = _safe_log_ratio(close[i], prev_close)
    return out


@njit(cache=True, fastmath=False)
def _gjr_asym_kernel(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized full-series Pine ``calc_gjr_asym`` (L159-177).

    Runs the two-process variance recursion bar-by-bar (a strict data
    dependency on the previous bar's ``gjr_var`` / ``sym_var``) and writes the
    normalised asymmetry; warm-up bars where ``ta.sma(r2, 252)`` is still ``na``
    are left as NaN while the recursive state keeps propagating.
    """
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    log_ret = _log_returns(close)
    r2 = log_ret * log_ret
    sma_r2 = _trailing_sma_warmup(r2, _LR_VAR_WINDOW)

    omega_coef = 1.0 - GJR_ALPHA - GJR_BETA - GJR_GAMMA / 2.0
    prev_gjr = np.nan
    prev_sym = np.nan
    for i in range(n):
        sma_i = sma_r2[i]
        warmed = not np.isnan(sma_i)
        lr_var = max(sma_i if warmed else r2[i], _GJR_VAR_FLOOR)
        omega = max(lr_var * omega_coef, _GJR_VAR_FLOOR)

        pg = prev_gjr if not np.isnan(prev_gjr) else lr_var
        ps = prev_sym if not np.isnan(prev_sym) else lr_var
        pr2 = r2[i - 1] if i >= 1 else lr_var
        leverage = 1.0 if (i >= 1 and log_ret[i - 1] < 0.0) else 0.0

        gjr_var = max(
            omega + (GJR_ALPHA + GJR_GAMMA * leverage) * pr2 + GJR_BETA * pg,
            _GJR_VAR_FLOOR,
        )
        sym_var = max(
            omega + (GJR_ALPHA + GJR_GAMMA * 0.5) * pr2 + GJR_BETA * ps,
            _GJR_VAR_FLOOR,
        )
        if warmed:
            ratio = gjr_var / sym_var
            out[i] = _clamp_unit((ratio - 1.0) / _GJR_NORM_DEN)

        prev_gjr = gjr_var
        prev_sym = sym_var
    return out


@njit(cache=True, fastmath=False)
def _gk_variance(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Garman-Klass variance per bar (L180-182).

    ``gk = max(0.5*log(h/l)^2 - (2*ln2 - 1)*log(c/o)^2, 1e-10)``.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    co_coef = 2.0 * _LN2 - 1.0
    for i in range(n):
        log_hl = _safe_log_ratio(high[i], low[i])
        log_co = _safe_log_ratio(close[i], open_[i])
        out[i] = max(0.5 * log_hl * log_hl - co_coef * log_co * log_co, _GK_VAR_FLOOR)
    return out


@njit(cache=True, fastmath=False)
def _har_vol_kernel(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized full-series Pine ``calc_har_vol`` (L179-188).

    Builds the HAR forecast from daily / weekly / monthly Garman-Klass variance
    and normalises ``sqrt(forecast)/sqrt(gk)``. Output is NaN until the monthly
    (22-bar) SMA warms up; the weekly term falls back to ``gk`` via ``nz`` while
    it is still ``na`` (matching Pine L183).
    """
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    gk = _gk_variance(open_, high, low, close)
    sma_weekly = _trailing_sma_warmup(gk, _HAR_WEEKLY_WINDOW)
    sma_monthly = _trailing_sma_warmup(gk, _HAR_MONTHLY_WINDOW)

    for i in range(n):
        weekly = sma_weekly[i] if not np.isnan(sma_weekly[i]) else gk[i]
        monthly_warmed = not np.isnan(sma_monthly[i])
        monthly = sma_monthly[i] if monthly_warmed else gk[i]
        forecast = max(
            _HAR_W_DAILY * gk[i] + _HAR_W_WEEKLY * weekly + _HAR_W_MONTHLY * monthly,
            _GK_VAR_FLOOR,
        )
        if monthly_warmed:
            ratio = np.sqrt(max(forecast, _GK_VAR_FLOOR)) / np.sqrt(
                max(gk[i], _GK_VAR_FLOOR)
            )
            out[i] = _clamp_unit((ratio - 1.0) / _HAR_NORM_DEN)
    return out


# --------------------------------------------------------------------------- #
# Public thin wrappers (validate / coerce inputs, dispatch to the kernels).    #
# --------------------------------------------------------------------------- #


def _as_f64(name: str, arr: NDArray[np.float64], n: int | None) -> NDArray[np.float64]:
    """Coerce ``arr`` to a contiguous 1-D ``float64`` array of length ``n``.

    Args:
        name: Series name, used in error messages.
        arr: Input array-like.
        n: Required length, or ``None`` to accept any length (and report it).

    Returns:
        A contiguous ``float64`` view/copy of ``arr``.

    Raises:
        ValueError: If ``arr`` is not 1-D or its length differs from ``n``.
    """
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={out.ndim}")
    if n is not None and out.shape[0] != n:
        raise ValueError(f"{name} length {out.shape[0]} != expected {n}")
    return out


def gjr_asym(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GJR-GARCH asymmetry feature (Pine ``calc_gjr_asym``, L159-177).

    Runs a leverage-aware GJR-GARCH(1,1) variance recursion alongside a symmetric
    one on squared log-returns of ``close`` and returns the normalised divergence
    ``clamp((gjr_var/sym_var - 1) / 0.1, -1, 1)``. The leverage term switches on
    when the previous bar's log-return is negative.

    ``open_``, ``high`` and ``low`` are accepted for a uniform feature signature
    and length validation; the GJR recursion itself depends only on ``close``
    (matching the Pine source, which derives every term from ``close``).

    Args:
        open_: 1-D open-price array (validated for length, otherwise unused).
        high: 1-D high-price array (validated for length, otherwise unused).
        low: 1-D low-price array (validated for length, otherwise unused).
        close: 1-D close-price array driving the recursion.

    Returns:
        A ``float64`` array aligned to ``close``; entries are in ``[-1, 1]`` once
        the 252-bar long-run-variance SMA has warmed up and ``NaN`` before that.

    Raises:
        ValueError: If any input is not 1-D or the four arrays differ in length.
    """
    close_c = _as_f64("close", close, None)
    n = close_c.shape[0]
    _as_f64("open_", open_, n)
    _as_f64("high", high, n)
    _as_f64("low", low, n)
    return _gjr_asym_kernel(close_c)


def har_vol(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """HAR + Garman-Klass volatility feature (Pine ``calc_har_vol``, L179-188).

    Computes per-bar Garman-Klass variance
    ``gk = max(0.5*log(h/l)^2 - (2*ln2 - 1)*log(c/o)^2, 1e-10)``, forms the HAR
    forecast ``0.36*gk + 0.28*sma(gk,5) + 0.28*sma(gk,22)`` and returns
    ``clamp((sqrt(forecast)/sqrt(gk) - 1) / 0.5, -1, 1)``.

    Args:
        open_: 1-D open-price array.
        high: 1-D high-price array.
        low: 1-D low-price array.
        close: 1-D close-price array.

    Returns:
        A ``float64`` array aligned to the inputs; entries are in ``[-1, 1]`` once
        the 22-bar monthly SMA has warmed up and ``NaN`` before that.

    Raises:
        ValueError: If any input is not 1-D or the four arrays differ in length.
    """
    close_c = _as_f64("close", close, None)
    n = close_c.shape[0]
    open_c = _as_f64("open_", open_, n)
    high_c = _as_f64("high", high, n)
    low_c = _as_f64("low", low, n)
    return _har_vol_kernel(open_c, high_c, low_c, close_c)


# --------------------------------------------------------------------------- #
# Registry / factory for the volatility feature functions.                     #
# --------------------------------------------------------------------------- #

# A full-series volatility feature: (open, high, low, close) -> float64 array.
VolFeatureFn = Callable[
    [
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    NDArray[np.float64],
]

VOL_FEATURE_FACTORY: dict[str, VolFeatureFn] = {}


def register_vol_feature(name: str) -> Callable[[VolFeatureFn], VolFeatureFn]:
    """Register a volatility feature function under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator recording the function in :data:`VOL_FEATURE_FACTORY` and
        returning it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: VolFeatureFn) -> VolFeatureFn:
        if name in VOL_FEATURE_FACTORY:
            raise ValueError(
                f"register_vol_feature: duplicate registration for {name!r}"
            )
        VOL_FEATURE_FACTORY[name] = fn
        logger.debug("register_vol_feature: registered %s", name)
        return fn

    return decorator


def get_vol_feature_fn(name: str) -> VolFeatureFn:
    """Resolve a registered volatility feature function by name.

    Args:
        name: Registry key used with :func:`register_vol_feature`.

    Returns:
        The registered callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return VOL_FEATURE_FACTORY[name]
    except KeyError as exc:
        known = ", ".join(sorted(VOL_FEATURE_FACTORY)) or "<empty>"
        raise KeyError(
            f"get_vol_feature_fn: unknown feature {name!r}; known: {known}"
        ) from exc


register_vol_feature("gjr_asym")(gjr_asym)
register_vol_feature("har_vol")(har_vol)
