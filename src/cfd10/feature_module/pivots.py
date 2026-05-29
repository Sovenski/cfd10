"""Confirmed-pivot detection and pivot-drift — cfd10 baseline-structure features.

Faithful Python port of two pieces of
``pine/speculatores_v15_presets_gold.pine``:

* ``ta.pivothigh(high, lb, lb)`` / ``ta.pivotlow(low, lb, lb)`` as called with the
  symmetric ``baseline_lb`` on L491-492. A bar at index ``i`` is a *pivot high*
  when ``high[i]`` is the **strict** maximum of the symmetric window
  ``[i - lb, i + lb]`` (strictly greater than every other bar in the window); a
  *pivot low* mirrors this with a strict minimum. Pine only *confirms* such a
  pivot ``lb`` bars later, so the first and last ``lb`` bars never carry the full
  window and are returned as ``False`` (warm-up).

* ``calc_pivot_drift(pivots, lookback)`` (L142-151): over the last
  ``N = max(lookback, 2)`` confirmed pivots, the per-pivot relative change

      ((end - start) / max(|start|, 1e-9)) / (N - 1)

  where ``start`` is the ``N``-th pivot from the end and ``end`` is the last
  pivot. Fewer than ``N`` confirmed pivots yields ``NaN`` (Pine ``na``).

Indexing convention
-------------------
Outputs of the pivot detectors are aligned to the input length and flag bar
``i`` at its own index ``i`` (the geometric pivot location), not at the later
confirmation bar; the ``lb``-bar confirmation latency manifests only as the
warm-up ``False`` band at each end.

Parity discipline
------------------
Each detector has a scalar per-bar reference and a vectorized (Numba-JIT)
full-series kernel; :mod:`tests.feature_module.test_pivots` pins the vectorized
result against an independent naive reference, and the scalar/vector kernels
share the same strict-comparison logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "pivot_high",
    "pivot_low",
    "pivot_drift",
    "is_pivot_high_at",
    "is_pivot_low_at",
    "PIVOT_FACTORY",
    "register_pivot",
    "get_pivot_fn",
    "PivotFn",
]

_EPS: float = 1e-9

# A pivot feature function. The signature is intentionally loose because the
# registered callables differ in arity (per-series detectors vs the scalar
# pivot-drift reducer); callers resolve a name and supply the documented
# positional arguments for that function.
PivotFn = Callable[..., object]

PIVOT_FACTORY: dict[str, PivotFn] = {}

_F = TypeVar("_F", bound=PivotFn)


# --------------------------------------------------------------------------- #
# Numba kernels (module scope so the JIT cache is shared across calls).        #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _is_pivot_high_at(high: NDArray[np.float64], lb: int, i: int) -> bool:
    """Scalar strict-max pivot-high test at bar ``i`` (Pine ``ta.pivothigh``).

    Returns ``False`` for warm-up bars whose symmetric ``[i-lb, i+lb]`` window
    extends past either end of the series.
    """
    n = high.shape[0]
    if i < lb or i >= n - lb:
        return False
    center = high[i]
    for j in range(i - lb, i + lb + 1):
        if j != i and high[j] >= center:
            return False
    return True


@njit(cache=True, fastmath=False)
def _is_pivot_low_at(low: NDArray[np.float64], lb: int, i: int) -> bool:
    """Scalar strict-min pivot-low test at bar ``i`` (Pine ``ta.pivotlow``).

    Returns ``False`` for warm-up bars whose symmetric ``[i-lb, i+lb]`` window
    extends past either end of the series.
    """
    n = low.shape[0]
    if i < lb or i >= n - lb:
        return False
    center = low[i]
    for j in range(i - lb, i + lb + 1):
        if j != i and low[j] <= center:
            return False
    return True


@njit(cache=True, fastmath=False)
def _pivot_high_series(high: NDArray[np.float64], lb: int) -> NDArray[np.bool_]:
    """Vectorized full-series strict-max pivot-high mask."""
    n = high.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(lb, n - lb):
        out[i] = _is_pivot_high_at(high, lb, i)
    return out


@njit(cache=True, fastmath=False)
def _pivot_low_series(low: NDArray[np.float64], lb: int) -> NDArray[np.bool_]:
    """Vectorized full-series strict-min pivot-low mask."""
    n = low.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(lb, n - lb):
        out[i] = _is_pivot_low_at(low, lb, i)
    return out


@njit(cache=True, fastmath=False)
def _pivot_drift(pivots: NDArray[np.float64], lookback: int) -> float:
    """Scalar Pine ``calc_pivot_drift`` (L142-151).

    ``NaN`` when there are fewer than ``max(lookback, 2)`` confirmed pivots.
    """
    min_pivots = lookback if lookback > 2 else 2
    sz = pivots.shape[0]
    if sz < min_pivots:
        return np.nan
    start_val = pivots[sz - min_pivots]
    end_val = pivots[sz - 1]
    pivot_count = min_pivots - 1
    denom = abs(start_val)
    if denom < _EPS:
        denom = _EPS
    return ((end_val - start_val) / denom) / pivot_count


# --------------------------------------------------------------------------- #
# Public thin wrappers (validate inputs, then dispatch to the JIT kernels).    #
# --------------------------------------------------------------------------- #


def _validate_lb(lb: int) -> None:
    """Raise ``ValueError`` if ``lb`` is not a positive pivot half-window."""
    if lb < 1:
        raise ValueError(f"pivot lookback/lb must be >= 1, got {lb}")


def pivot_high(high: NDArray[np.float64], lb: int) -> NDArray[np.bool_]:
    """Strict-max confirmed pivot-high mask (Pine ``ta.pivothigh(high, lb, lb)``).

    Args:
        high: 1-D array of bar highs.
        lb: Symmetric pivot half-window (left == right legs), must be ``>= 1``.

    Returns:
        Boolean array aligned to ``high`` where element ``i`` is ``True`` iff
        ``high[i]`` is the strict maximum of ``high[i-lb : i+lb+1]``. The first
        and last ``lb`` bars are ``False`` (warm-up, no confirmable window).

    Raises:
        ValueError: If ``lb < 1``.
    """
    _validate_lb(lb)
    high_c = np.ascontiguousarray(high, dtype=np.float64)
    return _pivot_high_series(high_c, lb)


def pivot_low(low: NDArray[np.float64], lb: int) -> NDArray[np.bool_]:
    """Strict-min confirmed pivot-low mask (Pine ``ta.pivotlow(low, lb, lb)``).

    Args:
        low: 1-D array of bar lows.
        lb: Symmetric pivot half-window (left == right legs), must be ``>= 1``.

    Returns:
        Boolean array aligned to ``low`` where element ``i`` is ``True`` iff
        ``low[i]`` is the strict minimum of ``low[i-lb : i+lb+1]``. The first and
        last ``lb`` bars are ``False`` (warm-up, no confirmable window).

    Raises:
        ValueError: If ``lb < 1``.
    """
    _validate_lb(lb)
    low_c = np.ascontiguousarray(low, dtype=np.float64)
    return _pivot_low_series(low_c, lb)


def is_pivot_high_at(high: NDArray[np.float64], lb: int, i: int) -> bool:
    """Scalar per-bar strict-max pivot-high test (reference for :func:`pivot_high`).

    Args:
        high: 1-D array of bar highs.
        lb: Symmetric pivot half-window, must be ``>= 1``.
        i: Absolute bar index to test.

    Returns:
        ``True`` iff ``high[i]`` is the strict maximum of its symmetric ``lb``
        window; ``False`` for warm-up bars at either end.

    Raises:
        ValueError: If ``lb < 1``.
    """
    _validate_lb(lb)
    high_c = np.ascontiguousarray(high, dtype=np.float64)
    return bool(_is_pivot_high_at(high_c, lb, i))


def is_pivot_low_at(low: NDArray[np.float64], lb: int, i: int) -> bool:
    """Scalar per-bar strict-min pivot-low test (reference for :func:`pivot_low`).

    Args:
        low: 1-D array of bar lows.
        lb: Symmetric pivot half-window, must be ``>= 1``.
        i: Absolute bar index to test.

    Returns:
        ``True`` iff ``low[i]`` is the strict minimum of its symmetric ``lb``
        window; ``False`` for warm-up bars at either end.

    Raises:
        ValueError: If ``lb < 1``.
    """
    _validate_lb(lb)
    low_c = np.ascontiguousarray(low, dtype=np.float64)
    return bool(_is_pivot_low_at(low_c, lb, i))


def pivot_drift(confirmed_pivots: NDArray[np.float64], lookback: int) -> float:
    """Per-pivot relative drift over the last ``max(lookback, 2)`` pivots.

    Faithful port of Pine ``calc_pivot_drift`` (L142-151):
    ``((end - start) / max(|start|, 1e-9)) / (N - 1)`` with
    ``N = max(lookback, 2)``, ``start`` the ``N``-th pivot from the end and
    ``end`` the last pivot.

    Args:
        confirmed_pivots: 1-D array of confirmed pivot values, in chronological
            order (oldest first), as accumulated by the caller.
        lookback: Pivot-count lookback; floored to 2 internally.

    Returns:
        The scalar drift, or ``NaN`` when fewer than ``max(lookback, 2)`` pivots
        are available (Pine ``na``).
    """
    pivots_c = np.ascontiguousarray(confirmed_pivots, dtype=np.float64)
    return float(_pivot_drift(pivots_c, lookback))


# --------------------------------------------------------------------------- #
# Registry / factory (mirrors the feature_module package convention).          #
# --------------------------------------------------------------------------- #


def register_pivot(name: str) -> Callable[[_F], _F]:
    """Register a pivot feature function under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator that records the function in :data:`PIVOT_FACTORY` and
        returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: _F) -> _F:
        if name in PIVOT_FACTORY:
            raise ValueError(f"register_pivot: duplicate registration for {name!r}")
        PIVOT_FACTORY[name] = fn
        logger.debug("register_pivot: registered %s", name)
        return fn

    return decorator


def get_pivot_fn(name: str) -> PivotFn:
    """Resolve a registered pivot feature function by name.

    Args:
        name: Registry key used with :func:`register_pivot`.

    Returns:
        The registered callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return PIVOT_FACTORY[name]
    except KeyError as exc:
        known = ", ".join(sorted(PIVOT_FACTORY)) or "<empty>"
        raise KeyError(f"get_pivot_fn: unknown pivot function {name!r}; known: {known}") from exc


# Populate the registry with the parity primitives.
register_pivot("pivot_high")(pivot_high)
register_pivot("pivot_low")(pivot_low)
register_pivot("pivot_drift")(pivot_drift)
