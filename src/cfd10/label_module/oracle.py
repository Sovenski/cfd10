"""Forward-looking label oracle for cfd10 market turns (ground truth).

The oracle assigns every bar a *top* and *bottom* turn score in ``[0, 1]`` that
doubles as a per-sample training weight, plus a discrete tier
(``"strong" | "regular" | "none"``). It is deliberately allowed to peek into the
future — it is the supervision signal, not a feature — but the peek is **bounded
by ``cfg.horizon`` bars**, so no label can depend on data more than ``horizon``
bars ahead of the bar it annotates. This bound is what makes the labels
reproducible under streaming / truncation (see the causal-horizon test).

Scoring model
-------------
A *nest* of scales ``scale_nest`` (ascending, e.g. ``(20, 50, 100, 200)``) defines
a multi-resolution notion of "is this bar a turn". For the bottom side, bar ``i``:

1. is a **candidate** iff it is the local minimum over the window tied to the
   largest nest scale (half-width ``max(scale_nest)``, forward leg clipped to
   ``horizon``). Non-candidates score 0 — this keeps the score sparse and anchors
   it to the dominant structure.
2. for each scale ``n`` **confirms** iff it is the ``n``-bar extreme (strict min of
   ``[i - n, i + min(n, horizon)]``) *and* price rebounds by at least
   ``drawdown_pct`` within the next ``min(n, horizon)`` bars (``low -> high`` for a
   bottom; ``high -> low`` for a top).
3. scores ``sum_n w(n) * confirm(i, n) / sum_n w(n)`` — the weight-normalized
   fraction of confirmed scales, where ``w(n)`` rises with ``n`` so a hit at the
   largest scale dominates a hit at the smallest (``w`` of the smallest scale is
   ~0). The score therefore lands in ``[0, 1]``.

``tier`` thresholds the score: ``>= tau_strong`` -> ``"strong"``,
``>= tau_regular`` -> ``"regular"``, else ``"none"``.

The top side mirrors the bottom side with maxima and ``high -> low`` reversals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "OracleConfig",
    "WeightCurve",
    "WEIGHT_CURVE_REGISTRY",
    "register_weight_curve",
    "get_weight_curve",
    "SideReducer",
    "ORACLE_SIDE_FACTORY",
    "register_side",
    "get_side_fn",
    "label_turns",
    "LABEL_COLUMNS",
]

_EPS: float = 1e-12

# The fixed output schema of :func:`label_turns`, in column order.
LABEL_COLUMNS: tuple[str, ...] = (
    "top_score",
    "bottom_score",
    "top_tier",
    "bottom_tier",
    "top_weight",
    "bottom_weight",
)

_TIER_STRONG: str = "strong"
_TIER_REGULAR: str = "regular"
_TIER_NONE: str = "none"


# --------------------------------------------------------------------------- #
# Weight curves (rank -> weight), registry + factory.                          #
# --------------------------------------------------------------------------- #

# A weight curve maps a normalized nest rank ``u in [0, 1]`` (0 = smallest scale,
# 1 = largest) to a non-negative weight. It must be non-decreasing in ``u`` and
# satisfy ``curve(0.0) == 0.0`` so the smallest scale contributes ~nothing.
WeightCurve = Callable[[float], float]

WEIGHT_CURVE_REGISTRY: dict[str, WeightCurve] = {}


def register_weight_curve(name: str) -> Callable[[WeightCurve], WeightCurve]:
    """Register a weight curve under ``name``.

    Args:
        name: Unique registry key (the value used for ``OracleConfig.weight_curve``).

    Returns:
        A decorator recording the curve in :data:`WEIGHT_CURVE_REGISTRY` and
        returning it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(curve: WeightCurve) -> WeightCurve:
        if name in WEIGHT_CURVE_REGISTRY:
            raise ValueError(f"register_weight_curve: duplicate registration for {name!r}")
        WEIGHT_CURVE_REGISTRY[name] = curve
        logger.debug("register_weight_curve: registered %s", name)
        return curve

    return decorator


def get_weight_curve(name: str) -> WeightCurve:
    """Resolve a registered weight curve by name.

    Args:
        name: Registry key used with :func:`register_weight_curve`.

    Returns:
        The registered curve callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return WEIGHT_CURVE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(WEIGHT_CURVE_REGISTRY)) or "<empty>"
        raise KeyError(
            f"get_weight_curve: unknown weight curve {name!r}; known: {known}"
        ) from exc


@register_weight_curve("linear")
def _curve_linear(u: float) -> float:
    """Linear ramp: ``curve(u) = u`` (so the smallest rank is exactly 0)."""
    return u


@register_weight_curve("quadratic")
def _curve_quadratic(u: float) -> float:
    """Convex ramp: ``curve(u) = u**2`` (suppresses small/mid scales harder)."""
    return u * u


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #

_DEFAULT_NEST: tuple[int, ...] = (20, 50, 100, 200)


@dataclass(frozen=True)
class OracleConfig:
    """Immutable configuration for the turn oracle.

    Attributes:
        scale_nest: Ascending nest of extreme scales. The largest entry sets the
            candidate detection window; each entry contributes one confirmation
            test. Must be non-empty, strictly increasing, and all ``>= 1``.
        weight_curve: Name of a registered weight curve (``"linear"`` or
            ``"quadratic"``) mapping nest rank to confirmation weight. The weight
            *increases* with scale; the smallest scale's weight is ~0.
        drawdown_pct: Minimum forward reversal, as a fraction of the extreme price,
            required to confirm a scale (e.g. ``0.10`` = a 10% rebound off a low).
        horizon: Hard cap (in bars) on how far ahead the oracle may look. No label
            depends on any bar beyond ``t + horizon``.
        tau_strong: Score threshold (inclusive) for the ``"strong"`` tier.
        tau_regular: Score threshold (inclusive) for the ``"regular"`` tier; must
            satisfy ``0 < tau_regular <= tau_strong <= 1``.
    """

    scale_nest: tuple[int, ...] = field(default=_DEFAULT_NEST)
    weight_curve: str = "linear"
    drawdown_pct: float = 0.10
    horizon: int = 60
    tau_strong: float = 0.60
    tau_regular: float = 0.30

    def __post_init__(self) -> None:
        """Validate the nest, horizon, drawdown and tier thresholds."""
        if len(self.scale_nest) == 0:
            raise ValueError("OracleConfig: scale_nest must be non-empty")
        if any(s < 1 for s in self.scale_nest):
            raise ValueError(f"OracleConfig: scales must be >= 1, got {self.scale_nest}")
        if list(self.scale_nest) != sorted(set(self.scale_nest)):
            raise ValueError(
                f"OracleConfig: scale_nest must be strictly increasing, got {self.scale_nest}"
            )
        if self.horizon < 1:
            raise ValueError(f"OracleConfig: horizon must be >= 1, got {self.horizon}")
        if not 0.0 < self.drawdown_pct < 1.0:
            raise ValueError(
                f"OracleConfig: drawdown_pct must be in (0, 1), got {self.drawdown_pct}"
            )
        if not 0.0 < self.tau_regular <= self.tau_strong <= 1.0:
            raise ValueError(
                "OracleConfig: require 0 < tau_regular <= tau_strong <= 1, got "
                f"tau_regular={self.tau_regular}, tau_strong={self.tau_strong}"
            )

    def rank_of(self, scale: int) -> int:
        """Return the 0-based ascending rank of ``scale`` within the nest.

        Args:
            scale: A scale value; must be present in :attr:`scale_nest`.

        Returns:
            The index of ``scale`` in the (already sorted) nest.

        Raises:
            ValueError: If ``scale`` is not a member of the nest.
        """
        try:
            return self.scale_nest.index(scale)
        except ValueError as exc:
            raise ValueError(
                f"OracleConfig.rank_of: scale {scale} not in nest {self.scale_nest}"
            ) from exc

    def weight(self, scale: int) -> float:
        """Confirmation weight for a nest ``scale`` (increases with ``scale``).

        The rank is normalized to ``u = rank / (len(nest) - 1)`` (so the smallest
        scale maps to ``u = 0`` and the largest to ``u = 1``) and passed through
        the configured weight curve. With the built-in curves the smallest scale
        therefore has weight ~0 and the largest the maximum weight.

        Args:
            scale: A scale value present in :attr:`scale_nest`.

        Returns:
            The non-negative weight assigned to confirmations at ``scale``.

        Raises:
            ValueError: If ``scale`` is not in the nest.
            KeyError: If :attr:`weight_curve` is not a registered curve.
        """
        rank = self.rank_of(scale)
        denom = len(self.scale_nest) - 1
        u = 0.0 if denom == 0 else rank / float(denom)
        return float(get_weight_curve(self.weight_curve)(u))

    def normalized_weights(self) -> NDArray[np.float64]:
        """Per-scale weights, in nest order, normalized to sum to 1.

        Returns:
            A ``float64`` array ``w`` with ``w.sum() == 1`` (so the weighted
            confirmation fraction lands in ``[0, 1]``). If every curve weight is 0
            (degenerate single-scale nest), falls back to uniform weights.
        """
        raw = np.array([self.weight(s) for s in self.scale_nest], dtype=np.float64)
        total = float(raw.sum())
        if total <= _EPS:
            # Degenerate (e.g. single scale, linear curve -> all zero): treat every
            # scale as equally important so a confirmation still scores.
            return np.full(raw.shape[0], 1.0 / raw.shape[0], dtype=np.float64)
        return raw / total


# --------------------------------------------------------------------------- #
# Numba kernels (forward-looking, horizon-bounded).                            #
# --------------------------------------------------------------------------- #


@njit(cache=True, fastmath=False)
def _side_scores(
    price_ext: NDArray[np.float64],
    price_rev: NDArray[np.float64],
    scales: NDArray[np.int64],
    weights: NDArray[np.float64],
    drawdown_pct: float,
    horizon: int,
    is_top: bool,
) -> NDArray[np.float64]:
    """Weighted multi-scale confirmation score for one side.

    Args:
        price_ext: Series whose extreme defines the turn (``low`` for bottoms,
            ``high`` for tops).
        price_rev: Series the reversal is measured against (``high`` for bottoms,
            ``low`` for tops).
        scales: Ascending nest scales.
        weights: Per-scale weights, already normalized to sum to 1.
        drawdown_pct: Minimum fractional reversal to confirm a scale.
        horizon: Hard forward look-ahead cap (bars).
        is_top: ``True`` for the top (maxima) side, ``False`` for bottoms.

    Returns:
        ``float64`` score in ``[0, 1]`` per bar. Bars without a full forward
        horizon (the last ``horizon`` bars) score 0, as do non-candidates.
    """
    n = price_ext.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    n_scales = scales.shape[0]
    max_scale = int(scales[n_scales - 1])

    for i in range(n):
        # Require a full forward horizon; otherwise the label is not yet decidable.
        if i + horizon >= n:
            continue

        center = price_ext[i]

        # --- candidate gate: local extreme over the largest-scale window. ---
        back = i - max_scale
        if back < 0:
            back = 0
        fwd_cap = horizon if max_scale > horizon else max_scale
        fwd = i + fwd_cap
        is_candidate = True
        for j in range(back, fwd + 1):
            if j == i:
                continue
            if is_top:
                if price_ext[j] >= center:
                    is_candidate = False
                    break
            else:
                if price_ext[j] <= center:
                    is_candidate = False
                    break
        if not is_candidate:
            continue

        # --- per-scale confirmations. ---
        score = 0.0
        for s_idx in range(n_scales):
            scale = int(scales[s_idx])
            f_cap = horizon if scale > horizon else scale
            lo = i - scale
            if lo < 0:
                lo = 0
            hi = i + f_cap

            # n-bar strict extreme over [i - scale, i + f_cap].
            is_extreme = True
            for j in range(lo, hi + 1):
                if j == i:
                    continue
                if is_top:
                    if price_ext[j] >= center:
                        is_extreme = False
                        break
                else:
                    if price_ext[j] <= center:
                        is_extreme = False
                        break
            if not is_extreme:
                continue

            # Forward reversal of >= drawdown_pct within [i+1, i + f_cap].
            reversed_enough = False
            for j in range(i + 1, hi + 1):
                if is_top:
                    move = (center - price_rev[j]) / (center if center > _EPS else _EPS)
                else:
                    move = (price_rev[j] - center) / (center if center > _EPS else _EPS)
                if move >= drawdown_pct:
                    reversed_enough = True
                    break
            if reversed_enough:
                score += weights[s_idx]

        out[i] = score
    return out


# --------------------------------------------------------------------------- #
# Side reducer registry (so the two sides share one configurable code path).   #
# --------------------------------------------------------------------------- #

# A side reducer maps a value series + config to a per-bar score. Registered so
# downstream/experimental scoring variants can be swapped in by name.
SideReducer = Callable[[NDArray[np.float64], OracleConfig], NDArray[np.float64]]

ORACLE_SIDE_FACTORY: dict[str, SideReducer] = {}


def register_side(name: str) -> Callable[[SideReducer], SideReducer]:
    """Register a side reducer under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator recording the reducer in :data:`ORACLE_SIDE_FACTORY` and
        returning it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: SideReducer) -> SideReducer:
        if name in ORACLE_SIDE_FACTORY:
            raise ValueError(f"register_side: duplicate registration for {name!r}")
        ORACLE_SIDE_FACTORY[name] = fn
        logger.debug("register_side: registered %s", name)
        return fn

    return decorator


def get_side_fn(name: str) -> SideReducer:
    """Resolve a registered side reducer by name.

    Args:
        name: Registry key used with :func:`register_side`.

    Returns:
        The registered reducer callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return ORACLE_SIDE_FACTORY[name]
    except KeyError as exc:
        known = ", ".join(sorted(ORACLE_SIDE_FACTORY)) or "<empty>"
        raise KeyError(f"get_side_fn: unknown side reducer {name!r}; known: {known}") from exc


# --------------------------------------------------------------------------- #
# Public API.                                                                 #
# --------------------------------------------------------------------------- #


def _col(df: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """Return canonical column ``name`` as a contiguous ``float64`` array."""
    if name not in df.columns:
        raise KeyError(f"label_turns: frame missing required column {name!r}")
    return np.ascontiguousarray(df[name].to_numpy(dtype=np.float64))


def _tiers(scores: NDArray[np.float64], cfg: OracleConfig) -> NDArray[np.object_]:
    """Threshold a score array into ``strong`` / ``regular`` / ``none`` labels."""
    tiers = np.full(scores.shape[0], _TIER_NONE, dtype=object)
    tiers[scores >= cfg.tau_regular] = _TIER_REGULAR
    tiers[scores >= cfg.tau_strong] = _TIER_STRONG
    return tiers


def label_turns(df: pd.DataFrame, cfg: OracleConfig) -> pd.DataFrame:
    """Label every bar with forward-looking top / bottom turn scores and tiers.

    The score is the weight-normalized fraction of nest scales that *confirm* the
    bar as a turn (it is both the ``n``-bar extreme and is followed by a
    ``>= drawdown_pct`` reversal within ``min(n, horizon)`` bars), gated by the
    bar being the local extreme over the largest-scale window. The score doubles
    as the sample weight; the tier discretizes it via ``tau_strong`` /
    ``tau_regular``.

    The computation only ever reads bars in ``[i - max(scale_nest), i + horizon]``
    for the label at ``i``; consequently labels are invariant to truncation of the
    series at or beyond ``i + horizon`` (no look-ahead leakage past the horizon).

    Args:
        df: Canonical OHLCV frame (see :mod:`cfd10.data_module.schema`); must
            contain ``high`` / ``low`` / ``close``.
        cfg: Oracle configuration.

    Returns:
        A DataFrame index-aligned to ``df`` with columns :data:`LABEL_COLUMNS`:
        ``top_score``, ``bottom_score`` (``float64`` in ``[0, 1]``), ``top_tier``,
        ``bottom_tier`` (categorical-valued strings), and ``top_weight`` /
        ``bottom_weight`` (== the respective scores).

    Raises:
        KeyError: If ``df`` lacks ``high`` / ``low`` / ``close``.
    """
    high = _col(df, "high")
    low = _col(df, "low")
    # ``close`` presence is part of the canonical contract even though the
    # extreme logic uses high/low; validate it so a malformed frame fails loudly.
    _col(df, "close")

    scales = np.ascontiguousarray(np.array(cfg.scale_nest, dtype=np.int64))
    weights = cfg.normalized_weights()

    top_score = _side_scores(
        high, low, scales, weights, cfg.drawdown_pct, cfg.horizon, True
    )
    bottom_score = _side_scores(
        low, high, scales, weights, cfg.drawdown_pct, cfg.horizon, False
    )

    # Clip away any 1-ULP overshoot from the weighted sum so the contract
    # ``score in [0, 1]`` holds exactly.
    np.clip(top_score, 0.0, 1.0, out=top_score)
    np.clip(bottom_score, 0.0, 1.0, out=bottom_score)

    out = pd.DataFrame(
        {
            "top_score": top_score,
            "bottom_score": bottom_score,
            "top_tier": _tiers(top_score, cfg),
            "bottom_tier": _tiers(bottom_score, cfg),
            "top_weight": top_score,
            "bottom_weight": bottom_score,
        },
        index=df.index,
        columns=list(LABEL_COLUMNS),
    )
    logger.info(
        "label_turns: scored %d bars (nest=%s, horizon=%d): "
        "%d strong tops, %d strong bottoms",
        len(df),
        cfg.scale_nest,
        cfg.horizon,
        int((out["top_tier"] == _TIER_STRONG).sum()),
        int((out["bottom_tier"] == _TIER_STRONG).sum()),
    )
    return out


# Register the two canonical sides as named reducers (thin closures over the
# shared kernel) so the side code path is factory-resolvable like the rest of the
# codebase. They are not used by ``label_turns`` directly (which calls the kernel
# once per side for efficiency) but expose the same logic under the registry.
@register_side("bottom")
def _reduce_bottom(low: NDArray[np.float64], cfg: OracleConfig) -> NDArray[np.float64]:
    """Bottom-side score from a ``low`` series (reversal measured against itself).

    Provided for registry completeness / experimentation; uses ``low`` as both the
    extreme and reversal series, so callers wanting the high/low split should use
    :func:`label_turns`.
    """
    scales = np.ascontiguousarray(np.array(cfg.scale_nest, dtype=np.int64))
    weights = cfg.normalized_weights()
    return _side_scores(low, low, scales, weights, cfg.drawdown_pct, cfg.horizon, False)


@register_side("top")
def _reduce_top(high: NDArray[np.float64], cfg: OracleConfig) -> NDArray[np.float64]:
    """Top-side score from a ``high`` series (reversal measured against itself)."""
    scales = np.ascontiguousarray(np.array(cfg.scale_nest, dtype=np.int64))
    weights = cfg.normalized_weights()
    return _side_scores(high, high, scales, weights, cfg.drawdown_pct, cfg.horizon, True)
