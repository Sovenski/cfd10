"""Dense feature-bank assembly for the cfd10 per-side regime model.

This module wires the parity-faithful per-feature primitives (PIR / agreement,
trend slopes, Kaufman efficiency ratio, volatility position, momentum, and the
GJR-GARCH / HAR volatility asymmetries) into a single *dense* feature matrix.

Design
------
Every feature family is a registrable *block*: a callable
``(df, cfg) -> dict[str, NDArray[np.float64]]`` mapping output column names to
full-length, index-aligned series. Blocks are recorded in
:data:`FEATURE_REGISTRY` via :func:`register_feature` and resolved through
:func:`FeatureBankFactory`. :func:`build_feature_matrix` runs the blocks named in
:attr:`FeatureConfig.blocks` (in order), concatenates their columns, and returns
``(X, feature_names)``.

The configured grids favour **bounded / dimensionless** features (positions-in-
range, agreement fractions, efficiency ratios, clamped volatility asymmetries)
so the matrix is well-scaled for downstream models; the slope and momentum
families are kept normalised (slope over the SMA, returns over the lagged close).
Each block propagates its primitive's warm-up ``NaN`` convention and emits no
``+/-inf``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfd10.feature_module.efficiency import efficiency_ratio
from cfd10.feature_module.garch_har import gjr_asym, har_vol
from cfd10.feature_module.momentum import mom_divergence, mom_velocity, price_return
from cfd10.feature_module.overextension import (
    dist_above_sma_z,
    drawdown_from_high,
    realized_vol_pct,
    up_streak_norm,
    vol_of_vol,
)
from cfd10.feature_module.sma_pir import csum_close, pir_for_scale_series
from cfd10.feature_module.trend import linreg_slope_norm, sma_slope
from cfd10.feature_module.vola import vola_position, vola_raw
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "FeatureConfig",
    "FeatureBlock",
    "FEATURE_REGISTRY",
    "register_feature",
    "FeatureBankFactory",
    "build_feature_matrix",
]

# A feature block: maps a canonical frame + config to named full-length series.
FeatureBlock = Callable[
    ["pd.DataFrame", "FeatureConfig"], dict[str, NDArray[np.float64]]
]

FEATURE_REGISTRY: dict[str, FeatureBlock] = {}


def register_feature(name: str) -> Callable[[FeatureBlock], FeatureBlock]:
    """Register a feature block under ``name``.

    Args:
        name: Unique registry key (also the block's logical group name).

    Returns:
        A decorator recording the block in :data:`FEATURE_REGISTRY` and returning
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(block: FeatureBlock) -> FeatureBlock:
        if name in FEATURE_REGISTRY:
            raise ValueError(f"register_feature: duplicate registration for {name!r}")
        FEATURE_REGISTRY[name] = block
        logger.debug("register_feature: registered %s", name)
        return block

    return decorator


def FeatureBankFactory(name: str) -> FeatureBlock:  # noqa: N802 (factory naming)
    """Resolve a registered feature block by name.

    Args:
        name: Registry key used with :func:`register_feature`.

    Returns:
        The registered block callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return FEATURE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(FEATURE_REGISTRY)) or "<empty>"
        raise KeyError(
            f"FeatureBankFactory: unknown feature block {name!r}; known: {known}"
        ) from exc


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #

# Default block execution order. Kept as a module constant so the frozen
# dataclass can reference an immutable tuple as its default.
_DEFAULT_BLOCKS: tuple[str, ...] = (
    "pir",
    "agreement",
    "efficiency",
    "vola",
    "trend",
    "momentum",
    "garch_har",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable grids for the dense feature bank.

    The defaults are dimensionless / bounded where possible and sized so the
    longest window stays small relative to a multi-decade daily series. All grids
    are tuples (hashable, immutable) so the config can be cached and compared.

    Attributes:
        pir_scales: SMA scales ``s`` for the per-scale position-in-range feature;
            the ratio lookback is ``max(s, pir_lb_floor)`` (Pine ``calc_agreement``
            uses ``max(s, 20)``).
        pir_lb_floor: Floor for the PIR ratio lookback.
        agree_scale_start / agree_scale_end / agree_scale_step: Inclusive scale
            grid for the multi-scale agreement fractions (Pine ``calc_agreement``).
        agree_pct_extreme: High threshold; the low threshold is
            ``1 - agree_pct_extreme``.
        er_periods: Window lengths for the Kaufman efficiency ratio (directional
            and absolute variants are both emitted).
        vola_methods: Volatility methods (subset of ``("ATR","StdDev","Intraday")``).
        vola_lengths: Calculation windows for each volatility method.
        vola_range_len: Trailing window for the volatility position-in-range.
        trend_scales: SMA scales for the slope / linreg-slope features.
        mom_lookbacks: Lookbacks ``L`` for the momentum family.
        include_overextension: When ``True``, append the optional overextension /
            vol-regime block (classic top tells) after the configured ``blocks``.
            Default ``False`` so existing behaviour is unchanged.
        oe_sma_lengths: Long SMA windows for :func:`dist_above_sma_z`.
        oe_z_win: Trailing window for the z-distance normalising std.
        oe_dd_lookbacks: Lookbacks for :func:`drawdown_from_high`.
        oe_rv_win: Window for the rolling realized volatility.
        oe_rv_range_len: Trailing window for the realized-vol position-in-range.
        oe_streak_cap: Saturation count for :func:`up_streak_norm`.
        oe_vov_win: Shared window for :func:`vol_of_vol`.
        blocks: Names of the registered blocks to assemble, in order.
    """

    pir_scales: tuple[int, ...] = (5, 10, 20, 50, 100)
    pir_lb_floor: int = 20

    agree_scale_start: int = 3
    agree_scale_end: int = 120
    agree_scale_step: int = 13
    agree_pct_extreme: float = 0.8

    er_periods: tuple[int, ...] = (10, 20, 40)

    vola_methods: tuple[str, ...] = ("ATR", "StdDev", "Intraday")
    vola_lengths: tuple[int, ...] = (14, 30)
    vola_range_len: int = 100

    trend_scales: tuple[int, ...] = (20, 50, 100)

    mom_lookbacks: tuple[int, ...] = (10, 20, 40)

    # Optional overextension / vol-regime block (top tells). Off by default.
    include_overextension: bool = False
    oe_sma_lengths: tuple[int, ...] = (50, 100, 200)
    oe_z_win: int = 100
    oe_dd_lookbacks: tuple[int, ...] = (20, 50, 100)
    oe_rv_win: int = 20
    oe_rv_range_len: int = 100
    oe_streak_cap: int = 5
    oe_vov_win: int = 20

    blocks: tuple[str, ...] = field(default=_DEFAULT_BLOCKS)


# --------------------------------------------------------------------------- #
# Frame accessors (canonical OHLCV columns -> contiguous float64 arrays).      #
# --------------------------------------------------------------------------- #


def _col(df: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """Return canonical column ``name`` as a contiguous ``float64`` array."""
    return np.ascontiguousarray(df[name].to_numpy(dtype=np.float64))


# --------------------------------------------------------------------------- #
# Feature blocks.                                                             #
# --------------------------------------------------------------------------- #


@register_feature("pir")
def _block_pir(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, NDArray[np.float64]]:
    """Per-scale position-in-range of ``close / SMA(s)`` (bounded ``[0, 1]``).

    One column ``pir_s{scale}`` per ``scale`` in :attr:`FeatureConfig.pir_scales`,
    with ratio lookback ``max(scale, pir_lb_floor)``.
    """
    close = _col(df, "close")
    csum = csum_close(close)
    out: dict[str, NDArray[np.float64]] = {}
    for s in cfg.pir_scales:
        lb = max(int(s), cfg.pir_lb_floor)
        out[f"pir_s{s}"] = pir_for_scale_series(close, csum, int(s), lb)
    return out


@register_feature("agreement")
def _block_agreement(
    df: pd.DataFrame, cfg: FeatureConfig
) -> dict[str, NDArray[np.float64]]:
    """Multi-scale agreement fractions (Pine ``calc_agreement``, bounded ``[0, 1]``).

    Emits ``agree_high`` / ``agree_low``: the fraction of grid scales whose PIR
    exceeds ``agree_pct_extreme`` (high) or falls below ``1 - agree_pct_extreme``
    (low). Computed by reusing the vectorized per-scale PIR series and counting,
    which is numerically identical to the Pine per-bar ``calc_agreement`` loop but
    avoids a Python bar loop. Warm-up is ``NaN`` until the largest scale's PIR is
    defined for every counted scale on that bar.
    """
    close = _col(df, "close")
    csum = csum_close(close)
    n = close.shape[0]

    scales = range(
        cfg.agree_scale_start, cfg.agree_scale_end + 1, cfg.agree_scale_step
    )
    high_thr = cfg.agree_pct_extreme
    low_thr = 1.0 - cfg.agree_pct_extreme

    high_count = np.zeros(n, dtype=np.float64)
    low_count = np.zeros(n, dtype=np.float64)
    valid_count = np.zeros(n, dtype=np.float64)
    n_scales = 0
    for s in scales:
        lb = max(int(s), 20)  # Pine calc_agreement uses lb = max(s, 20).
        pir_s = pir_for_scale_series(close, csum, int(s), lb)
        defined = ~np.isnan(pir_s)
        valid_count += defined
        # NaN-safe comparisons: undefined bars contribute 0 to both counts.
        high_count += np.where(defined & (pir_s > high_thr), 1.0, 0.0)
        low_count += np.where(defined & (pir_s < low_thr), 1.0, 0.0)
        n_scales += 1

    # A bar is warm only once every counted scale's PIR is defined there; before
    # that the agreement fraction is incomplete, so emit NaN (warm-up).
    warm = valid_count >= n_scales
    denom = float(n_scales if n_scales > 1 else 1)
    agree_high = np.where(warm, high_count / denom, np.nan)
    agree_low = np.where(warm, low_count / denom, np.nan)
    return {"agree_high": agree_high, "agree_low": agree_low}


@register_feature("efficiency")
def _block_efficiency(
    df: pd.DataFrame, cfg: FeatureConfig
) -> dict[str, NDArray[np.float64]]:
    """Kaufman efficiency ratio per period (directional ``[-1,1]`` + absolute ``[0,1]``).

    Columns ``er_dir_p{period}`` and ``er_abs_p{period}`` for each configured
    period.
    """
    close = _col(df, "close")
    out: dict[str, NDArray[np.float64]] = {}
    for p in cfg.er_periods:
        out[f"er_dir_p{p}"] = efficiency_ratio(close, int(p), directional=True)
        out[f"er_abs_p{p}"] = efficiency_ratio(close, int(p), directional=False)
    return out


@register_feature("vola")
def _block_vola(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, NDArray[np.float64]]:
    """Volatility position-in-range per method/length (bounded ``[0, 1]``).

    For each ``method`` x ``length`` the raw volatility series is mapped through
    :func:`cfd10.feature_module.vola.vola_position` over ``vola_range_len``,
    giving a column ``vola_pos_{method}_l{length}``.
    """
    high = _col(df, "high")
    low = _col(df, "low")
    close = _col(df, "close")
    out: dict[str, NDArray[np.float64]] = {}
    for method in cfg.vola_methods:
        for length in cfg.vola_lengths:
            raw = vola_raw(high, low, close, method, int(length))
            out[f"vola_pos_{method}_l{length}"] = vola_position(
                raw, cfg.vola_range_len
            )
    return out


@register_feature("trend")
def _block_trend(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, NDArray[np.float64]]:
    """Normalised trend slopes per scale (SMA slope and linreg slope).

    Columns ``sma_slope_s{scale}`` and ``linreg_slope_s{scale}``. Both are the
    slope divided by the contemporaneous SMA (``* 1000``), i.e. a scale-free
    per-thousand growth rate.
    """
    close = _col(df, "close")
    out: dict[str, NDArray[np.float64]] = {}
    for s in cfg.trend_scales:
        out[f"sma_slope_s{s}"] = sma_slope(close, int(s))
        out[f"linreg_slope_s{s}"] = linreg_slope_norm(close, int(s))
    return out


@register_feature("momentum")
def _block_momentum(
    df: pd.DataFrame, cfg: FeatureConfig
) -> dict[str, NDArray[np.float64]]:
    """Momentum family per lookback (return, vol-weighted divergence, velocity).

    Columns ``price_return_L{L}``, ``mom_divergence_L{L}``, ``mom_velocity_L{L}``.
    The ``max(volume[L], 1)`` denominator clamp keeps the divergence finite even
    where historical volume is zero (early SPX bars carry zero volume).
    """
    close = _col(df, "close")
    volume = _col(df, "volume")
    out: dict[str, NDArray[np.float64]] = {}
    for lb in cfg.mom_lookbacks:
        out[f"price_return_L{lb}"] = price_return(close, int(lb))
        out[f"mom_divergence_L{lb}"] = mom_divergence(close, volume, int(lb))
        out[f"mom_velocity_L{lb}"] = mom_velocity(close, int(lb))
    return out


@register_feature("garch_har")
def _block_garch_har(
    df: pd.DataFrame, cfg: FeatureConfig
) -> dict[str, NDArray[np.float64]]:
    """GJR-GARCH asymmetry and HAR / Garman-Klass volatility (both clamped ``[-1,1]``)."""
    del cfg  # No tunable parameters: the Pine constants are fixed.
    open_ = _col(df, "open")
    high = _col(df, "high")
    low = _col(df, "low")
    close = _col(df, "close")
    return {
        "gjr_asym": gjr_asym(open_, high, low, close),
        "har_vol": har_vol(open_, high, low, close),
    }


@register_feature("overextension")
def _block_overextension(
    df: pd.DataFrame, cfg: FeatureConfig
) -> dict[str, NDArray[np.float64]]:
    """Overextension / vol-regime block — classic top tells (optional).

    Emits, all dimensionless and bounded:

    * ``dist_above_sma_z_l{length}`` per ``oe_sma_lengths`` (squashed z-distance
      above the long SMA, ``(-1, 1)``; normalised over ``oe_z_win``).
    * ``drawdown_from_high_l{lb}`` per ``oe_dd_lookbacks`` (``<= 0``; near 0 at
      tops).
    * ``realized_vol_pct`` (vol-regime percentile, ``[0, 1]``).
    * ``up_streak_norm`` (overbought-persistence count, ``[0, 1]``).
    * ``vol_of_vol`` (vol-of-vol percentile, ``[0, 1]``).

    Wired in only when :attr:`FeatureConfig.include_overextension` is ``True``
    (see :func:`build_feature_matrix`).
    """
    close = _col(df, "close")
    out: dict[str, NDArray[np.float64]] = {}
    for length in cfg.oe_sma_lengths:
        out[f"dist_above_sma_z_l{length}"] = dist_above_sma_z(
            close, int(length), cfg.oe_z_win
        )
    for lb in cfg.oe_dd_lookbacks:
        out[f"drawdown_from_high_l{lb}"] = drawdown_from_high(close, int(lb))
    out["realized_vol_pct"] = realized_vol_pct(close, cfg.oe_rv_win, cfg.oe_rv_range_len)
    out["up_streak_norm"] = up_streak_norm(close, cfg.oe_streak_cap)
    out["vol_of_vol"] = vol_of_vol(close, cfg.oe_vov_win)
    return out


# --------------------------------------------------------------------------- #
# Assembly.                                                                   #
# --------------------------------------------------------------------------- #

_REQUIRED_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _validate_frame(df: pd.DataFrame) -> None:
    """Raise ``KeyError`` if the canonical OHLCV columns are missing."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"build_feature_matrix: frame missing columns {missing}")


def build_feature_matrix(
    df: pd.DataFrame, cfg: FeatureConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the configured dense feature bank from a canonical OHLCV frame.

    Runs each block named in ``cfg.blocks`` (in order) via the registry, collects
    their named series, and concatenates them into a single ``float64`` matrix
    aligned to ``df.index``.

    Args:
        df: Canonical OHLCV frame (see :mod:`cfd10.data_module.schema`); must
            contain ``open, high, low, close, volume``.
        cfg: Feature grids and block selection.

    Returns:
        ``(X, feature_names)`` where ``X`` is a :class:`pandas.DataFrame` whose
        columns are exactly ``feature_names`` (in block / grid order, all
        ``float64``) and ``feature_names`` is the ordered list of column names.
        Warm-up bars are ``NaN``; no value is ``+/-inf``.

    Raises:
        KeyError: If ``df`` lacks a required column or names an unknown block.
        ValueError: If two blocks emit a duplicate column name.
    """
    _validate_frame(df)

    # The optional overextension block is appended (not part of the default
    # ``blocks`` tuple) so existing configs are byte-for-byte unchanged. Guard
    # against double-listing if a caller already names it explicitly.
    block_order: tuple[str, ...] = cfg.blocks
    if cfg.include_overextension and "overextension" not in block_order:
        block_order = (*block_order, "overextension")

    columns: dict[str, NDArray[np.float64]] = {}
    feature_names: list[str] = []
    for block_name in block_order:
        block = FeatureBankFactory(block_name)
        produced = block(df, cfg)
        for name, series in produced.items():
            if name in columns:
                raise ValueError(
                    f"build_feature_matrix: duplicate feature column {name!r} "
                    f"from block {block_name!r}"
                )
            arr = np.ascontiguousarray(series, dtype=np.float64)
            if arr.shape[0] != len(df):
                raise ValueError(
                    f"build_feature_matrix: block {block_name!r} column {name!r} "
                    f"has length {arr.shape[0]}, expected {len(df)}"
                )
            columns[name] = arr
            feature_names.append(name)

    logger.info(
        "build_feature_matrix: assembled %d features from %d blocks over %d bars",
        len(feature_names),
        len(block_order),
        len(df),
    )
    # Construct in one shot to keep column order and avoid fragmentation.
    data: Mapping[str, NDArray[np.float64]] = {n: columns[n] for n in feature_names}
    X = pd.DataFrame(data, index=df.index, columns=feature_names)
    return X, feature_names
