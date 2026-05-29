"""cfd10 feature layer: parity primitives and the assembled dense feature bank.

This package ports the feature shim of
``pine/speculatores_v15_presets_gold.pine``. Its lynchpin is the SMA-based
Position-In-Range (PIR) block (lines 68-126): every downstream multi-scale
agreement / volatility-position feature builds on those primitives, so they
match the Pine semantics exactly and each scalar reference is unit-tested against
its vectorized counterpart to 1e-9.

Public API
----------
PIR primitives
    :func:`csum_close`, :func:`sma_at`, :func:`pir_for_scale`,
    :func:`pir_for_scale_series`, :func:`pir_of_series`, :func:`agreement`, plus
    the :data:`PIR_FACTORY` / :func:`register_pir` / :func:`get_pir_fn` registry.
Per-feature families
    trend (:func:`sma_slope`, :func:`linreg_slope_norm`), efficiency
    (:func:`efficiency_ratio`), volatility (:func:`vola_raw`,
    :func:`vola_position`), momentum (:func:`price_return`, :func:`mom_divergence`,
    :func:`mom_velocity`) and GJR/HAR (:func:`gjr_asym`, :func:`har_vol`).
Feature bank
    :func:`build_feature_matrix`, :class:`FeatureConfig`, :func:`register_feature`
    and :func:`FeatureBankFactory` assemble the configured dense bank.

Registry / factory
-------------------
The per-bar scalar feature functions and the dense-bank blocks are registered by
name so downstream config-driven code can resolve them without hard imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from cfd10.feature_module.bank import (
    FEATURE_REGISTRY,
    FeatureBankFactory,
    FeatureConfig,
    build_feature_matrix,
    register_feature,
)
from cfd10.feature_module.efficiency import efficiency_ratio, efficiency_ratio_scalar
from cfd10.feature_module.garch_har import gjr_asym, har_vol
from cfd10.feature_module.momentum import mom_divergence, mom_velocity, price_return
from cfd10.feature_module.overextension import (
    OVEREXTENSION_FACTORY,
    dist_above_sma_z,
    drawdown_from_high,
    get_overextension_fn,
    realized_vol_pct,
    register_overextension,
    up_streak_norm,
    vol_of_vol,
)
from cfd10.feature_module.pivots import pivot_drift, pivot_high, pivot_low
from cfd10.feature_module.sma_pir import (
    AgreementResult,
    agreement,
    csum_close,
    pir_for_scale,
    pir_for_scale_series,
    pir_of_series,
    sma_at,
)
from cfd10.feature_module.trend import linreg_slope_norm, sma_slope, slope_delta
from cfd10.feature_module.vola import vola_position, vola_raw
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

# A per-bar PIR feature function. The signature is intentionally loose
# (``*args`` carries the bar-local parameters) because the registered callables
# differ in arity (e.g. ``pir_for_scale`` vs ``agreement``); callers resolve a
# name and supply the documented positional arguments for that function.
PirFn = Callable[..., object]

PIR_FACTORY: dict[str, PirFn] = {}

_F = TypeVar("_F", bound=PirFn)


def register_pir(name: str) -> Callable[[_F], _F]:
    """Register a PIR-style feature function under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator that records the function in :data:`PIR_FACTORY` and returns
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: _F) -> _F:
        if name in PIR_FACTORY:
            raise ValueError(f"register_pir: duplicate registration for {name!r}")
        PIR_FACTORY[name] = fn
        logger.debug("register_pir: registered %s", name)
        return fn

    return decorator


def get_pir_fn(name: str) -> PirFn:
    """Resolve a registered PIR feature function by name.

    Args:
        name: Registry key used with :func:`register_pir`.

    Returns:
        The registered callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return PIR_FACTORY[name]
    except KeyError as exc:
        known = ", ".join(sorted(PIR_FACTORY)) or "<empty>"
        raise KeyError(f"get_pir_fn: unknown PIR function {name!r}; known: {known}") from exc


# Populate the registry with the parity primitives.
register_pir("sma_at")(sma_at)
register_pir("pir_for_scale")(pir_for_scale)
register_pir("agreement")(agreement)

__all__ = [
    # PIR parity primitives.
    "csum_close",
    "sma_at",
    "pir_for_scale",
    "pir_for_scale_series",
    "pir_of_series",
    "agreement",
    "AgreementResult",
    "PIR_FACTORY",
    "register_pir",
    "get_pir_fn",
    "PirFn",
    # Per-feature families.
    "sma_slope",
    "linreg_slope_norm",
    "slope_delta",
    "efficiency_ratio",
    "efficiency_ratio_scalar",
    "vola_raw",
    "vola_position",
    "price_return",
    "mom_divergence",
    "mom_velocity",
    "gjr_asym",
    "har_vol",
    "pivot_high",
    "pivot_low",
    "pivot_drift",
    # Overextension / vol-regime (optional top tells).
    "dist_above_sma_z",
    "drawdown_from_high",
    "realized_vol_pct",
    "up_streak_norm",
    "vol_of_vol",
    "OVEREXTENSION_FACTORY",
    "register_overextension",
    "get_overextension_fn",
    # Dense feature bank.
    "build_feature_matrix",
    "FeatureConfig",
    "register_feature",
    "FeatureBankFactory",
    "FEATURE_REGISTRY",
]
