"""cfd10 label layer: the forward-looking turn oracle (ground-truth supervision).

This package produces the per-bar *turn* labels the combiner is trained against.
:func:`label_turns` assigns each bar a top and bottom score in ``[0, 1]`` (which
doubles as a sample weight) plus a discrete tier, using a multi-scale nest of
forward-looking extreme/reversal confirmations. The oracle may look ahead, but
strictly no further than :attr:`OracleConfig.horizon` bars, so labels are
reproducible under truncation / streaming.

Public API
----------
Configuration
    :class:`OracleConfig` (frozen) with its monotone :meth:`OracleConfig.weight`
    curve, plus the :data:`WEIGHT_CURVE_REGISTRY` / :func:`register_weight_curve`
    / :func:`get_weight_curve` weight-curve registry.
Labelling
    :func:`label_turns` and the fixed :data:`LABEL_COLUMNS` output schema.
Side registry
    :data:`ORACLE_SIDE_FACTORY` / :func:`register_side` / :func:`get_side_fn`
    expose the per-side scoring reducer by name.
"""

from __future__ import annotations

from cfd10.label_module.oracle import (
    LABEL_COLUMNS,
    ORACLE_SIDE_FACTORY,
    WEIGHT_CURVE_REGISTRY,
    OracleConfig,
    SideReducer,
    WeightCurve,
    get_side_fn,
    get_weight_curve,
    label_turns,
    register_side,
    register_weight_curve,
)

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
