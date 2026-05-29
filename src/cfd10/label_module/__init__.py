"""cfd10 label layer: forward-looking turn oracle (ground truth).

:func:`label_turns` (the *turn oracle*) assigns each bar a top and bottom score in
``[0, 1]`` (which doubles as a sample weight) plus a discrete tier, using a multi-scale
nest of forward-looking extreme/reversal confirmations weighted so a large-scale
(structural) pivot counts far more than a small-scale one. It looks ahead no further
than ``horizon`` bars, so labels are reproducible under truncation / streaming.

Public API
----------
:class:`OracleConfig` (frozen) with its monotone :meth:`OracleConfig.weight` curve,
plus the :data:`WEIGHT_CURVE_REGISTRY` / :func:`register_weight_curve` /
:func:`get_weight_curve` weight-curve registry; :func:`label_turns` and the fixed
:data:`LABEL_COLUMNS` output schema; the per-side reducer registry
:data:`ORACLE_SIDE_FACTORY` / :func:`register_side` / :func:`get_side_fn`.
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
