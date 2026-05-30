"""Emit a distilled student (JSON) as a standalone Pine v6 indicator.

The deployable detector is a SMALL, Pine-portable model per side
(``outputs/student_{low,high}_v2.json``): either a single
``DecisionTreeClassifier`` (``depth`` up to 8) or a small
``GradientBoostingClassifier``. This module transcribes both sides into a single
TradingView ``@version=6`` indicator that, on each bar, recomputes only the
features the models split on, evaluates each side's rule, and plots a LOW-pivot /
HIGH-pivot marker — mirroring the marker style of the original
``pine/speculatores_v15_presets_gold.pine``.

Per-side rule transcription
--------------------------
* **tree** — nested ``if/else`` (``feature <= threshold`` routes to the ``if``
  branch, sklearn's rule); each leaf sets the side flag from its stored
  ``predict``.
* **gboost** — the additive margin ``init + learning_rate *
  sum_over_stages(stage leaf value)`` in pure Pine float arithmetic, with the
  signal ``margin >= logit(threshold)``. No sigmoid is emitted: the logistic link
  is monotone, so ``proba >= threshold`` is exactly ``margin >= logit(threshold)``
  (and at ``threshold = 0.5`` this is sklearn's ``predict()``: ``logit(0.5) ==
  0``). The chosen ``threshold`` is baked into the Pine in logit space.

Parity contract
---------------
Every emitted feature recomputation matches the *Python* feature bank
(:mod:`cfd10.feature_module`, ``FeatureConfig(include_overextension=True)``)
bar-for-bar, which in turn is a faithful port of the original Pine semantics:

* ``price_return_L{L}`` / ``mom_divergence_L{L}`` / ``mom_velocity_L{L}`` — Pine
  L443-452 momentum block.
* ``er_dir_p{P}`` / ``er_abs_p{P}`` — Pine L468-480 Kaufman efficiency ratio.
* ``sma_slope_s{S}`` / ``linreg_slope_s{S}`` — Pine L402-413 trend slopes, with the
  ``ta.cum`` SMA idiom (L87-93) and the ``d = max(round(S/4), 2)`` lag.
* ``vola_pos_{METHOD}_l{L}`` — Pine L456-457 raw volatility + ``pir_of`` position.
* ``pir_s{S}`` / ``agree_{high,low}`` — Pine L98-126 ``pir_for_scale`` /
  ``calc_agreement`` (the ``ta.cum`` agreement idiom).
* ``har_vol`` / ``gjr_asym`` — Pine L179-188 ``calc_har_vol`` / L159-177
  ``calc_gjr_asym``.
* ``dist_above_sma_z_l{L}`` / ``drawdown_from_high_l{L}`` / ``realized_vol_pct`` /
  ``up_streak_norm`` — the overextension / vol-regime block
  (:mod:`cfd10.feature_module.overextension`), classic top tells.

The feature-bank *constants* are baked in from :class:`FeatureConfig` defaults
(``pir_lb_floor=20``, ``vola_range_len=100``, the slope ``* 1000`` scaling,
``oe_z_win=100``, ``oe_rv_win=20``, ``oe_streak_cap=5``, the agreement grid
``3..120`` step ``13`` at ``pct_extreme=0.8``), *not* from the original
indicator's per-preset inputs: the student was trained on the cfd10 feature bank,
so the export must reproduce that bank, not the legacy voting detector.

Public API
----------
:func:`load_student_json` — read a student JSON (tree OR gboost) written by the
distillation pipeline.
:func:`emit_indicator` — render the full two-side Pine v6 indicator string.
:func:`write_indicator` — render + write ``pine/cfd10_student.pine``.
:func:`feature_pine_expr` — the Pine variable name + computation for one feature
(also drives the parity contract doc).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "PINE_PATH",
    "FeaturePine",
    "TreeJson",
    "load_student_json",
    "feature_pine_expr",
    "emit_feature_block",
    "emit_indicator",
    "write_indicator",
]

# A student tree parsed from JSON (see ``distill_students._tree_to_json``): a
# heterogeneous mapping whose ``nodes`` key holds the node array. ``Any``-valued
# because values mix str / int / float / bool / None / list.
TreeJson = dict[str, Any]

# Default output path for the standalone student indicator.
PINE_PATH: Path = Path(__file__).resolve().parents[3] / "pine" / "cfd10_student.pine"

# Feature-bank constants baked into the export (cfd10 ``FeatureConfig`` defaults).
_PIR_LB_FLOOR: int = 20  # FeatureConfig.pir_lb_floor (also Pine calc_agreement floor).
_VOLA_RANGE_LEN: int = 100  # FeatureConfig.vola_range_len.
_SLOPE_SCALE: float = 1000.0  # Pine L404/L406 ``* 1000`` slope normalisation.

# Overextension / vol-regime block constants (FeatureConfig defaults).
_OE_Z_WIN: int = 100  # FeatureConfig.oe_z_win (z-distance normalising std window).
_OE_RV_WIN: int = 20  # FeatureConfig.oe_rv_win (realized-vol window).
_OE_RV_RANGE_LEN: int = 100  # FeatureConfig.oe_rv_range_len (rv position-in-range).
_OE_STREAK_CAP: int = 5  # FeatureConfig.oe_streak_cap (up-streak saturation count).

# Multi-scale agreement grid (FeatureConfig defaults; Pine ``calc_agreement``).
_AGREE_SCALE_START: int = 3  # FeatureConfig.agree_scale_start.
_AGREE_SCALE_END: int = 120  # FeatureConfig.agree_scale_end.
_AGREE_SCALE_STEP: int = 13  # FeatureConfig.agree_scale_step.
_AGREE_PCT_EXTREME: float = 0.8  # FeatureConfig.agree_pct_extreme.

# Default probability cut baked into a gboost export. ``0.5`` reproduces sklearn's
# ``predict()`` exactly: for the binary log-loss booster ``predict()`` is
# ``proba >= 0.5`` <=> ``margin >= 0`` (``logit(0.5) == 0``). The Pine compares the
# raw additive margin to ``logit(threshold)`` (a monotone re-expression of the
# probability cut), so no sigmoid is ever emitted.
_GBOOST_DECISION_THRESHOLD: float = 0.5


# --------------------------------------------------------------------------- #
# Per-feature Pine emission.                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeaturePine:
    """A feature's Pine realisation.

    Attributes:
        name: The cfd10 feature column name (e.g. ``price_return_L20``).
        var: The Pine variable name holding the feature value (a sanitised,
            collision-free identifier derived from ``name``).
        code: One or more Pine statements (newline-joined, no trailing newline)
            that assign ``var``. Helper functions used here are emitted once in
            the indicator preamble (see :data:`_HELPERS`).
    """

    name: str
    var: str
    code: str


def _pine_var(name: str) -> str:
    """Return a Pine-safe variable identifier for feature ``name``.

    Pine identifiers allow ``[A-Za-z_][A-Za-z0-9_]*``; the cfd10 names already
    satisfy this, so the mapping is the identity prefixed with ``f_`` to avoid
    clashing with the indicator's own locals / built-ins.
    """
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return f"f_{safe}"


def _slope_lag(s: int) -> int:
    """Pine/Python SMA-slope lag ``d = max(round(s / 4), 2)`` (round half away).

    Mirrors :func:`cfd10.feature_module.trend.slope_delta`: ``floor(s/4 + 1/2)``
    realises round-half-away-from-zero for ``s > 0``, then floored at 2.
    """
    if s <= 0:
        raise ValueError(f"_slope_lag: S must be positive, got {s}")
    rounded = (2 * s + 4) // 8  # == floor(s / 4 + 1 / 2) for s > 0
    return rounded if rounded > 2 else 2


def _parse_suffix_int(name: str, prefix: str) -> int:
    """Extract the trailing integer of ``name`` after ``prefix`` (e.g. L20 -> 20)."""
    tail = name[len(prefix):]
    if not tail.isdigit():
        raise ValueError(f"feature_pine_expr: cannot parse int from {name!r} (prefix {prefix!r})")
    return int(tail)


def _emit_price_return(name: str, var: str) -> str:
    """Pine ``(close - close[L]) / close[L]`` (Pine L443/449; warm-up via nz)."""
    lb = _parse_suffix_int(name, "price_return_L")
    return f"float {var} = nz((close - close[{lb}]) / close[{lb}])"


def _emit_mom_divergence(name: str, var: str) -> str:
    """Pine ``price_ret * vol_ret`` with the ``max(volume[L], 1)`` clamp (L444-451)."""
    lb = _parse_suffix_int(name, "mom_divergence_L")
    return (
        f"float {var}_pr = nz((close - close[{lb}]) / close[{lb}])\n"
        f"float {var}_vr = nz((volume - volume[{lb}]) / math.max(nz(volume[{lb}]), 1))\n"
        f"float {var} = {var}_pr * {var}_vr"
    )


def _emit_efficiency(name: str, var: str, directional: bool) -> str:
    """Pine Kaufman efficiency ratio over ``period`` (L468-480)."""
    prefix = "er_dir_p" if directional else "er_abs_p"
    period = _parse_suffix_int(name, prefix)
    net = f"(close - close[{period}])" if directional else f"math.abs(close - close[{period}])"
    return (
        f"float {var}_path = 0.0\n"
        f"for {var}_i = 0 to {period} - 1\n"
        f"    {var}_path += math.abs(close[{var}_i] - close[{var}_i + 1])\n"
        f"float {var} = {var}_path > 0 ? {net} / {var}_path : 0.0"
    )


def _emit_sma_slope(name: str, var: str) -> str:
    """Pine SMA-slope via the ta.cum idiom (L404), lag ``d=max(round(S/4),2)``."""
    s = _parse_suffix_int(name, "sma_slope_s")
    d = _slope_lag(s)
    return (
        f"float {var}_now = sma_at({s}, 0)\n"
        f"float {var}_prev = sma_at({s}, {d})\n"
        f"float {var} = {var}_now > 0 ? nz({var}_now - {var}_prev) "
        f"/ ({d} * {var}_now) * {_SLOPE_SCALE:.1f} : 0.0"
    )


def _emit_linreg_slope(name: str, var: str) -> str:
    """Pine normalised linreg-slope (L405-406): OLS slope / SMA(S) * 1000."""
    s = _parse_suffix_int(name, "linreg_slope_s")
    return (
        f"float {var}_b = ta.linreg(close, {s}, 0) - ta.linreg(close, {s}, 1)\n"
        f"float {var}_sma = sma_at({s}, 0)\n"
        f"float {var} = {var}_sma > 0 ? {var}_b / {var}_sma * {_SLOPE_SCALE:.1f} : 0.0"
    )


def _emit_vola_pos(name: str, var: str) -> str:
    """Pine raw volatility + ``pir_of`` position over ``vola_range_len`` (L456-457)."""
    body = name[len("vola_pos_"):]
    method, _, ltail = body.partition("_l")
    if not ltail.isdigit():
        raise ValueError(f"feature_pine_expr: cannot parse vola length from {name!r}")
    length = int(ltail)
    if method == "ATR":
        raw = f"ta.atr({length})"
    elif method == "StdDev":
        raw = f"ta.stdev(close, {length})"
    elif method == "Intraday":
        raw = f"ta.sma(close > 0 ? (high - low) / close : 0.0, {length})"
    else:
        raise ValueError(f"feature_pine_expr: unknown vola method {method!r} in {name!r}")
    return (
        f"float {var}_raw = {raw}\n"
        f"float {var} = pir_of({var}_raw, {_VOLA_RANGE_LEN})"
    )


def _emit_har_vol(name: str, var: str) -> str:
    """Pine HAR / Garman-Klass volatility norm (L179-188) via ``calc_har_vol``."""
    del name
    return f"[{var}_norm, {var}_ratio] = calc_har_vol()\nfloat {var} = {var}_norm"


def _emit_pir_s(name: str, var: str) -> str:
    """Pine per-scale PIR via ``pir_for_scale`` (L98-126), lb = max(S, 20)."""
    s = _parse_suffix_int(name, "pir_s")
    lb = max(s, _PIR_LB_FLOOR)
    return f"float {var} = pir_for_scale({s}, {lb})"


def _emit_mom_velocity(name: str, var: str) -> str:
    """Pine momentum-velocity ``price_ret - price_ret[1]`` (L446/452).

    ``price_ret = nz((close - close[L]) / close[L])`` (the warm-up ``nz`` mirrors
    :func:`cfd10.feature_module.momentum.price_return`, whose first ``L`` bars are
    NaN; on the warm-up-free pooled matrix the difference is exact). The velocity
    is the bar-over-bar change of that return, computed via the ``[1]`` history of
    the per-bar return so it is a single Pine series expression.
    """
    lb = _parse_suffix_int(name, "mom_velocity_L")
    return (
        f"float {var}_ret = nz((close - close[{lb}]) / close[{lb}])\n"
        f"float {var} = nz({var}_ret - {var}_ret[1])"
    )


def _emit_dist_above_sma_z(name: str, var: str) -> str:
    """Pine squashed z-distance above the long SMA (overextension, L237-270 py).

    ``tanh((close - sma_at(L,0)) / ta.stdev(close, oe_z_win))`` with a flat-window
    guard mapping a zero std to ``0.0`` (``tanh(0)``), matching
    :func:`cfd10.feature_module.overextension.dist_above_sma_z`. The SMA reuses the
    ``ta.cum`` ``sma_at`` idiom (so it is bit-identical to the other SMA features)
    and the normalising std is Pine ``ta.stdev`` (population / biased, as NumPy).
    """
    length = _parse_suffix_int(name, "dist_above_sma_z_l")
    return (
        f"float {var}_sma = sma_at({length}, 0)\n"
        f"float {var}_std = ta.stdev(close, {_OE_Z_WIN})\n"
        f"float {var}_z = ({var}_std > 0 and not na({var}_sma)) ? "
        f"(close - {var}_sma) / {var}_std : 0.0\n"
        f"float {var} = (na({var}_sma) or na({var}_std)) ? float(na) : "
        f"(math.exp(2.0 * {var}_z) - 1.0) / (math.exp(2.0 * {var}_z) + 1.0)"
    )


def _emit_drawdown_from_high(name: str, var: str) -> str:
    """Pine drawdown from the trailing running high (<= 0; L273-301 py).

    ``(close - ta.highest(close, lb)) / ta.highest(close, lb)`` with the
    non-positive-high guard mapping to ``0.0`` (matching
    :func:`cfd10.feature_module.overextension.drawdown_from_high`).
    """
    lb = _parse_suffix_int(name, "drawdown_from_high_l")
    return (
        f"float {var}_hi = ta.highest(close, {lb})\n"
        f"float {var} = {var}_hi > 0 ? (close - {var}_hi) / {var}_hi : 0.0"
    )


def _emit_realized_vol_pct(name: str, var: str) -> str:
    """Pine realized-vol position-in-range (vol regime, L304-329 py).

    Realized vol = population std of one-bar log returns over ``oe_rv_win``; mapped
    through ``pir_of`` over ``oe_rv_range_len``. The log-return series is the bank's
    ``ln(close/close[1])`` (degenerate non-positive closes -> the series is
    undefined there, but SPX closes are strictly positive). ``ta.stdev`` over the
    return series is the population std, matching NumPy ``ddof=0``.
    """
    del name
    return (
        f"float {var}_lr = math.log(close / close[1])\n"
        f"float {var}_rv = ta.stdev({var}_lr, {_OE_RV_WIN})\n"
        f"float {var} = pir_of({var}_rv, {_OE_RV_RANGE_LEN})"
    )


def _emit_up_streak_norm(name: str, var: str) -> str:
    """Pine normalised consecutive-up-bar streak (overbought persistence, py L332).

    ``min(streak, cap) / cap`` where ``streak`` counts consecutive bars closing
    strictly above the prior bar. The streak is carried as ``var`` state and reset
    on a down / flat bar, exactly mirroring
    :func:`cfd10.feature_module.overextension.up_streak_norm` (bar 0 is ``0.0``).
    """
    del name
    return (
        f"var int {var}_streak = 0\n"
        f"{var}_streak := close > close[1] ? {var}_streak + 1 : 0\n"
        f"float {var} = math.min({var}_streak, {_OE_STREAK_CAP}) / "
        f"{float(_OE_STREAK_CAP):.1f}"
    )


def _emit_gjr_asym(name: str, var: str) -> str:
    """Pine GJR-GARCH asymmetry (``calc_gjr_asym``, L159-177).

    Transcribed verbatim from the original indicator: a leverage-aware GJR-GARCH
    recursion alongside a symmetric one on squared log-returns, reporting the
    clamped normalised divergence. The ``var`` state carries the two variances
    forward (Pine ``[1]`` history), matching
    :func:`cfd10.feature_module.garch_har.gjr_asym`.
    """
    del name
    return f"[{var}_norm, {var}_ratio] = calc_gjr_asym()\nfloat {var} = {var}_norm"


def _agreement_scales() -> list[int]:
    """Bake the agreement scale grid ``range(start, end + 1, step)`` (FeatureConfig)."""
    return list(range(_AGREE_SCALE_START, _AGREE_SCALE_END + 1, _AGREE_SCALE_STEP))


def _emit_agree(name: str, var: str, high: bool) -> str:
    """Pine multi-scale agreement fraction (``calc_agreement``, L115-126).

    Emits the fraction of grid scales whose ``pir_for_scale`` exceeds
    ``agree_pct_extreme`` (high) or falls below ``1 - agree_pct_extreme`` (low),
    over the baked scale grid. This mirrors the vectorized counting in the cfd10
    ``agreement`` block (which is numerically identical to the Pine per-bar loop);
    the warm-up guard re-masks bars where any counted scale's PIR is undefined
    (``na``) so the fraction matches the Python NaN convention on incomplete bars.
    """
    del name
    scales = _agreement_scales()
    denom = len(scales) if len(scales) > 1 else 1
    if high:
        cmp = f"> {_AGREE_PCT_EXTREME!r}"
    else:
        cmp = f"< {(1.0 - _AGREE_PCT_EXTREME)!r}"
    lines: list[str] = [
        f"int {var}_count = 0",
        f"int {var}_valid = 0",
    ]
    for s in scales:
        lb = max(s, _PIR_LB_FLOOR)
        lines.append(f"float {var}_p{s} = pir_for_scale({s}, {lb})")
        lines.append(f"if not na({var}_p{s})")
        lines.append(f"    {var}_valid += 1")
        lines.append(f"    if {var}_p{s} {cmp}")
        lines.append(f"        {var}_count += 1")
    lines.append(
        f"float {var} = {var}_valid >= {len(scales)} ? "
        f"{var}_count / {float(denom):.1f} : float(na)"
    )
    return "\n".join(lines)


def feature_pine_expr(name: str) -> FeaturePine:
    """Return the Pine variable + computation for cfd10 feature ``name``.

    Dispatches on the feature-family prefix to the matching emitter. The emitted
    code reproduces the Python feature bank (and thus the original Pine
    semantics) bar-for-bar, using the helper functions declared once in the
    indicator preamble (:data:`_HELPERS`).

    Args:
        name: A cfd10 feature column name the student tree splits on.

    Returns:
        The :class:`FeaturePine` triple ``(name, var, code)``.

    Raises:
        ValueError: If ``name`` matches no known feature family.
    """
    var = _pine_var(name)
    if name.startswith("price_return_L"):
        return FeaturePine(name, var, _emit_price_return(name, var))
    if name.startswith("mom_divergence_L"):
        return FeaturePine(name, var, _emit_mom_divergence(name, var))
    if name.startswith("mom_velocity_L"):
        return FeaturePine(name, var, _emit_mom_velocity(name, var))
    if name.startswith("er_dir_p"):
        return FeaturePine(name, var, _emit_efficiency(name, var, directional=True))
    if name.startswith("er_abs_p"):
        return FeaturePine(name, var, _emit_efficiency(name, var, directional=False))
    if name.startswith("sma_slope_s"):
        return FeaturePine(name, var, _emit_sma_slope(name, var))
    if name.startswith("linreg_slope_s"):
        return FeaturePine(name, var, _emit_linreg_slope(name, var))
    if name.startswith("vola_pos_"):
        return FeaturePine(name, var, _emit_vola_pos(name, var))
    if name == "har_vol":
        return FeaturePine(name, var, _emit_har_vol(name, var))
    if name == "gjr_asym":
        return FeaturePine(name, var, _emit_gjr_asym(name, var))
    if name.startswith("pir_s"):
        return FeaturePine(name, var, _emit_pir_s(name, var))
    if name.startswith("dist_above_sma_z_l"):
        return FeaturePine(name, var, _emit_dist_above_sma_z(name, var))
    if name.startswith("drawdown_from_high_l"):
        return FeaturePine(name, var, _emit_drawdown_from_high(name, var))
    if name == "realized_vol_pct":
        return FeaturePine(name, var, _emit_realized_vol_pct(name, var))
    if name == "up_streak_norm":
        return FeaturePine(name, var, _emit_up_streak_norm(name, var))
    if name == "agree_high":
        return FeaturePine(name, var, _emit_agree(name, var, high=True))
    if name == "agree_low":
        return FeaturePine(name, var, _emit_agree(name, var, high=False))
    raise ValueError(f"feature_pine_expr: no Pine emitter for feature {name!r}")


# --------------------------------------------------------------------------- #
# Tree -> nested Pine if/else.                                                 #
# --------------------------------------------------------------------------- #


def _fmt_threshold(value: float) -> str:
    """Format a split threshold with full round-trip precision (``repr``).

    Full precision keeps the emitted Pine constant byte-identical to the JSON /
    sklearn threshold, so the rule boundary is exact (the parity contract).
    """
    return repr(float(value))


def _is_gboost(model_json: TreeJson) -> bool:
    """Return ``True`` iff ``model_json`` is a gradient-boosting student.

    The two schemas (written by ``redistill_students._model_to_json``) are
    distinguished by ``model_kind``; a missing key defaults to the legacy single
    ``tree`` schema (which carries a ``nodes`` array, no ``stages``).
    """
    return str(model_json.get("model_kind", "tree")) == "gboost"


def load_student_json(path: str | Path) -> TreeJson:
    """Load and minimally validate a student model JSON (tree OR gboost).

    Args:
        path: Path to ``student_{side}.json`` / ``student_{side}_v2.json`` (written
            by the distillation pipeline).

    Returns:
        The parsed model dict: a single decision tree (``model_kind="tree"`` with a
        ``nodes`` array) or a gradient boost (``model_kind="gboost"`` with a
        ``stages`` list of per-estimator regressor node arrays plus ``init`` /
        ``learning_rate``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON is neither a valid tree nor a valid gboost export.
    """
    p = Path(path)
    model = json.loads(p.read_text(encoding="utf-8"))
    if _is_gboost(model):
        stages = model.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"load_student_json: {p} gboost has no 'stages' list")
        if "init" not in model or "learning_rate" not in model:
            raise ValueError(
                f"load_student_json: {p} gboost missing 'init' / 'learning_rate'"
            )
        return model
    if "nodes" not in model or not isinstance(model["nodes"], list):
        raise ValueError(f"load_student_json: {p} has no 'nodes' array")
    return model


def _nodes_by_id(tree: TreeJson) -> dict[int, dict[str, Any]]:
    """Index the node list by ``node_id`` for O(1) child resolution."""
    return {int(n["node_id"]): n for n in tree["nodes"]}


def _emit_tree_body(
    tree: TreeJson,
    flag_var: str,
    indent: str,
) -> str:
    """Render the tree as a nested Pine ``if/else`` assigning ``flag_var``.

    scikit-learn routes ``feature <= threshold`` to the LEFT child, so the Pine
    ``if`` branch takes ``<=`` and the ``else`` branch ``>``. Each leaf assigns
    ``flag_var := true`` when its stored ``predict`` is ``1`` (mirroring the
    sklearn ``predict()`` we pin against in :mod:`cfd10.parity_module.verify`).

    Args:
        tree: The parsed student tree dict.
        flag_var: The Pine bool variable assigned at each leaf.
        indent: Leading whitespace for the top-level ``if`` (nested deeper inside).

    Returns:
        The Pine ``if/else`` block (no trailing newline).
    """
    nodes = _nodes_by_id(tree)
    lines: list[str] = []

    def _recurse(node_id: int, depth: int) -> None:
        node = nodes[node_id]
        pad = indent + "    " * depth
        if bool(node["is_leaf"]):
            if int(node["predict"]) == 1:
                lines.append(f"{pad}{flag_var} := true")
            else:
                lines.append(f"{pad}{flag_var} := {flag_var}")  # explicit no-op: stays false
            return
        var = _pine_var(str(node["feature"]))
        thr = _fmt_threshold(float(node["threshold"]))
        lines.append(f"{pad}if {var} <= {thr}")
        _recurse(int(node["left"]), depth + 1)
        lines.append(f"{pad}else")
        _recurse(int(node["right"]), depth + 1)

    _recurse(0, 0)
    return "\n".join(lines)


def _logit(p: float) -> float:
    """Return ``log(p / (1 - p))`` with ``p`` clipped to keep the log finite.

    The gboost decision ``proba >= threshold`` is re-expressed in margin space as
    ``margin >= logit(threshold)`` (the sigmoid is monotone, so the two cuts are
    identical). Baking ``logit(threshold)`` keeps the Pine call exact float
    arithmetic with no sigmoid.
    """
    clipped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(math.log(clipped / (1.0 - clipped)))


def _emit_regressor_body(
    stage_nodes: list[dict[str, Any]],
    value_var: str,
    indent: str,
) -> str:
    """Render one gboost stage regressor as a nested Pine ``if/else``.

    Mirrors :func:`_emit_tree_body` but for a *regressor*: each leaf assigns the
    stage's raw leaf ``value`` (the contribution the booster sums, before
    ``learning_rate``) to ``value_var``. ``feature <= threshold`` routes LEFT
    (scikit-learn's rule), exactly as the classifier path.

    Args:
        stage_nodes: One stage's node array (a list of node dicts; leaves carry
            ``value`` not ``predict``), as written by
            ``redistill_students._model_to_json``.
        value_var: The Pine ``float`` variable assigned at each leaf.
        indent: Leading whitespace for the top-level ``if``.

    Returns:
        The Pine ``if/else`` block (no trailing newline).
    """
    nodes = {int(n["node_id"]): n for n in stage_nodes}
    lines: list[str] = []

    def _recurse(node_id: int, depth: int) -> None:
        node = nodes[node_id]
        pad = indent + "    " * depth
        if bool(node["is_leaf"]):
            lines.append(f"{pad}{value_var} := {_fmt_threshold(float(node['value']))}")
            return
        var = _pine_var(str(node["feature"]))
        thr = _fmt_threshold(float(node["threshold"]))
        lines.append(f"{pad}if {var} <= {thr}")
        _recurse(int(node["left"]), depth + 1)
        lines.append(f"{pad}else")
        _recurse(int(node["right"]), depth + 1)

    _recurse(0, 0)
    return "\n".join(lines)


def _emit_gboost_body(
    model: TreeJson,
    flag_var: str,
    threshold: float,
) -> str:
    """Render a gradient-boosting student as its additive margin + threshold rule.

    Emits, in pure Pine float arithmetic:

    * one ``float {flag_var}_s{k}`` per stage, assigned by that stage's nested
      regressor ``if/else`` (the stage's raw leaf value);
    * ``float {flag_var}_margin = init + learning_rate * (s0 + s1 + ...)``;
    * ``{flag_var} := {flag_var}_margin >= logit(threshold)``.

    The decision is the raw-margin comparison ``margin >= logit(threshold)`` — NOT
    a sigmoid — which is the exact monotone re-expression of ``proba >= threshold``
    (and, at ``threshold = 0.5``, of sklearn's ``predict()``: ``logit(0.5) == 0``).

    Args:
        model: A gboost model dict (``init``, ``learning_rate``, ``stages``).
        flag_var: The Pine bool variable assigned by the rule.
        threshold: The probability cut baked into the rule (logit space in Pine).

    Returns:
        The Pine block computing the margin and assigning ``flag_var`` (no trailing
        newline).
    """
    init = float(model["init"])
    lr = float(model["learning_rate"])
    stages = model["stages"]
    logit_thr = _logit(threshold)

    lines: list[str] = []
    stage_vars: list[str] = []
    for k, stage in enumerate(stages):
        sv = f"{flag_var}_s{k}"
        stage_vars.append(sv)
        lines.append(f"float {sv} = 0.0")
        lines.append(_emit_regressor_body(stage, sv, indent=""))

    summed = " + ".join(stage_vars) if stage_vars else "0.0"
    lines.append(
        f"float {flag_var}_margin = {_fmt_threshold(init)} "
        f"+ {_fmt_threshold(lr)} * ({summed})"
    )
    # Decision in margin space: proba >= threshold  <=>  margin >= logit(threshold).
    lines.append(f"{flag_var} := {flag_var}_margin >= {_fmt_threshold(logit_thr)}")
    return "\n".join(lines)


def _emit_model_body(model: TreeJson, flag_var: str, threshold: float) -> str:
    """Dispatch to the tree or gboost body emitter on ``model_kind``.

    Args:
        model: A parsed student model dict (tree or gboost).
        flag_var: The Pine bool variable the rule assigns.
        threshold: The probability cut for a gboost (ignored for a tree, whose
            leaves already carry the ``predict`` label).

    Returns:
        The Pine rule block assigning ``flag_var``.
    """
    if _is_gboost(model):
        return _emit_gboost_body(model, flag_var, threshold)
    return _emit_tree_body(model, flag_var, indent="")


def emit_feature_block(feature_names: list[str]) -> str:
    """Emit the de-duplicated per-feature computation block for ``feature_names``.

    Args:
        feature_names: Feature columns used across both trees (order preserved,
            duplicates dropped so a feature shared by both sides is computed once).

    Returns:
        The newline-joined Pine statements computing every listed feature.
    """
    seen: set[str] = set()
    blocks: list[str] = []
    for name in feature_names:
        if name in seen:
            continue
        seen.add(name)
        fp = feature_pine_expr(name)
        blocks.append(f"// {fp.name}\n{fp.code}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Pine preamble (helper functions ported 1:1 from the original indicator).     #
# --------------------------------------------------------------------------- #

_HELPERS: str = """// --- Parity helpers (ported from speculatores_v15_presets_gold.pine) ---------
// One stateless cumulative sum at module scope; SMA of any length is then a
// constant-time subtraction with no per-call-site state to corrupt (L87-93).
float _csum_close = ta.cum(close)

sma_at(int s, int back) =>
    float a = _csum_close[back]
    float b = _csum_close[back + s]
    na(a) or na(b) ? float(na) : (a - b) / s

// pir_of: position of `val` within its trailing `lookback` [min, max] (L68-71).
pir_of(float val, int lookback) =>
    float lo = ta.lowest(val, lookback)
    float hi = ta.highest(val, lookback)
    hi != lo ? (val - lo) / (hi - lo) : 0.5

// pir_for_scale: PIR of the close/SMA(s) ratio scanned over the last `lb` bars,
// reconstructed bar-by-bar from historical access so no stateful ta.* is called
// inside the loop (L98-113).
pir_for_scale(int s, int lb) =>
    float sma_now = sma_at(s, 0)
    float val_now = sma_now > 0 ? close / sma_now : 1.0
    float result = 0.5
    if not na(val_now)
        float lo = val_now
        float hi = val_now
        for back = 1 to lb - 1
            float sma_b = sma_at(s, back)
            float c_b = close[back]
            float r_b = sma_b > 0 ? c_b / sma_b : 1.0
            if not na(r_b)
                lo := math.min(lo, r_b)
                hi := math.max(hi, r_b)
        result := hi != lo ? (val_now - lo) / (hi - lo) : 0.5
    result

safe_log_ratio(float num, float den) =>
    math.log(math.max(num, 1e-10) / math.max(den, 1e-10))

clamp_unit(float val) =>
    math.max(-1.0, math.min(1.0, val))

// calc_har_vol: HAR / Garman-Klass volatility norm (L179-188).
calc_har_vol() =>
    float log_hl = safe_log_ratio(high, low)
    float log_co = safe_log_ratio(close, open)
    float gk_var = math.max(0.5 * log_hl * log_hl - (2.0 * math.log(2.0) - 1.0) * log_co * log_co, 1e-10)
    float gk_weekly = nz(ta.sma(gk_var, 5), gk_var)
    float gk_monthly = nz(ta.sma(gk_var, 22), gk_var)
    float har_forecast = math.max(0.36 * gk_var + 0.28 * gk_weekly + 0.28 * gk_monthly, 1e-10)
    float har_vol_ratio = math.sqrt(math.max(har_forecast, 1e-10)) / math.sqrt(math.max(gk_var, 1e-10))
    float har_vol_norm = clamp_unit((har_vol_ratio - 1.0) / 0.5)
    [har_vol_norm, har_vol_ratio]

// calc_gjr_asym: GJR-GARCH(1,1) leverage asymmetry norm (L159-177).
calc_gjr_asym() =>
    float log_ret = safe_log_ratio(close, nz(close[1], close))
    float r2 = log_ret * log_ret
    float lr_var = math.max(nz(ta.sma(r2, 252), r2), 1e-12)
    float gjr_alpha = 0.03
    float gjr_beta = 0.90
    float gjr_gamma = 0.08
    float gjr_omega = math.max(lr_var * (1.0 - gjr_alpha - gjr_beta - gjr_gamma / 2.0), 1e-12)
    var float gjr_var = na
    var float sym_var = na
    float prev_gjr = nz(gjr_var[1], lr_var)
    float prev_sym = nz(sym_var[1], lr_var)
    float prev_r2 = nz(r2[1], lr_var)
    float leverage = nz(log_ret[1], 0.0) < 0 ? 1.0 : 0.0
    gjr_var := math.max(gjr_omega + (gjr_alpha + gjr_gamma * leverage) * prev_r2 + gjr_beta * prev_gjr, 1e-12)
    sym_var := math.max(gjr_omega + (gjr_alpha + gjr_gamma * 0.5) * prev_r2 + gjr_beta * prev_sym, 1e-12)
    float gjr_asym_ratio = gjr_var / sym_var
    float gjr_asym_norm = clamp_unit((gjr_asym_ratio - 1.0) / 0.1)
    [gjr_asym_norm, gjr_asym_ratio]
"""


def _union_features(low_tree: TreeJson, high_tree: TreeJson) -> list[str]:
    """Union of both trees' used features (low first, then high extras)."""
    low_feats = [str(f) for f in low_tree.get("features", [])]
    high_feats = [str(f) for f in high_tree.get("features", [])]
    out = list(low_feats)
    for f in high_feats:
        if f not in out:
            out.append(f)
    return out


def _model_summary(model: TreeJson, threshold: float) -> str:
    """One-line description of a side's student (tree depth/leaves or gboost size)."""
    if _is_gboost(model):
        n_est = int(model.get("n_estimators", len(model.get("stages", []))))
        lr = float(model.get("learning_rate", 0.0))
        return (
            f"gboost: {n_est} stages, learning_rate={lr:g}, "
            f"margin >= logit({threshold:g})={_logit(threshold):.6f}"
        )
    depth = int(model.get("max_depth", 0))
    leaves = int(model.get("n_leaves", 0))
    return f"tree: depth={depth}, leaves={leaves}"


def emit_indicator(
    low_tree: TreeJson,
    high_tree: TreeJson,
    threshold: float = _GBOOST_DECISION_THRESHOLD,
) -> str:
    """Render the full standalone Pine v6 student indicator (tree OR gboost).

    Each side is transcribed by :func:`_emit_model_body`: a single decision tree
    becomes nested ``if/else`` rules whose leaves set the side flag; a gradient
    boost becomes the additive margin ``init + learning_rate * sum(stage leaf)``
    compared to ``logit(threshold)`` in raw margin space (no sigmoid). The two
    sides may independently be either kind.

    Args:
        low_tree: Parsed LOW-side student JSON (bottom turns).
        high_tree: Parsed HIGH-side student JSON (top turns).
        threshold: Probability cut baked into a gboost side (``0.5`` reproduces
            sklearn ``predict()`` exactly; ignored by a tree side).

    Returns:
        The complete ``@version=6`` indicator source (ASCII only), ready to write
        to ``pine/cfd10_student.pine``.
    """
    feature_block = emit_feature_block(_union_features(low_tree, high_tree))
    low_body = _emit_model_body(low_tree, "student_low", threshold)
    high_body = _emit_model_body(high_tree, "student_high", threshold)

    header = (
        "//@version=6\n"
        'indicator("cfd10 Student - distilled structural-turn detector", '
        "overlay=true, max_bars_back=5000)\n\n"
        "// AUTO-GENERATED by cfd10.parity_module.export_pine -- do not edit by hand.\n"
        "// Two SMALL students distilled from the cfd10 STRUCTURAL oracle (one per\n"
        "// side), transcribed verbatim. A tree side is nested if/else rules; a\n"
        "// gboost side is the additive margin init + learning_rate * sum(stage leaf)\n"
        "// compared to logit(threshold) in RAW MARGIN space (no sigmoid is emitted,\n"
        "// so the call is exact float arithmetic). Each feature is recomputed exactly\n"
        "// as the cfd10 Python feature bank does (FeatureConfig defaults), itself a\n"
        "// parity port of the original speculatores indicator. Internal parity\n"
        "// (emitted rules == sklearn predict()) is proven by\n"
        "// cfd10.parity_module.verify.verify_student_export.\n"
        f"// LOW  {_model_summary(low_tree, threshold)}.\n"
        f"// HIGH {_model_summary(high_tree, threshold)}.\n"
    )

    return (
        f"{header}\n"
        f"{_HELPERS}\n"
        "// --- Feature computations (used by the student trees) -----------------------\n"
        f"{feature_block}\n\n"
        "// --- LOW-pivot rule set (bottom turns) --------------------------------------\n"
        "bool student_low = false\n"
        f"{low_body}\n\n"
        "// --- HIGH-pivot rule set (top turns) ----------------------------------------\n"
        "bool student_high = false\n"
        f"{high_body}\n\n"
        "// --- Markers (mirror the original indicator's pivot triangles) --------------\n"
        "plotshape(student_high and not student_high[1], title=\"Student High\", "
        "style=shape.triangledown, location=location.abovebar, color=color.red, "
        "size=size.normal)\n"
        "plotshape(student_low and not student_low[1], title=\"Student Low\", "
        "style=shape.triangleup, location=location.belowbar, color=color.green, "
        "size=size.normal)\n\n"
        "color bg = na\n"
        "if student_high\n"
        "    bg := color.new(color.red, 80)\n"
        "else if student_low\n"
        "    bg := color.new(color.green, 80)\n"
        "bgcolor(bg)\n\n"
        "alertcondition(student_high and not student_high[1], \"Student High\", "
        "\"cfd10 student: HIGH pivot\")\n"
        "alertcondition(student_low and not student_low[1], \"Student Low\", "
        "\"cfd10 student: LOW pivot\")\n"
    )


def write_indicator(
    low_json: str | Path,
    high_json: str | Path,
    out_path: str | Path = PINE_PATH,
    threshold: float = _GBOOST_DECISION_THRESHOLD,
) -> Path:
    """Load both student JSONs, render the indicator, and write it to ``out_path``.

    Args:
        low_json: Path to the LOW-side student JSON (tree or gboost).
        high_json: Path to the HIGH-side student JSON (tree or gboost).
        out_path: Destination ``.pine`` file (defaults to :data:`PINE_PATH`).
        threshold: Probability cut baked into a gboost side (``0.5`` reproduces
            sklearn ``predict()``); ignored by a tree side.

    Returns:
        The written path.
    """
    low_tree = load_student_json(low_json)
    high_tree = load_student_json(high_json)
    text = emit_indicator(low_tree, high_tree, threshold=threshold)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info(
        "write_indicator: wrote %s (%d chars, %d features)",
        out,
        len(text),
        len(_union_features(low_tree, high_tree)),
    )
    return out
