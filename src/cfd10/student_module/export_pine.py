"""Emit a distilled student tree (JSON) as a standalone Pine v6 indicator.

The deployable detector is a shallow ``DecisionTreeClassifier`` per side
(``outputs/student_{low,high}.json``). This module transcribes both trees into a
single TradingView ``@version=6`` indicator that, on each bar, recomputes only
the handful of features the trees split on, evaluates the two nested ``if/else``
rule sets, and plots a LOW-pivot / HIGH-pivot marker — mirroring the marker style
of the original ``pine/speculatores_v15_presets_gold.pine``.

Parity contract
---------------
Every emitted feature recomputation matches the *Python* feature bank
(:mod:`cfd10.feature_module`) bar-for-bar, which in turn is a faithful port of
the original Pine semantics:

* ``price_return_L{L}`` / ``mom_divergence_L{L}`` — Pine L443-451 momentum block.
* ``er_dir_p{P}`` / ``er_abs_p{P}`` — Pine L468-480 Kaufman efficiency ratio.
* ``sma_slope_s{S}`` / ``linreg_slope_s{S}`` — Pine L402-413 trend slopes, with the
  ``ta.cum`` SMA idiom (L87-93) and the ``d = max(round(S/4), 2)`` lag.
* ``vola_pos_{METHOD}_l{L}`` — Pine L456-457 raw volatility + ``pir_of`` position.
* ``har_vol`` — Pine L179-188 ``calc_har_vol``.
* ``pir_s{S}`` — Pine L98-126 ``pir_for_scale`` (the ``ta.cum`` agreement idiom).

The feature-bank *constants* are baked in from :class:`FeatureConfig` defaults
(``pir_lb_floor=20``, ``vola_range_len=100``, the slope ``* 1000`` scaling),
*not* from the original indicator's per-preset inputs: the student was trained on
the cfd10 feature bank, so the export must reproduce that bank, not the legacy
voting detector.

Public API
----------
:func:`load_student_json` — read a tree JSON written by ``distill_students.py``.
:func:`emit_indicator` — render the full two-side Pine v6 indicator string.
:func:`write_indicator` — render + write ``pine/cfd10_student.pine``.
:func:`feature_pine_expr` — the Pine variable name + computation for one feature
(also drives the parity contract doc).
"""

from __future__ import annotations

import json
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
    if name.startswith("pir_s"):
        return FeaturePine(name, var, _emit_pir_s(name, var))
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


def load_student_json(path: str | Path) -> TreeJson:
    """Load and minimally validate a student tree JSON.

    Args:
        path: Path to ``student_{side}.json`` (written by ``distill_students.py``).

    Returns:
        The parsed tree dict (keys ``nodes``, ``features``, ...).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON lacks the expected ``nodes`` array.
    """
    p = Path(path)
    tree = json.loads(p.read_text(encoding="utf-8"))
    if "nodes" not in tree or not isinstance(tree["nodes"], list):
        raise ValueError(f"load_student_json: {p} has no 'nodes' array")
    return tree


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


def emit_indicator(low_tree: TreeJson, high_tree: TreeJson) -> str:
    """Render the full standalone Pine v6 student indicator.

    Args:
        low_tree: Parsed LOW-side student tree JSON (bottom turns).
        high_tree: Parsed HIGH-side student tree JSON (top turns).

    Returns:
        The complete ``@version=6`` indicator source (ASCII only), ready to write
        to ``pine/cfd10_student.pine``.
    """
    feature_block = emit_feature_block(_union_features(low_tree, high_tree))
    low_body = _emit_tree_body(low_tree, "student_low", indent="")
    high_body = _emit_tree_body(high_tree, "student_high", indent="")

    low_d = int(low_tree.get("max_depth", 0))
    low_l = int(low_tree.get("n_leaves", 0))
    high_d = int(high_tree.get("max_depth", 0))
    high_l = int(high_tree.get("n_leaves", 0))

    header = (
        "//@version=6\n"
        'indicator("cfd10 Student - distilled structural-turn detector", '
        "overlay=true, max_bars_back=5000)\n\n"
        "// AUTO-GENERATED by cfd10.parity_module.export_pine -- do not edit by hand.\n"
        "// Two shallow decision trees distilled from the cfd10 STRUCTURAL oracle\n"
        "// (one per side), transcribed verbatim as nested if/else rule sets. Each\n"
        "// feature is recomputed exactly as the cfd10 Python feature bank does\n"
        "// (FeatureConfig defaults), which is itself a parity port of the original\n"
        "// speculatores indicator. Internal parity (emitted rules == sklearn tree)\n"
        "// is proven by cfd10.parity_module.verify.verify_student_export.\n"
        f"// LOW  tree: depth={low_d}, leaves={low_l}.\n"
        f"// HIGH tree: depth={high_d}, leaves={high_l}.\n"
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
) -> Path:
    """Load both student JSONs, render the indicator, and write it to ``out_path``.

    Args:
        low_json: Path to the LOW-side student JSON.
        high_json: Path to the HIGH-side student JSON.
        out_path: Destination ``.pine`` file (defaults to :data:`PINE_PATH`).

    Returns:
        The written path.
    """
    low_tree = load_student_json(low_json)
    high_tree = load_student_json(high_json)
    text = emit_indicator(low_tree, high_tree)
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
