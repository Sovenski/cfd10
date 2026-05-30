"""Tests for ``cfd10.parity_module.verify`` and the Pine export thresholds.

The contract under test is *internal parity*: the rules emitted into the Pine
indicator are an exact transcription of the fitted sklearn tree. We prove it two
ways on small, fast fixtures (no real data, no heavy fit):

* a real shallow ``DecisionTreeClassifier`` (here a 3-leaf tree) is serialised to
  the same node-JSON schema the pipeline writes, and :func:`simulate_tree` /
  :func:`verify_student_export` re-simulate that JSON straight off the node array
  — the result must equal ``tree.predict()`` on every random row (the re-sim *is*
  what the Pine ``if/else`` computes); and
* the generated ``.pine`` text must contain each split threshold verbatim at full
  round-trip precision, so the on-chart rule boundary is byte-identical to the
  tree's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, _tree

from cfd10.student_module.distill import _gboost_init_raw
from cfd10.student_module.export_pine import (
    emit_indicator,
    feature_pine_expr,
)
from cfd10.parity_module.verify import (
    simulate_gboost,
    simulate_tree,
    verify_student_export,
)

_FEATURES = ["alpha", "beta", "gamma"]

# Real cfd10 feature names, used when the test must drive the Pine emitter (which
# only knows the genuine feature families). Mapped onto the same 3-leaf fixture.
_REAL_FEATURES = ["price_return_L20", "er_dir_p10", "har_vol"]

# The full union of feature families the re-distilled v2 students split on, drawn
# from the 45-feature +overext bank. Every entry must have a Pine emitter.
_V2_STUDENT_FEATURES = [
    # momentum
    "price_return_L10", "price_return_L20", "price_return_L40",
    "mom_divergence_L20", "mom_divergence_L40",
    "mom_velocity_L10", "mom_velocity_L20", "mom_velocity_L40",
    # efficiency
    "er_dir_p10", "er_dir_p20", "er_dir_p40", "er_abs_p20", "er_abs_p40",
    # trend
    "sma_slope_s20", "sma_slope_s50", "sma_slope_s100",
    "linreg_slope_s20", "linreg_slope_s50", "linreg_slope_s100",
    # vola position
    "vola_pos_ATR_l30", "vola_pos_StdDev_l14", "vola_pos_StdDev_l30",
    "vola_pos_Intraday_l14", "vola_pos_Intraday_l30",
    # pir / agreement / garch-har
    "pir_s5", "pir_s10", "pir_s20", "pir_s50", "pir_s100",
    "agree_high", "har_vol", "gjr_asym",
    # overextension / vol-regime
    "dist_above_sma_z_l50", "dist_above_sma_z_l100", "dist_above_sma_z_l200",
    "drawdown_from_high_l20", "drawdown_from_high_l100",
    "realized_vol_pct", "up_streak_norm",
]


# --------------------------------------------------------------------------- #
# Helper: serialise a fitted tree to the pipeline's node-JSON schema.          #
# --------------------------------------------------------------------------- #


def _tree_to_json(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict:
    """Serialise a fitted tree to the ``distill_students`` node-JSON schema.

    Mirrors ``pipeline/distill_students._tree_to_json`` exactly (same field names
    and the ``predict = int(pos_proba >= 0.5)`` leaf rule) so the test exercises
    the real export contract without importing the pipeline script.

    Args:
        tree: A fitted decision tree.
        feature_names: Column names aligned to the training matrix.

    Returns:
        A JSON-serialisable tree dict consumed by the parity / export code.
    """
    inner = tree.tree_
    classes = list(tree.classes_)
    pos_col = classes.index(1) if 1 in classes else -1

    nodes: list[dict] = []
    used: list[str] = []
    seen: set[int] = set()
    for node in range(inner.node_count):
        feat = int(inner.feature[node])
        is_leaf = feat == _tree.TREE_UNDEFINED
        counts = inner.value[node][0]
        total = float(counts.sum())
        pos_mass = float(counts[pos_col]) if pos_col >= 0 else 0.0
        prob = pos_mass / total if total > 0.0 else 0.0
        if not is_leaf and feat not in seen:
            seen.add(feat)
            used.append(feature_names[feat])
        nodes.append(
            {
                "node_id": node,
                "is_leaf": is_leaf,
                "feature": None if is_leaf else feature_names[feat],
                "feature_index": None if is_leaf else feat,
                "threshold": None if is_leaf else float(inner.threshold[node]),
                "left": None if is_leaf else int(inner.children_left[node]),
                "right": None if is_leaf else int(inner.children_right[node]),
                "n_samples": int(inner.n_node_samples[node]),
                "pos_proba": round(prob, 6),
                "predict": int(prob >= 0.5),
            }
        )
    return {
        "n_nodes": int(inner.node_count),
        "max_depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "features": used,
        "nodes": nodes,
    }


def _three_leaf_tree(
    n: int = 600, seed: int = 0
) -> tuple[DecisionTreeClassifier, pd.DataFrame, dict]:
    """Fit a 3-leaf tree on a fixture that forces two splits, return (tree, X, json).

    ``y`` depends on two features (``alpha`` then ``beta``), so a tree capped at
    ``max_leaf_nodes=3`` carves exactly three leaves: one on the ``alpha`` split
    and two under the ``beta`` split.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(n)
    beta = rng.standard_normal(n)
    gamma = rng.standard_normal(n)  # pure noise: must never appear in the tree.
    # Two-level structure: alpha gates the first split, beta the second.
    y = np.where(alpha > 0.3, 1, np.where(beta > -0.2, 1, 0)).astype(np.int64)
    X = pd.DataFrame({"alpha": alpha, "beta": beta, "gamma": gamma})
    # Fit on the bare array so the tree stores no feature names (it is addressed
    # purely positionally by the JSON / parity code); avoids a benign sklearn
    # "X does not have valid feature names" warning at predict time.
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=seed).fit(
        X.to_numpy(), y
    )
    return tree, X, _tree_to_json(tree, _FEATURES)


def _three_leaf_tree_real_features(
    seed: int = 1,
) -> tuple[DecisionTreeClassifier, dict]:
    """Same 3-leaf fixture but serialised with real cfd10 feature names.

    The Pine emitter only knows genuine feature families, so any test that calls
    :func:`emit_indicator` must label the tree with real names. ``alpha`` ->
    ``price_return_L20``, ``beta`` -> ``er_dir_p10`` (both used as splits by the
    fixture); ``gamma`` -> ``har_vol`` (noise, never split on).
    """
    tree, _X, _json_generic = _three_leaf_tree(seed=seed)
    return tree, _tree_to_json(tree, _REAL_FEATURES)


# --------------------------------------------------------------------------- #
# simulate_tree / verify_student_export round-trip.                            #
# --------------------------------------------------------------------------- #


def test_three_leaf_tree_is_actually_three_leaves() -> None:
    """The fixture really exercises a 3-leaf (two-split) tree."""
    tree, _X, tree_json = _three_leaf_tree()
    assert tree.get_n_leaves() == 3
    assert tree_json["n_leaves"] == 3
    n_internal = sum(0 if node["is_leaf"] else 1 for node in tree_json["nodes"])
    assert n_internal == 2  # exactly two ``feature <= threshold`` splits.


def test_simulate_tree_matches_sklearn_predict_on_random_data() -> None:
    """Re-simulated emitted rules reproduce ``tree.predict()`` exactly (random X)."""
    tree, X, tree_json = _three_leaf_tree()
    rng = np.random.default_rng(123)
    # Fresh random probe rows (not the training rows) over the full feature space.
    probe = pd.DataFrame(
        {
            "alpha": rng.standard_normal(2000),
            "beta": rng.standard_normal(2000),
            "gamma": rng.standard_normal(2000),
        }
    )
    sim = simulate_tree(tree_json, probe.to_numpy(), list(probe.columns))
    expected = tree.predict(probe.to_numpy())
    assert np.array_equal(sim, expected)


def test_verify_student_export_exact_against_fitted_tree() -> None:
    """`verify_student_export` reports an exact match against the sklearn tree."""
    tree, X, tree_json = _three_leaf_tree()
    report = verify_student_export(tree_json, X, sklearn_tree=tree)
    assert report.exact is True
    assert report.n_disagree == 0
    assert report.n_rows == len(X)
    assert report.n_pred_pos_sim == report.n_pred_pos_sklearn


def test_verify_accepts_array_with_feature_names() -> None:
    """A bare array round-trips when ``feature_names`` is supplied explicitly."""
    tree, X, tree_json = _three_leaf_tree()
    report = verify_student_export(
        tree_json, X.to_numpy(), sklearn_tree=tree, feature_names=_FEATURES
    )
    assert report.exact is True


def test_verify_requires_feature_names_for_array() -> None:
    """Passing a bare array without ``feature_names`` is a usage error."""
    _tree, X, tree_json = _three_leaf_tree()
    try:
        verify_student_export(tree_json, X.to_numpy())
    except ValueError as exc:
        assert "feature_names" in str(exc)
    else:  # pragma: no cover - the call must raise.
        raise AssertionError("expected ValueError for array X without feature_names")


def test_verify_detects_corrupted_rules() -> None:
    """Flipping a leaf label makes the re-sim disagree with the sklearn tree."""
    tree, X, tree_json = _three_leaf_tree()
    # Corrupt the export: flip every leaf's predicted class.
    for node in tree_json["nodes"]:
        if node["is_leaf"]:
            node["predict"] = 1 - int(node["predict"])
    try:
        verify_student_export(tree_json, X, sklearn_tree=tree)
    except ValueError as exc:
        assert "faithful" in str(exc) or "disagree" in str(exc)
    else:  # pragma: no cover - corruption must be caught.
        raise AssertionError("expected ValueError on corrupted rules")


# --------------------------------------------------------------------------- #
# Pine export contains the exact thresholds.                                   #
# --------------------------------------------------------------------------- #


def test_pine_output_contains_exact_thresholds() -> None:
    """Every split threshold appears verbatim (full precision) in the .pine text."""
    _low_tree, low_json = _three_leaf_tree_real_features(seed=1)
    _high_tree, high_json = _three_leaf_tree_real_features(seed=2)
    pine = emit_indicator(low_json, high_json)

    for tree_json in (low_json, high_json):
        for node in tree_json["nodes"]:
            if node["is_leaf"]:
                continue
            threshold_text = repr(float(node["threshold"]))
            assert threshold_text in pine, (
                f"threshold {threshold_text} for feature {node['feature']!r} "
                "missing from emitted Pine"
            )


def test_pine_output_is_version6_indicator_with_both_sides() -> None:
    """The emitted script is a v6 indicator plotting both pivot sides."""
    _low_tree, low_json = _three_leaf_tree_real_features(seed=1)
    _high_tree, high_json = _three_leaf_tree_real_features(seed=2)
    pine = emit_indicator(low_json, high_json)

    assert pine.startswith("//@version=6")
    assert "indicator(" in pine
    assert "bool student_low = false" in pine
    assert "bool student_high = false" in pine
    assert 'title="Student Low"' in pine
    assert 'title="Student High"' in pine
    # ASCII-only (Windows cp1252 safe).
    pine.encode("ascii")


# --------------------------------------------------------------------------- #
# feature_pine_expr emitter.                                                   #
# --------------------------------------------------------------------------- #


def test_feature_pine_expr_covers_all_student_families() -> None:
    """Every feature family the real students use has a Pine emitter."""
    student_features = [
        "price_return_L40",
        "er_dir_p10",
        "er_abs_p20",
        "sma_slope_s100",
        "linreg_slope_s50",
        "linreg_slope_s100",
        "har_vol",
        "vola_pos_ATR_l14",
        "vola_pos_StdDev_l30",
        "mom_divergence_L10",
        "mom_divergence_L20",
        "pir_s100",
    ]
    for name in student_features:
        fp = feature_pine_expr(name)
        assert fp.name == name
        assert fp.var.startswith("f_")
        assert fp.code  # non-empty Pine.


def test_feature_pine_expr_rejects_unknown_feature() -> None:
    """An unmapped feature name is a hard error (no silent skip)."""
    try:
        feature_pine_expr("totally_unknown_feature")
    except ValueError as exc:
        assert "no Pine emitter" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown feature")


def test_feature_pine_expr_covers_full_v2_union() -> None:
    """Every feature family the re-distilled v2 students use has a Pine emitter.

    The v2 students draw from the 45-feature +overext bank, which adds families
    the original depth-4 export never saw: momentum velocity, multi-scale
    agreement, the GJR-GARCH asymmetry, and the whole overextension / vol-regime
    block (squashed z-distance, drawdown-from-high, realized-vol percentile,
    up-streak). Each must emit non-empty, ``f_``-prefixed Pine.
    """
    for name in _V2_STUDENT_FEATURES:
        fp = feature_pine_expr(name)
        assert fp.name == name
        assert fp.var.startswith("f_")
        assert fp.code  # non-empty Pine.
        fp.code.encode("ascii")  # ASCII-only (Windows cp1252 safe).


def test_overextension_emitters_reference_correct_constants() -> None:
    """The overextension Pine bakes the FeatureConfig() default windows."""
    # dist_above_sma_z normalises by ta.stdev over oe_z_win = 100.
    assert "ta.stdev(close, 100)" in feature_pine_expr("dist_above_sma_z_l50").code
    # realized_vol_pct: rolling vol over oe_rv_win = 20, PIR over oe_rv_range_len = 100.
    rv = feature_pine_expr("realized_vol_pct").code
    assert "ta.stdev(" in rv and ", 20)" in rv and "pir_of(" in rv and ", 100)" in rv
    # up_streak_norm saturates at oe_streak_cap = 5.
    assert "math.min(" in feature_pine_expr("up_streak_norm").code
    assert "5)" in feature_pine_expr("up_streak_norm").code
    # drawdown_from_high uses ta.highest over the named lookback.
    assert "ta.highest(close, 100)" in feature_pine_expr("drawdown_from_high_l100").code


# --------------------------------------------------------------------------- #
# Gradient-boosting margin path: simulate_gboost + verify_student_export.      #
# --------------------------------------------------------------------------- #


def _gboost_to_json(
    model: GradientBoostingClassifier, feature_names: list[str]
) -> dict:
    """Serialise a fitted gboost to the ``redistill_students`` v2 JSON schema.

    Mirrors ``pipeline/redistill_students._model_to_json`` exactly for the gboost
    branch: ``model_kind="gboost"`` with ``init`` (prior log-odds), ``learning_rate``
    and a ``stages`` list of per-estimator regressor node arrays (leaves carry
    ``value``). Built here so the test exercises the real export contract without
    importing the pipeline script.
    """
    stages: list[list[dict]] = []
    for s in range(model.n_estimators_):
        inner = model.estimators_[s, 0].tree_
        nodes: list[dict] = []
        for node in range(inner.node_count):
            feat = int(inner.feature[node])
            is_leaf = feat == _tree.TREE_UNDEFINED
            nodes.append(
                {
                    "node_id": node,
                    "is_leaf": is_leaf,
                    "feature": None if is_leaf else feature_names[feat],
                    "feature_index": None if is_leaf else feat,
                    "threshold": None if is_leaf else float(inner.threshold[node]),
                    "left": None if is_leaf else int(inner.children_left[node]),
                    "right": None if is_leaf else int(inner.children_right[node]),
                    "n_samples": int(inner.n_node_samples[node]),
                    "value": float(inner.value[node][0][0]),
                }
            )
        stages.append(nodes)
    used: list[str] = []
    seen: set[int] = set()
    for stage_nodes in stages:
        for node in stage_nodes:
            if not node["is_leaf"] and node["feature_index"] not in seen:
                seen.add(node["feature_index"])
                used.append(node["feature"])
    return {
        "model_kind": "gboost",
        "n_estimators": int(model.n_estimators_),
        "learning_rate": float(model.learning_rate),
        "init": round(_gboost_init_raw(model), 8),
        "decision": "positive iff init + learning_rate * sum(stage leaf) > 0",
        "features": used,
        "stages": stages,
    }


def _small_gboost(
    n: int = 2000, seed: int = 0
) -> tuple[GradientBoostingClassifier, np.ndarray, np.ndarray, dict]:
    """Fit a small gboost on a two-feature fixture; return (model, X, y, json).

    ``y`` depends on two features so the shallow stage trees actually split; the
    ensemble is kept tiny (Pine-portable). Fit on a bare array so the model stores
    no feature names (addressed positionally by the JSON / parity code), avoiding
    the benign sklearn feature-name warning at predict time.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(n)
    beta = rng.standard_normal(n)
    gamma = rng.standard_normal(n)  # noise; may or may not be split on.
    y = np.where(alpha > 0.2, 1, np.where(beta > 0.0, 1, 0)).astype(np.int64)
    X = np.column_stack([alpha, beta, gamma])
    model = GradientBoostingClassifier(
        n_estimators=12, max_depth=2, learning_rate=0.1, min_samples_leaf=30,
        random_state=seed,
    ).fit(X, y)
    return model, X, y, _gboost_to_json(model, _FEATURES)


def test_gboost_json_is_well_formed() -> None:
    """The gboost fixture serialises to the v2 schema the export/verify consume."""
    _model, _X, _y, mj = _small_gboost()
    assert mj["model_kind"] == "gboost"
    assert isinstance(mj["stages"], list) and len(mj["stages"]) == 12
    assert "init" in mj and "learning_rate" in mj
    # Every stage is a node array whose leaves carry a raw ``value``.
    for stage in mj["stages"]:
        assert all(("value" in node) for node in stage)


def test_simulate_gboost_margin_matches_decision_function() -> None:
    """Re-simulated additive margin equals the booster's decision_function.

    The only gap is the JSON's deliberate 8-decimal rounding of the ``init``
    log-odds (``redistill_students._model_to_json``), which shifts every margin by
    a constant <= 5e-9; the per-row reconstruction of ``learning_rate * sum(stage
    leaf)`` is otherwise exact. So the margin agrees to that rounding bound, well
    under 1e-7.
    """
    model, _X, _y, mj = _small_gboost()
    rng = np.random.default_rng(7)
    probe = np.column_stack(
        [rng.standard_normal(3000), rng.standard_normal(3000), rng.standard_normal(3000)]
    )
    margin = simulate_gboost(mj, probe, _FEATURES)
    raw = model.decision_function(probe).reshape(-1)
    assert margin.shape == raw.shape
    # The error is the constant init-rounding shift (<= 5e-9); use full-precision
    # init to confirm the *reconstruction* is exact to floating point.
    assert float(np.max(np.abs(margin - raw))) < 1e-7
    mj_exact = dict(mj, init=float(_gboost_init_raw(model)))
    margin_exact = simulate_gboost(mj_exact, probe, _FEATURES)
    assert float(np.max(np.abs(margin_exact - raw))) < 1e-9


def test_verify_gboost_signal_is_exact_against_sklearn_predict() -> None:
    """The baked margin>=logit(0.5) signal equals gboost predict() on every row."""
    model, X, _y, mj = _small_gboost()
    report = verify_student_export(mj, X, sklearn_tree=model, feature_names=_FEATURES)
    assert report.model_kind == "gboost"
    # The BINARY SIGNAL is exact: every row's margin>=logit(0.5) call equals predict().
    assert report.exact is True
    assert report.n_disagree == 0
    assert report.n_pred_pos_sim == report.n_pred_pos_sklearn
    # Margin error is just the 8-dp init-rounding shift (<= 5e-9), reported for info.
    assert report.margin_max_abs_err < 1e-7


def test_verify_gboost_detects_corrupted_leaf_value() -> None:
    """Perturbing a stage leaf value far enough flips the signal -> disagreement."""
    model, X, _y, mj = _small_gboost()
    # Inflate every leaf value massively: the margin shoots positive everywhere, so
    # the signal saturates to all-ones and must disagree with sklearn predict().
    for stage in mj["stages"]:
        for node in stage:
            if node["is_leaf"]:
                node["value"] = float(node["value"]) + 1000.0
    try:
        verify_student_export(mj, X, sklearn_tree=model, feature_names=_FEATURES)
    except ValueError as exc:
        assert "faithful" in str(exc) or "disagree" in str(exc)
    else:  # pragma: no cover - corruption must be caught.
        raise AssertionError("expected ValueError on corrupted gboost leaves")


def test_gboost_pine_emits_raw_margin_threshold_no_sigmoid() -> None:
    """The gboost Pine compares the raw margin to logit(thr) and emits NO sigmoid."""
    _low, _Xl, _yl, low_json = _small_gboost(seed=1)
    _high, _Xh, _yh, high_json = _small_gboost(seed=2)
    # Relabel both with real feature names so the emitter recognises the families.
    for mj in (low_json, high_json):
        for stage in mj["stages"]:
            for node in stage:
                if not node["is_leaf"]:
                    node["feature"] = _REAL_FEATURES[int(node["feature_index"])]
        mj["features"] = [
            _REAL_FEATURES[i]
            for i in sorted({
                node["feature_index"]
                for stage in mj["stages"]
                for node in stage
                if not node["is_leaf"]
            })
        ]
    pine = emit_indicator(low_json, high_json, threshold=0.5)

    assert pine.startswith("//@version=6")
    # Additive margin + raw-margin decision, both sides.
    assert "student_low_margin" in pine
    assert "student_high_margin" in pine
    assert "student_low := student_low_margin >= 0.0" in pine
    assert "student_high := student_high_margin >= 0.0" in pine
    # The DECISION must be the raw margin compared to the baked logit cut -- never a
    # sigmoid/logistic applied to the margin. Check the executable lines only (the
    # header comment legitimately contains the word "sigmoid").
    code_lines = [ln for ln in pine.splitlines() if not ln.lstrip().startswith("//")]
    code = "\n".join(code_lines)
    assert "sigmoid" not in code.lower()
    assert "1.0 / (1.0 + " not in code  # no inline logistic of the margin
    assert "math.exp(-" not in code  # no logistic of a negated margin
    pine.encode("ascii")  # ASCII-only.


def test_gboost_pine_threshold_is_baked_in_logit_space() -> None:
    """A non-0.5 threshold is baked as logit(threshold) into the decision line."""
    from cfd10.student_module.export_pine import _logit

    _m, _X, _y, mj = _small_gboost(seed=3)
    for stage in mj["stages"]:
        for node in stage:
            if not node["is_leaf"]:
                node["feature"] = _REAL_FEATURES[int(node["feature_index"])]
    mj["features"] = [_REAL_FEATURES[0]]
    pine = emit_indicator(mj, mj, threshold=0.7)
    # The export bakes repr(_logit(0.7)); it must appear verbatim as the cut and
    # differ from the 0.5 default (logit 0.0), confirming the threshold is honoured.
    expected = repr(_logit(0.7))
    assert expected in pine
    assert "student_low := student_low_margin >= " + expected in pine
