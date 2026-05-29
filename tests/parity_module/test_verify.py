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
from sklearn.tree import DecisionTreeClassifier, _tree

from cfd10.student_module.export_pine import (
    emit_indicator,
    feature_pine_expr,
)
from cfd10.parity_module.verify import simulate_tree, verify_student_export

_FEATURES = ["alpha", "beta", "gamma"]

# Real cfd10 feature names, used when the test must drive the Pine emitter (which
# only knows the genuine feature families). Mapped onto the same 3-leaf fixture.
_REAL_FEATURES = ["price_return_L20", "er_dir_p10", "har_vol"]


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
