"""Tests for ``cfd10.student_module.distill`` — the shallow Pine-export student.

These run on tiny synthetic fixtures (no real data, no GBDT fit), so the suite is
fast. We assert the contract, not a specific score:

* :func:`fit_student` recovers a *separable* toy label with a shallow tree and
  respects the depth / leaf capacity caps;
* :func:`student_features` returns exactly the feature(s) the tree splits on, in
  first-use order, and is empty for a degenerate single-leaf tree;
* :func:`tree_to_rules` renders the actual split feature, threshold and branch
  structure as plain ASCII nested ``if`` rules;
* :func:`distill_from_teacher` mimics a teacher's thresholded soft calls and
  rejects a single-class collapse;
* the input-validation guards fire on malformed inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from cfd10.student_module import (
    StudentConfig,
    distill_from_teacher,
    ensemble_features,
    ensemble_to_rules,
    fit_student,
    student_features,
    tree_to_rules,
)

_FEATURES = ["alpha", "beta", "gamma"]


def _separable_fixture(
    n: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A linearly separable toy: ``y = 1`` iff ``alpha > 0``; beta/gamma noise.

    A single threshold on ``alpha`` separates the classes perfectly, so a
    depth-1 tree can reach 100% training accuracy. Returns ``(X, y, weight)``
    with the oracle-style weight ``w = 1 + score`` on positives.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(n)
    y = (alpha > 0.0).astype(np.int64)
    X = pd.DataFrame(
        {
            "alpha": alpha,
            "beta": rng.standard_normal(n),
            "gamma": rng.standard_normal(n),
        }
    )
    sample_weight = np.where(y == 1, 1.0 + rng.uniform(0.0, 2.0, n), 1.0).astype(
        np.float64
    )
    return X, y, sample_weight


def _xor_fixture(
    n: int = 600, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A two-feature XOR toy: ``y = 1`` iff ``sign(alpha) != sign(beta)``.

    No single threshold separates the classes, so a useful model must split on
    *both* ``alpha`` and ``beta`` (a depth-1 tree cannot beat the base rate). This
    lets us assert that a depth>1 tree and a gradient-boosting ensemble actually
    use more than one feature. ``gamma`` is pure noise. Returns ``(X, y, weight)``
    with the oracle-style positive boost folded into the weight.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(n)
    beta = rng.standard_normal(n)
    y = ((alpha > 0.0) ^ (beta > 0.0)).astype(np.int64)
    X = pd.DataFrame(
        {"alpha": alpha, "beta": beta, "gamma": rng.standard_normal(n)}
    )
    sample_weight = np.where(y == 1, 1.0 + rng.uniform(0.0, 2.0, n), 1.0).astype(
        np.float64
    )
    return X, y, sample_weight


# --------------------------------------------------------------------------- #
# Config.                                                                      #
# --------------------------------------------------------------------------- #


def test_config_defaults_and_validation() -> None:
    """`StudentConfig` carries the brief's defaults and rejects bad capacity."""
    cfg = StudentConfig()
    assert cfg.kind == "tree"
    assert cfg.max_depth == 4
    assert cfg.max_leaves == 16
    assert cfg.min_samples_leaf == 30
    # Gradient-boosting defaults (small, Pine-portable).
    assert cfg.n_estimators == 40
    assert cfg.gb_max_depth == 3
    assert cfg.learning_rate == 0.1

    with pytest.raises(ValueError):
        StudentConfig(max_depth=0)
    with pytest.raises(ValueError):
        StudentConfig(max_leaves=1)
    with pytest.raises(ValueError):
        StudentConfig(min_samples_leaf=0)


def test_config_gboost_and_depth_bounds() -> None:
    """`kind`/gboost knobs validate, and tree depth is capped at 8 for Pine."""
    gb = StudentConfig(kind="gboost", n_estimators=80, gb_max_depth=2, learning_rate=0.05)
    assert gb.kind == "gboost"
    assert gb.gboost_kwargs()["n_estimators"] == 80
    assert gb.gboost_kwargs()["max_depth"] == 2

    # Depth up to 8 is allowed for trees; 9 is rejected (Pine portability).
    assert StudentConfig(kind="tree", max_depth=8).max_depth == 8
    with pytest.raises(ValueError):
        StudentConfig(max_depth=9)

    with pytest.raises(ValueError):
        StudentConfig(kind="forest")  # unknown kind.
    with pytest.raises(ValueError):
        StudentConfig(kind="gboost", n_estimators=0)
    with pytest.raises(ValueError):
        StudentConfig(kind="gboost", gb_max_depth=0)
    with pytest.raises(ValueError):
        StudentConfig(kind="gboost", learning_rate=0.0)


def test_tree_kwargs_drops_class_weight() -> None:
    """The student relies on sample_weight only -- no class_weight balancing."""
    assert "class_weight" not in StudentConfig().tree_kwargs()


# --------------------------------------------------------------------------- #
# fit_student.                                                                 #
# --------------------------------------------------------------------------- #


def test_fit_student_learns_separable_toy() -> None:
    """A shallow tree recovers the separable label near-perfectly."""
    X, y, w = _separable_fixture()
    cfg = StudentConfig(max_depth=3, max_leaves=8, min_samples_leaf=5)
    tree = fit_student(X, y, w, cfg)

    pred = tree.predict(X)
    accuracy = float((pred == y).mean())
    assert accuracy > 0.95, f"separable toy only {accuracy:.3f} train accuracy"


def test_fit_student_respects_capacity_caps() -> None:
    """The fitted tree never exceeds the configured depth / leaf caps."""
    X, y, w = _separable_fixture(seed=3)
    cfg = StudentConfig(max_depth=4, max_leaves=16, min_samples_leaf=5)
    tree = fit_student(X, y, w, cfg)

    assert tree.get_depth() <= cfg.max_depth
    assert tree.get_n_leaves() <= cfg.max_leaves


def test_fit_student_no_weight_runs() -> None:
    """``sample_weight=None`` is accepted (uniform weighting)."""
    X, y, _ = _separable_fixture(seed=5)
    tree = fit_student(X, y, None, StudentConfig(max_depth=2, min_samples_leaf=5))
    assert tree.get_depth() >= 1


def test_fit_student_rejects_bad_inputs() -> None:
    """Length-mismatch and non-binary-label guards fire."""
    X, y, w = _separable_fixture(n=120)
    cfg = StudentConfig(min_samples_leaf=5)

    with pytest.raises(ValueError):  # y too short.
        fit_student(X, y[:-1], w, cfg)
    with pytest.raises(ValueError):  # weight too short.
        fit_student(X, y, w[:-1], cfg)
    with pytest.raises(ValueError):  # non-binary label.
        bad = y.copy()
        bad[0] = 2
        fit_student(X, bad, w, cfg)


def test_fit_student_deep_tree_uses_both_signals() -> None:
    """A depth>4 tree on the XOR toy learns it and splits on both signals."""
    X, y, w = _xor_fixture()
    cfg = StudentConfig(kind="tree", max_depth=6, max_leaves=32, min_samples_leaf=5)
    tree = fit_student(X, y, w, cfg)

    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.get_depth() <= cfg.max_depth
    accuracy = float((tree.predict(X) == y).mean())
    assert accuracy > 0.9, f"deep tree only {accuracy:.3f} on the XOR toy"
    used = student_features(tree, _FEATURES)
    assert {"alpha", "beta"} <= set(used), f"XOR needs both signals, got {used}"


# --------------------------------------------------------------------------- #
# fit_student: kind="gboost".                                                  #
# --------------------------------------------------------------------------- #


def test_fit_student_gboost_learns_separable_toy() -> None:
    """A small gradient-boosting student recovers a separable label."""
    X, y, w = _separable_fixture(seed=11)
    cfg = StudentConfig(
        kind="gboost", n_estimators=40, gb_max_depth=2, learning_rate=0.1,
        min_samples_leaf=5,
    )
    model = fit_student(X, y, w, cfg)

    assert isinstance(model, GradientBoostingClassifier)
    # One regression tree per stage for a binary booster.
    assert model.estimators_.shape == (cfg.n_estimators, 1)
    accuracy = float((model.predict(X) == y).mean())
    assert accuracy > 0.95, f"gboost student only {accuracy:.3f} train accuracy"


def test_fit_student_gboost_learns_xor() -> None:
    """The ensemble solves XOR (a single tree split cannot) -> a real ensemble win."""
    X, y, w = _xor_fixture(seed=2)
    cfg = StudentConfig(
        kind="gboost", n_estimators=60, gb_max_depth=2, learning_rate=0.2,
        min_samples_leaf=5,
    )
    model = fit_student(X, y, w, cfg)
    accuracy = float((model.predict(X) == y).mean())
    assert accuracy > 0.9, f"gboost only {accuracy:.3f} on the XOR toy"


def test_fit_student_gboost_no_weight_runs() -> None:
    """``sample_weight=None`` is accepted for the gradient-boosting kind too."""
    X, y, _ = _separable_fixture(seed=13)
    model = fit_student(
        X, y, None, StudentConfig(kind="gboost", n_estimators=10, gb_max_depth=2)
    )
    assert isinstance(model, GradientBoostingClassifier)
    assert model.estimators_.shape[0] == 10


def test_fit_student_gboost_rejects_bad_inputs() -> None:
    """The shared input guards fire regardless of kind."""
    X, y, w = _separable_fixture(n=150, seed=4)
    cfg = StudentConfig(kind="gboost", n_estimators=10, gb_max_depth=2, min_samples_leaf=5)

    with pytest.raises(ValueError):  # weight too short.
        fit_student(X, y, w[:-1], cfg)
    with pytest.raises(ValueError):  # non-binary label.
        bad = y.copy()
        bad[0] = 5
        fit_student(X, bad, w, cfg)


# --------------------------------------------------------------------------- #
# ensemble_features / ensemble_to_rules.                                       #
# --------------------------------------------------------------------------- #


def test_ensemble_features_is_union_over_estimators() -> None:
    """``ensemble_features`` returns the de-duplicated union of stage splits."""
    X, y, w = _xor_fixture(seed=5)
    cfg = StudentConfig(
        kind="gboost", n_estimators=60, gb_max_depth=2, learning_rate=0.2,
        min_samples_leaf=5,
    )
    model = fit_student(X, y, w, cfg)

    used = ensemble_features(model, _FEATURES)
    # XOR forces both signal features to appear somewhere across the stages.
    assert {"alpha", "beta"} <= set(used), f"expected both signals, got {used}"
    # Every reported feature is real and none repeats (it is a union).
    assert set(used) <= set(_FEATURES)
    assert len(used) == len(set(used))


def test_ensemble_features_matches_manual_union() -> None:
    """The reported set equals the manual union of each stage tree's splits."""
    X, y, w = _xor_fixture(seed=6)
    cfg = StudentConfig(
        kind="gboost", n_estimators=30, gb_max_depth=2, learning_rate=0.2,
        min_samples_leaf=5,
    )
    model = fit_student(X, y, w, cfg)

    manual: set[str] = set()
    for stage in range(model.estimators_.shape[0]):
        inner = model.estimators_[stage, 0].tree_
        for node in range(inner.node_count):
            feat = int(inner.feature[node])
            if feat >= 0:
                manual.add(_FEATURES[feat])
    assert set(ensemble_features(model, _FEATURES)) == manual


def test_ensemble_to_rules_renders_stages_and_is_ascii() -> None:
    """The rules expose init/learning_rate, per-stage trees, and stay ASCII."""
    X, y, w = _xor_fixture(seed=7)
    cfg = StudentConfig(
        kind="gboost", n_estimators=12, gb_max_depth=2, learning_rate=0.1,
        min_samples_leaf=5,
    )
    model = fit_student(X, y, w, cfg)

    rules = ensemble_to_rules(model, _FEATURES)
    assert "init" in rules
    assert "learning_rate" in rules
    assert "stage 0" in rules
    assert "stage 11" in rules  # all 12 stages rendered.
    assert "value=" in rules  # regressor leaves print a raw value, not a class.
    assert "if alpha <=" in rules or "if beta <=" in rules
    # ASCII-only (Windows cp1252 safety): no character above U+007F.
    assert rules.encode("ascii")


# --------------------------------------------------------------------------- #
# student_features.                                                            #
# --------------------------------------------------------------------------- #


def test_student_features_returns_used_only() -> None:
    """Only the splitting feature(s) come back; the separable toy uses 'alpha'."""
    X, y, w = _separable_fixture()
    tree = fit_student(X, y, w, StudentConfig(max_depth=1, min_samples_leaf=5))

    used = student_features(tree, _FEATURES)
    assert used == ["alpha"], f"expected ['alpha'], got {used}"
    # Every reported feature is a real column, and none repeats.
    assert set(used) <= set(_FEATURES)
    assert len(used) == len(set(used))


def test_student_features_empty_for_single_leaf() -> None:
    """A single-class fixture yields a leaf-only tree that uses no feature."""
    n = 60
    X = pd.DataFrame(
        {name: np.random.default_rng(1).standard_normal(n) for name in _FEATURES}
    )
    y = np.zeros(n, dtype=np.int64)  # all negative -> root is already pure.
    tree = fit_student(X, y, None, StudentConfig(max_depth=3, min_samples_leaf=5))

    assert tree.get_n_leaves() == 1
    assert student_features(tree, _FEATURES) == []


# --------------------------------------------------------------------------- #
# tree_to_rules.                                                               #
# --------------------------------------------------------------------------- #


def test_tree_to_rules_renders_split_and_is_ascii() -> None:
    """The rules name the split feature, show a threshold, and stay ASCII."""
    X, y, w = _separable_fixture()
    tree = fit_student(X, y, w, StudentConfig(max_depth=1, min_samples_leaf=5))

    rules = tree_to_rules(tree, _FEATURES)
    assert "if alpha <=" in rules
    assert "else:" in rules
    assert "class=" in rules
    # ASCII-only (Windows cp1252 safety): no character above U+007F.
    assert rules.encode("ascii")
    # The unused features must not appear as a split.
    assert "if beta" not in rules
    assert "if gamma" not in rules


def test_tree_to_rules_leaf_only_tree() -> None:
    """A leaf-only tree renders a single class line, no ``if``."""
    n = 60
    X = pd.DataFrame(
        {name: np.random.default_rng(2).standard_normal(n) for name in _FEATURES}
    )
    y = np.zeros(n, dtype=np.int64)
    tree = fit_student(X, y, None, StudentConfig(max_depth=2, min_samples_leaf=5))

    rules = tree_to_rules(tree, _FEATURES)
    assert "class=0" in rules
    assert "if " not in rules


# --------------------------------------------------------------------------- #
# distill_from_teacher.                                                        #
# --------------------------------------------------------------------------- #


def test_distill_from_teacher_mimics_soft_calls() -> None:
    """A shallow tree reproduces a teacher's thresholded soft predictions."""
    X, y, _ = _separable_fixture(seed=7)
    # A "teacher" that is essentially right: high proba where alpha > 0.
    teacher_proba = np.clip(0.5 + 0.4 * np.sign(X["alpha"].to_numpy()), 0.0, 1.0)

    tree = distill_from_teacher(
        X, teacher_proba, StudentConfig(max_depth=2, min_samples_leaf=5)
    )
    pseudo = (teacher_proba >= 0.5).astype(np.int64)
    agreement = float((tree.predict(X) == pseudo).mean())
    assert agreement > 0.95, f"student mimicked teacher only {agreement:.3f}"


def test_distill_from_teacher_rejects_single_class() -> None:
    """A teacher whose calls collapse to one class cannot be distilled."""
    X, _, _ = _separable_fixture(seed=9)
    all_low = np.full(X.shape[0], 0.1, dtype=np.float64)  # never clears 0.5.

    with pytest.raises(ValueError):
        distill_from_teacher(X, all_low, StudentConfig(min_samples_leaf=5))
    with pytest.raises(ValueError):  # length mismatch.
        distill_from_teacher(X, all_low[:-1], StudentConfig(min_samples_leaf=5))
