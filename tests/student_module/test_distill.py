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

from cfd10.student_module import (
    StudentConfig,
    distill_from_teacher,
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


# --------------------------------------------------------------------------- #
# Config.                                                                      #
# --------------------------------------------------------------------------- #


def test_config_defaults_and_validation() -> None:
    """`StudentConfig` carries the brief's defaults and rejects bad capacity."""
    cfg = StudentConfig()
    assert cfg.max_depth == 4
    assert cfg.max_leaves == 16
    assert cfg.min_samples_leaf == 30

    with pytest.raises(ValueError):
        StudentConfig(max_depth=0)
    with pytest.raises(ValueError):
        StudentConfig(max_leaves=1)
    with pytest.raises(ValueError):
        StudentConfig(min_samples_leaf=0)


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
