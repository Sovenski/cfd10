"""Distil the turn detector into a SHALLOW, Pine-exportable decision tree.

The GBDT teacher (:mod:`cfd10.teacher_module.baseline`) is the current-best turn
detector, but a boosted ensemble of hundreds of trees cannot be hand-transcribed
into a TradingView Pine script. This module produces the *deployable* artefact: a
single, shallow :class:`~sklearn.tree.DecisionTreeClassifier` per side (HIGH /
LOW) whose every split is a plain ``feature <= threshold`` test — a ruleset a
human (or a code generator) can read straight off and emit as nested Pine
``if`` blocks.

Two fitting paths share the same shallow-tree backend:

* :func:`fit_student` — fit directly to the oracle's *hard* labels
  (``y in {0, 1}``) with the oracle score folded into the sample weight. This IS
  the student that ships.
* :func:`distill_from_teacher` — fit to the teacher's *soft* probabilities
  (thresholded into pseudo-labels, teacher confidence as weight). Used only to
  *measure fidelity*: how well a shallow tree can mimic the GBDT's calls.

Capacity is capped hard (``max_depth``, ``max_leaves``, ``min_samples_leaf``) so
the exported rule list stays short, and ``class_weight='balanced'`` keeps the
sparse positive turn class from being washed out by the ~99.8% negatives.

Two reporting helpers turn a fitted tree into review-ready text:
:func:`tree_to_rules` renders the human-readable nested rules, and
:func:`student_features` returns the handful of features the tree actually splits
on — the critical input to the downstream Pine export (only those features need
to be computed on-chart).

Public API
----------
:class:`StudentConfig` (frozen) — the shallow-tree capacity knobs.
:func:`fit_student` / :func:`distill_from_teacher` — the two fitting paths.
:func:`tree_to_rules` / :func:`student_features` — tree introspection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier, _tree

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "StudentConfig",
    "fit_student",
    "distill_from_teacher",
    "tree_to_rules",
    "student_features",
]


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StudentConfig:
    """Immutable shallow-tree capacity configuration.

    The defaults intentionally keep the tree small enough to transcribe into a
    Pine script by hand: at most ``max_leaves`` leaves means at most
    ``max_leaves`` rule paths, each a conjunction of at most ``max_depth``
    ``feature <= threshold`` tests.

    Attributes:
        max_depth: Maximum tree depth (the longest rule conjunction).
        max_leaves: Maximum number of leaves, i.e. the number of distinct rule
            paths (mapped to ``max_leaf_nodes``).
        min_samples_leaf: Minimum samples required at a leaf; a regulariser that
            stops the tree carving a rule for a handful of bars.
        seed: RNG seed for the tree's deterministic tie-breaking
            (``random_state``).
    """

    max_depth: int = 4
    max_leaves: int = 16
    min_samples_leaf: int = 30
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the capacity knobs.

        Raises:
            ValueError: If any capacity knob is out of its allowed domain.
        """
        if self.max_depth < 1:
            raise ValueError(
                f"StudentConfig: max_depth must be >= 1, got {self.max_depth}"
            )
        if self.max_leaves < 2:
            raise ValueError(
                f"StudentConfig: max_leaves must be >= 2, got {self.max_leaves}"
            )
        if self.min_samples_leaf < 1:
            raise ValueError(
                "StudentConfig: min_samples_leaf must be >= 1, got "
                f"{self.min_samples_leaf}"
            )

    def tree_kwargs(self) -> dict[str, object]:
        """Return the :class:`~sklearn.tree.DecisionTreeClassifier` kwargs.

        Returns:
            A kwargs dict pinning depth / leaf caps, the balanced class weight,
            and the seed. ``class_weight='balanced'`` reweights inversely to
            class frequency so the sparse positive turn class drives splits.
        """
        return {
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaves,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": "balanced",
            "random_state": self.seed,
        }


# --------------------------------------------------------------------------- #
# Input validation.                                                           #
# --------------------------------------------------------------------------- #


def _validate_fit_inputs(
    n_rows: int,
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64] | None,
) -> None:
    """Validate label / weight shapes against the feature row count.

    Args:
        n_rows: Number of feature rows.
        y: Binary label array.
        sample_weight: Optional per-row weight array.

    Raises:
        ValueError: If lengths disagree, ``y`` is empty, or ``y`` is not binary.
    """
    if y.shape[0] != n_rows:
        raise ValueError(f"fit_student: len(y)={y.shape[0]} != len(X)={n_rows}")
    if y.size == 0:
        raise ValueError("fit_student: empty label array")
    if sample_weight is not None and sample_weight.shape[0] != n_rows:
        raise ValueError(
            "fit_student: len(sample_weight)="
            f"{sample_weight.shape[0]} != len(X)={n_rows}"
        )
    uniq = set(np.unique(y).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"fit_student: y must be binary 0/1, got values {sorted(uniq)}")


# --------------------------------------------------------------------------- #
# Fitting.                                                                     #
# --------------------------------------------------------------------------- #


def fit_student(
    X,  # type: ignore[no-untyped-def]  # pandas.DataFrame; kept loose to avoid hard import
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64] | None,
    cfg: StudentConfig,
) -> DecisionTreeClassifier:
    """Fit the shallow student tree to the oracle's hard labels.

    This IS the deployable detector: a single shallow decision tree (capacity
    capped by ``cfg``, ``class_weight='balanced'``) whose rule paths are the
    exportable signal. The oracle score is expected to be folded into
    ``sample_weight`` by the caller (``w = 1 + score`` on turns, ``1`` otherwise),
    so stronger / larger-scale turns pull harder on the splits.

    Args:
        X: Dense feature matrix (``pandas.DataFrame``); column order defines the
            feature index used by the exported rules.
        y: Binary label array (``1`` = oracle turn) aligned to ``X``.
        sample_weight: Optional per-row weight aligned to ``X`` (the oracle
            score-weight); ``None`` weights every row equally.
        cfg: Shallow-tree capacity configuration.

    Returns:
        The fitted :class:`~sklearn.tree.DecisionTreeClassifier`.

    Raises:
        ValueError: If label / weight lengths disagree or ``y`` is not binary.
    """
    y = np.ascontiguousarray(y, dtype=np.int64)
    weight = (
        np.ascontiguousarray(sample_weight, dtype=np.float64)
        if sample_weight is not None
        else None
    )
    _validate_fit_inputs(int(X.shape[0]), y, weight)

    tree = DecisionTreeClassifier(**cfg.tree_kwargs())
    tree.fit(X, y, sample_weight=weight)

    logger.info(
        "fit_student: depth=%d leaves=%d (cap d<=%d, leaves<=%d), pos_rate=%.4f",
        int(tree.get_depth()),
        int(tree.get_n_leaves()),
        cfg.max_depth,
        cfg.max_leaves,
        float(y.mean()) if y.size else 0.0,
    )
    return tree


def distill_from_teacher(
    X,  # type: ignore[no-untyped-def]  # pandas.DataFrame
    teacher_proba: NDArray[np.float64],
    cfg: StudentConfig,
    threshold: float = 0.5,
) -> DecisionTreeClassifier:
    """Fit a shallow tree to the *teacher's* soft predictions (fidelity path).

    The teacher's class-1 probabilities are thresholded into pseudo-labels and
    its confidence ``|proba - 0.5|`` becomes the sample weight, so the tree
    focuses on reproducing the calls the teacher makes confidently. This path is
    for *fidelity reporting* — how faithfully a shallow tree can mimic the GBDT —
    not for shipping; the deployable student is :func:`fit_student`.

    Args:
        X: Dense feature matrix (``pandas.DataFrame``) the teacher scored.
        teacher_proba: Teacher class-1 probability per row, aligned to ``X``.
        cfg: Shallow-tree capacity configuration.
        threshold: Probability cut mapping ``teacher_proba`` to a pseudo-label.

    Returns:
        The fitted :class:`~sklearn.tree.DecisionTreeClassifier` mimicking the
        teacher.

    Raises:
        ValueError: If ``teacher_proba`` length disagrees with ``X`` or the
            pseudo-labels collapse to a single class (nothing to distil).
    """
    proba = np.ascontiguousarray(teacher_proba, dtype=np.float64)
    if proba.shape[0] != int(X.shape[0]):
        raise ValueError(
            f"distill_from_teacher: len(teacher_proba)={proba.shape[0]} "
            f"!= len(X)={int(X.shape[0])}"
        )

    pseudo = (proba >= threshold).astype(np.int64)
    if np.unique(pseudo).size < 2:
        raise ValueError(
            "distill_from_teacher: teacher pseudo-labels collapse to one class at "
            f"threshold={threshold}; cannot distil a two-class tree"
        )

    # Teacher confidence as the soft-target weight: a bar the teacher is sure
    # about (proba near 0 or 1) matters more than a near-coin-flip bar.
    confidence = np.abs(proba - 0.5)
    tree = DecisionTreeClassifier(**cfg.tree_kwargs())
    tree.fit(X, pseudo, sample_weight=confidence)

    logger.info(
        "distill_from_teacher: depth=%d leaves=%d, pseudo_pos_rate=%.4f (thr=%.3f)",
        int(tree.get_depth()),
        int(tree.get_n_leaves()),
        float(pseudo.mean()),
        threshold,
    )
    return tree


# --------------------------------------------------------------------------- #
# Tree introspection.                                                          #
# --------------------------------------------------------------------------- #


def student_features(
    tree: DecisionTreeClassifier, feature_names: list[str]
) -> list[str]:
    """Return the features the tree actually splits on, in first-use order.

    A capped tree typically uses far fewer than all available features; only
    these need to be computed for the Pine export. The order follows the node
    array (root first), so the most "important" structural splits come first.

    Args:
        tree: A fitted decision tree.
        feature_names: Column names aligned to the matrix the tree was fit on.

    Returns:
        The used feature names, de-duplicated and ordered by first appearance in
        the node array. Empty for a degenerate single-leaf tree.

    Raises:
        ValueError: If a split references a feature index outside
            ``feature_names``.
    """
    inner = tree.tree_
    used: list[str] = []
    seen: set[int] = set()
    for node in range(inner.node_count):
        feat = int(inner.feature[node])
        if feat == _tree.TREE_UNDEFINED:  # leaf node.
            continue
        if feat < 0 or feat >= len(feature_names):
            raise ValueError(
                f"student_features: split feature index {feat} out of range for "
                f"{len(feature_names)} feature names"
            )
        if feat not in seen:
            seen.add(feat)
            used.append(feature_names[feat])
    return used


def _class_label_index(tree: DecisionTreeClassifier) -> int:
    """Return the column index of the positive class (``1``) in ``tree.classes_``.

    Args:
        tree: A fitted decision tree.

    Returns:
        The index of class ``1`` in ``tree.classes_``, or ``-1`` if the tree saw
        only the negative class (a degenerate single-class fit).
    """
    classes = list(tree.classes_)
    return classes.index(1) if 1 in classes else -1


def tree_to_rules(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    decimals: int = 4,
) -> str:
    """Render a fitted tree as human-readable nested ``if`` rules (ASCII only).

    Each internal node becomes an ``if <feature> <= <threshold>:`` block; the two
    children are indented beneath the true / false branches. Leaves print the
    predicted class, the positive-class probability, and the (weighted) sample
    count, so a reviewer sees both the decision and its support.

    The output is plain ASCII (safe for Windows ``cp1252`` files / consoles) and
    mirrors the structure a Pine ``if/else`` transcription would take.

    Args:
        tree: A fitted decision tree.
        feature_names: Column names aligned to the matrix the tree was fit on.
        decimals: Threshold / probability rounding for readability.

    Returns:
        A multi-line string of the nested rules (no trailing newline).

    Raises:
        ValueError: If a split references an out-of-range feature index.
    """
    inner = tree.tree_
    pos_col = _class_label_index(tree)
    lines: list[str] = []

    def _leaf_text(node: int) -> str:
        """Format a leaf node's prediction, positive probability and support."""
        counts = inner.value[node][0]
        total = float(counts.sum())
        # ``value`` holds (possibly class-weighted) per-class mass; normalise it
        # to a probability so the leaf reads as a calibrated-ish confidence.
        pos_mass = float(counts[pos_col]) if pos_col >= 0 else 0.0
        prob = pos_mass / total if total > 0.0 else 0.0
        predict = 1 if prob >= 0.5 else 0
        n_samples = int(inner.n_node_samples[node])
        return (
            f"-> class={predict} (p={round(prob, decimals)}, "
            f"n={n_samples}, w={round(total, decimals)})"
        )

    def _recurse(node: int, depth: int) -> None:
        """Depth-first render of ``node`` and its subtree."""
        indent = "    " * depth
        feat = int(inner.feature[node])
        if feat == _tree.TREE_UNDEFINED:
            lines.append(f"{indent}{_leaf_text(node)}")
            return
        if feat < 0 or feat >= len(feature_names):
            raise ValueError(
                f"tree_to_rules: split feature index {feat} out of range for "
                f"{len(feature_names)} feature names"
            )
        name = feature_names[feat]
        thr = round(float(inner.threshold[node]), decimals)
        left = int(inner.children_left[node])
        right = int(inner.children_right[node])
        # scikit-learn routes ``feature <= threshold`` to the LEFT child.
        lines.append(f"{indent}if {name} <= {thr}:")
        _recurse(left, depth + 1)
        lines.append(f"{indent}else:  # {name} > {thr}")
        _recurse(right, depth + 1)

    _recurse(0, 0)
    return "\n".join(lines)
