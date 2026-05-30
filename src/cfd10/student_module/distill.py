"""Distil the turn detector into a SMALL, Pine-exportable student model.

The GBDT teacher (:mod:`cfd10.teacher_module.baseline`) is the current-best turn
detector, but a boosted ensemble of hundreds of trees cannot be hand-transcribed
into a TradingView Pine script. This module produces the *deployable* artefact: a
single, small model per side (HIGH / LOW) whose every split is a plain
``feature <= threshold`` test — a ruleset a human (or a code generator) can read
straight off and emit as nested Pine ``if`` blocks.

Two model *kinds* share one config and fitter:

* ``kind="tree"`` — a single :class:`~sklearn.tree.DecisionTreeClassifier`
  (``max_depth`` up to 8). The smallest, most transparent student; its rule paths
  transcribe one-to-one into nested Pine ``if/else``.
* ``kind="gboost"`` — a *small* :class:`~sklearn.ensemble.GradientBoostingClassifier`
  (a handful of shallow regression trees plus an ``init_`` log-odds offset). Still
  Pine-portable: each estimator is a tiny tree and the score is
  ``init_ + learning_rate * sum(leaf_value)``, thresholded at 0.

Two fitting paths share the backend:

* :func:`fit_student` — fit directly to the oracle's *hard* labels
  (``y in {0, 1}``) with the oracle score folded into the sample weight. This IS
  the student that ships.
* :func:`distill_from_teacher` — fit a *tree* to the teacher's *soft* probabilities
  (thresholded into pseudo-labels, teacher confidence as weight). Used only to
  *measure fidelity*: how well a shallow tree can mimic the GBDT's calls.

Crucially the student relies on the **oracle sample weight only** — there is no
``class_weight='balanced'``. Balancing floods the leaves with the rare positive
class and dissolves the sparse, high-precision cuts the Pine export needs; the
caller instead tunes a positive boost into ``sample_weight``.

Reporting helpers turn a fitted model into review-ready text and feature lists:
:func:`tree_to_rules` renders a single tree's nested rules, :func:`student_features`
lists the features that tree splits on; :func:`ensemble_to_rules` and
:func:`ensemble_features` are the gradient-boosting counterparts (the latter
returns the union of features used across the ensemble's estimators) — the
critical input to the downstream Pine export (only those features need to be
computed on-chart).

Public API
----------
:class:`StudentConfig` (frozen) — kind + capacity knobs (tree depth, ensemble
size).
:func:`fit_student` / :func:`distill_from_teacher` — the two fitting paths.
:func:`tree_to_rules` / :func:`student_features` — single-tree introspection.
:func:`ensemble_to_rules` / :func:`ensemble_features` — gradient-boosting
introspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "StudentConfig",
    "fit_student",
    "distill_from_teacher",
    "tree_to_rules",
    "student_features",
    "ensemble_to_rules",
    "ensemble_features",
]

# Supported student kinds.
_KIND_TREE = "tree"
_KIND_GBOOST = "gboost"
_KINDS: frozenset[str] = frozenset({_KIND_TREE, _KIND_GBOOST})

# A fitted student is either a single classification tree or a small gradient
# boosting ensemble; both are introspectable down to per-node thresholds.
Student = DecisionTreeClassifier | GradientBoostingClassifier


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StudentConfig:
    """Immutable small-student capacity configuration (tree or gradient boost).

    The defaults keep the model small enough to transcribe into Pine by hand. For
    ``kind="tree"`` at most ``max_leaves`` leaves means at most ``max_leaves`` rule
    paths, each a conjunction of at most ``max_depth`` ``feature <= threshold``
    tests. For ``kind="gboost"`` the export is ``n_estimators`` tiny depth
    ``gb_max_depth`` regression trees summed with ``learning_rate`` onto the
    ``init_`` log-odds — still a short, bounded rule set.

    Attributes:
        kind: ``"tree"`` (single decision tree) or ``"gboost"`` (small gradient
            boosting ensemble).
        max_depth: Maximum *tree* depth (the longest rule conjunction); up to 8.
            Only used when ``kind == "tree"``.
        max_leaves: Maximum number of leaves for a single tree (mapped to
            ``max_leaf_nodes``). Only used when ``kind == "tree"``.
        min_samples_leaf: Minimum samples required at a leaf; a regulariser that
            stops the model carving a rule for a handful of bars (used by both
            kinds).
        n_estimators: Number of boosting stages (regression trees) when
            ``kind == "gboost"``.
        gb_max_depth: Per-estimator tree depth when ``kind == "gboost"`` (kept
            shallow so each stage stays Pine-small).
        learning_rate: Boosting shrinkage when ``kind == "gboost"``.
        seed: RNG seed for deterministic tie-breaking / subsampling
            (``random_state``).
    """

    kind: str = _KIND_TREE
    max_depth: int = 4
    max_leaves: int = 16
    min_samples_leaf: int = 30
    n_estimators: int = 40
    gb_max_depth: int = 3
    learning_rate: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the kind and capacity knobs.

        Raises:
            ValueError: If the kind is unknown or any capacity knob is out of its
                allowed domain (tree depth must be 1..8).
        """
        if self.kind not in _KINDS:
            raise ValueError(
                f"StudentConfig: kind must be one of {sorted(_KINDS)}, got {self.kind!r}"
            )
        if self.max_depth < 1:
            raise ValueError(
                f"StudentConfig: max_depth must be >= 1, got {self.max_depth}"
            )
        if self.max_depth > 8:
            raise ValueError(
                f"StudentConfig: max_depth must be <= 8 (Pine portability), "
                f"got {self.max_depth}"
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
        if self.n_estimators < 1:
            raise ValueError(
                f"StudentConfig: n_estimators must be >= 1, got {self.n_estimators}"
            )
        if self.gb_max_depth < 1:
            raise ValueError(
                f"StudentConfig: gb_max_depth must be >= 1, got {self.gb_max_depth}"
            )
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                "StudentConfig: learning_rate must be in (0, 1], got "
                f"{self.learning_rate}"
            )

    def tree_kwargs(self) -> dict[str, object]:
        """Return the :class:`~sklearn.tree.DecisionTreeClassifier` kwargs.

        No ``class_weight`` is set: the student deliberately relies on the oracle
        sample weight alone. ``class_weight='balanced'`` would reweight inversely
        to class frequency, flooding every leaf with the rare positive class and
        dissolving the sparse high-precision cuts the Pine export needs — so it is
        intentionally absent and the caller boosts positives via ``sample_weight``.

        Returns:
            A kwargs dict pinning depth / leaf caps, the min-leaf regulariser, and
            the seed.
        """
        return {
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaves,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.seed,
        }

    def gboost_kwargs(self) -> dict[str, object]:
        """Return the :class:`~sklearn.ensemble.GradientBoostingClassifier` kwargs.

        A deliberately small ensemble: ``n_estimators`` shallow (``gb_max_depth``)
        regression trees with ``learning_rate`` shrinkage, regularised by the same
        ``min_samples_leaf``. Like the tree path it carries no class balancing;
        the oracle sample weight does the reweighting.

        Returns:
            A kwargs dict for the gradient boosting classifier.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.gb_max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
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
) -> Student:
    """Fit the deployable student to the oracle's hard labels.

    This IS the deployable detector. Depending on ``cfg.kind`` it is either a
    single shallow decision tree (``kind="tree"``, ``max_depth`` up to 8) or a
    *small* gradient boosting ensemble (``kind="gboost"``). Neither uses
    ``class_weight='balanced'``: the oracle score is folded into ``sample_weight``
    by the caller (``w = pos_boost * (1 + score)`` on turns, ``1`` otherwise), so
    stronger / larger-scale turns pull harder on the splits while the rare
    positive class is not artificially flooded.

    Both returned models expose the structure the Pine export needs: a tree via
    ``tree_.feature`` / ``tree_.threshold`` / children, an ensemble via
    ``estimators_`` (per-stage regression trees), ``init_`` and ``learning_rate``.

    Args:
        X: Dense feature matrix (``pandas.DataFrame``); column order defines the
            feature index used by the exported rules.
        y: Binary label array (``1`` = oracle turn) aligned to ``X``.
        sample_weight: Optional per-row weight aligned to ``X`` (the oracle
            score-weight); ``None`` weights every row equally.
        cfg: Student capacity configuration (kind + knobs).

    Returns:
        The fitted student: a :class:`~sklearn.tree.DecisionTreeClassifier` for
        ``kind="tree"`` or a :class:`~sklearn.ensemble.GradientBoostingClassifier`
        for ``kind="gboost"``.

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

    if cfg.kind == _KIND_GBOOST:
        model: Student = GradientBoostingClassifier(**cfg.gboost_kwargs())
        model.fit(X, y, sample_weight=weight)
        logger.info(
            "fit_student[gboost]: n_estimators=%d depth=%d lr=%.3f, pos_rate=%.4f",
            cfg.n_estimators,
            cfg.gb_max_depth,
            cfg.learning_rate,
            float(y.mean()) if y.size else 0.0,
        )
        return model

    tree = DecisionTreeClassifier(**cfg.tree_kwargs())
    tree.fit(X, y, sample_weight=weight)
    logger.info(
        "fit_student[tree]: depth=%d leaves=%d (cap d<=%d, leaves<=%d), pos_rate=%.4f",
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
    """Fit a shallow *tree* to the *teacher's* soft predictions (fidelity path).

    The teacher's class-1 probabilities are thresholded into pseudo-labels and
    its confidence ``|proba - 0.5|`` becomes the sample weight, so the tree
    focuses on reproducing the calls the teacher makes confidently. This path is
    for *fidelity reporting* — how faithfully a shallow tree can mimic the GBDT —
    not for shipping; the deployable student is :func:`fit_student`. It always
    fits a single tree (``cfg.tree_kwargs()``) regardless of ``cfg.kind``.

    Args:
        X: Dense feature matrix (``pandas.DataFrame``) the teacher scored.
        teacher_proba: Teacher class-1 probability per row, aligned to ``X``.
        cfg: Student capacity configuration (the tree caps are used).
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
# Single-tree introspection.                                                   #
# --------------------------------------------------------------------------- #


def _used_features_of_tree(
    inner: _tree.Tree,
    feature_names: list[str],
    seen: set[int],
    out: list[str],
) -> None:
    """Append a single tree's split features (first-use order) into ``out``.

    Mutates ``seen`` / ``out`` so the same buffers can accumulate the union over
    an ensemble's estimators.

    Args:
        inner: The fitted tree's ``tree_`` structure.
        feature_names: Column names aligned to the fit matrix.
        seen: Set of already-recorded feature indices (mutated).
        out: Accumulating list of feature names in first-use order (mutated).

    Raises:
        ValueError: If a split references a feature index outside
            ``feature_names``.
    """
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
            out.append(feature_names[feat])


def student_features(
    tree: DecisionTreeClassifier, feature_names: list[str]
) -> list[str]:
    """Return the features a single tree splits on, in first-use order.

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
    used: list[str] = []
    seen: set[int] = set()
    _used_features_of_tree(tree.tree_, feature_names, seen, used)
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


# --------------------------------------------------------------------------- #
# Gradient-boosting introspection.                                             #
# --------------------------------------------------------------------------- #


def _iter_stage_trees(model: GradientBoostingClassifier) -> list[DecisionTreeRegressor]:
    """Return the per-stage regression trees of a binary gradient booster.

    For binary classification scikit-learn stores one regression tree per stage
    in ``estimators_[stage, 0]``. This flattens that ``(n_stages, 1)`` array into
    a simple list, in boosting order.

    Args:
        model: A fitted :class:`~sklearn.ensemble.GradientBoostingClassifier`.

    Returns:
        The stage regression trees in boosting order.

    Raises:
        ValueError: If the ensemble is not a binary (single-column) booster.
    """
    estimators = model.estimators_
    if estimators.ndim != 2 or estimators.shape[1] != 1:
        raise ValueError(
            "ensemble introspection: expected a binary GradientBoostingClassifier "
            f"with one tree per stage, got estimators shape {estimators.shape}"
        )
    return [estimators[stage, 0] for stage in range(estimators.shape[0])]


def ensemble_features(
    model: GradientBoostingClassifier, feature_names: list[str]
) -> list[str]:
    """Return the union of features used across a gradient booster's estimators.

    Walks every stage regression tree (boosting order) and records each split
    feature the first time it appears anywhere in the ensemble. These are exactly
    the features the Pine export must recompute on-chart for a ``gboost`` student.

    Args:
        model: A fitted :class:`~sklearn.ensemble.GradientBoostingClassifier`.
        feature_names: Column names aligned to the matrix the model was fit on.

    Returns:
        The used feature names, de-duplicated and ordered by first appearance
        across the ensemble (stage-major, then node order). Empty if no stage tree
        ever splits (a degenerate constant ensemble).

    Raises:
        ValueError: If a split references a feature index outside
            ``feature_names`` or the ensemble is not binary.
    """
    used: list[str] = []
    seen: set[int] = set()
    for stage_tree in _iter_stage_trees(model):
        _used_features_of_tree(stage_tree.tree_, feature_names, seen, used)
    return used


def ensemble_to_rules(
    model: GradientBoostingClassifier,
    feature_names: list[str],
    decimals: int = 4,
) -> str:
    """Render a gradient booster as the additive score it computes (ASCII only).

    A binary gradient boosting score is ``raw = init_ + learning_rate * sum_stage
    leaf_value(stage)``, with a positive call when ``raw > 0`` (probability
    ``> 0.5``). This renders that contract for review: the ``init_`` log-odds
    offset, the shrinkage, then each stage's regression tree as nested
    ``if <feature> <= <threshold>:`` blocks whose leaves print the *raw additive
    value* (not a class) the stage contributes. The Pine export sums these leaf
    values, scales by ``learning_rate``, adds ``init_`` and tests ``> 0``.

    Args:
        model: A fitted :class:`~sklearn.ensemble.GradientBoostingClassifier`.
        feature_names: Column names aligned to the matrix the model was fit on.
        decimals: Threshold / value rounding for readability.

    Returns:
        A multi-line ASCII string: a header (init_, learning_rate, stage count,
        decision rule) followed by every stage's tree.

    Raises:
        ValueError: If a split references an out-of-range feature index or the
            ensemble is not binary.
    """
    stage_trees = _iter_stage_trees(model)
    # ``init_`` is a fitted prior estimator; the booster's starting raw score is
    # its prior log-odds. Recover it from the prior so we do not depend on a
    # private attribute layout.
    init_raw = _gboost_init_raw(model)
    lr = float(model.learning_rate)

    header = (
        f"# gradient-boosting student: raw = init + learning_rate * sum(stage leaf)\n"
        f"# init (log-odds) = {round(init_raw, decimals)}\n"
        f"# learning_rate   = {round(lr, decimals)}\n"
        f"# stages          = {len(stage_trees)}\n"
        f"# decision        = positive (class 1) iff raw > 0\n"
        f"# (each stage leaf below is the RAW value it adds, before learning_rate)"
    )

    blocks: list[str] = [header]
    for stage, stage_tree in enumerate(stage_trees):
        blocks.append(f"\n# --- stage {stage} ---")
        blocks.append(_regressor_to_rules(stage_tree, feature_names, decimals))
    return "\n".join(blocks)


def _gboost_init_raw(model: GradientBoostingClassifier) -> float:
    """Return the booster's initial raw (log-odds) score before any stage.

    scikit-learn seeds the additive model with ``init_`` (a prior). For the
    log-loss binary objective the starting raw margin is the prior log-odds
    ``log(p / (1 - p))``. We read it back from the fitted ``init_`` estimator's
    class-1 probability so the value is correct regardless of the private
    attribute layout, clipping to keep the log finite.

    Args:
        model: A fitted gradient boosting classifier.

    Returns:
        The initial raw score (log-odds) as a plain ``float`` (``0.0`` if the
        prior cannot be read, i.e. a balanced 0.5 start).
    """
    init = getattr(model, "init_", None)
    if init is None or not hasattr(init, "class_prior_"):
        return 0.0
    prior = np.asarray(init.class_prior_, dtype=np.float64)
    if prior.size < 2:
        return 0.0
    p1 = float(np.clip(prior[-1], 1e-12, 1.0 - 1e-12))
    return float(np.log(p1 / (1.0 - p1)))


def _regressor_to_rules(
    tree: DecisionTreeRegressor,
    feature_names: list[str],
    decimals: int,
) -> str:
    """Render one stage regression tree as nested ``if`` rules with raw leaf values.

    Mirrors :func:`tree_to_rules` but for a *regressor*: each leaf prints the raw
    real-valued contribution (the value the booster sums, before
    ``learning_rate``) rather than a class.

    Args:
        tree: A fitted stage :class:`~sklearn.tree.DecisionTreeRegressor`.
        feature_names: Column names aligned to the fit matrix.
        decimals: Threshold / value rounding.

    Returns:
        A multi-line ASCII string for this stage's tree (no trailing newline).

    Raises:
        ValueError: If a split references an out-of-range feature index.
    """
    inner = tree.tree_
    lines: list[str] = []

    def _leaf_text(node: int) -> str:
        value = float(inner.value[node][0][0])
        n_samples = int(inner.n_node_samples[node])
        return f"-> value={round(value, decimals)} (n={n_samples})"

    def _recurse(node: int, depth: int) -> None:
        indent = "    " * depth
        feat = int(inner.feature[node])
        if feat == _tree.TREE_UNDEFINED:
            lines.append(f"{indent}{_leaf_text(node)}")
            return
        if feat < 0 or feat >= len(feature_names):
            raise ValueError(
                f"ensemble_to_rules: split feature index {feat} out of range for "
                f"{len(feature_names)} feature names"
            )
        name = feature_names[feat]
        thr = round(float(inner.threshold[node]), decimals)
        left = int(inner.children_left[node])
        right = int(inner.children_right[node])
        lines.append(f"{indent}if {name} <= {thr}:")
        _recurse(left, depth + 1)
        lines.append(f"{indent}else:  # {name} > {thr}")
        _recurse(right, depth + 1)

    _recurse(0, 0)
    return "\n".join(lines)
