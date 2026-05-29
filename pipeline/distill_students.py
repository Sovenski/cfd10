"""Distil the current-best detector into SHALLOW, Pine-exportable student trees.

Run with::

    uv run python pipeline/distill_students.py

The GBDT teacher (:mod:`pipeline.fit_pooled`) is the yardstick, but an ensemble
of hundreds of boosted trees cannot be hand-transcribed into a TradingView Pine
script. This pipeline produces the *deployable* artefact: one shallow
:class:`~sklearn.tree.DecisionTreeClassifier` per side (HIGH / LOW) whose every
split is a plain ``feature <= threshold`` test.

For each side the chain is:

1. Build the multi-asset pooled corpus with the famous-lows-validated STRUCTURAL
   oracle (nest ``(20, 50, 100, 200)``, linear weight curve, 10% drawdown,
   horizon 60, tau 0.60/0.30) and the default feature bank — the *same* labels /
   features the GBDT baseline uses, so the comparison is apples-to-apples.
2. Derive the side's binary label (tier ``in {strong, regular}`` -> ``y = 1``)
   and the oracle score-weight (``w = 1 + score`` on turns, ``1`` otherwise).
3. Fit the FINAL deployable student on ALL pooled rows (this is the tree that
   ships) and record its depth, leaf count and — critically — the handful of
   FEATURES it actually splits on (these drive the Pine export).
4. Report a quick OOS event-F1 via purged + embargoed walk-forward CV, fitting a
   *fresh* student per fold and tuning its decision threshold on the train fold
   only (mirroring the GBDT harness), so we know the shallow tree is not worse
   than the GBDT it distils.
5. Report fidelity: the agreement between the student's calls and the GBDT's
   calls on the last fold's held-out test block (how faithfully the shallow tree
   mimics the teacher).

Artefacts per side, under ``outputs/``:
    ``student_{side}.pkl``  — the pickled fitted tree (deployable);
    ``student_{side}.json`` — its nodes / thresholds (export-friendly);
    ``student_{side}_rules.txt`` — the human-readable nested rules.

This is an oracle-vs-detector OOS evaluation only — no Pine-signal comparison
and no PnL. Import-safe (nothing runs on import).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier, _tree

from cfd10.cv_module import Fold, purged_walk_forward
from cfd10.data_module import PooledDataset, build_pooled_dataset
from cfd10.eval_module import event_prf, match_events
from cfd10.feature_module import FeatureConfig
from cfd10.label_module import OracleConfig
from cfd10.student_module import (
    StudentConfig,
    fit_student,
    student_features,
    tree_to_rules,
)
from cfd10.teacher_module import GBDTConfig
from cfd10.teacher_module.baseline import _fit_one_fold, _positive_indices, _tune_threshold
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DATA_ROOT: Path = _REPO_ROOT / "data" / "raw_v16"
_OUT_DIR: Path = _REPO_ROOT / "outputs"

_SPX: str = "SPX"
_SEED: int = 42
_TIMEFRAME: str = "1D"
_TOLERANCE: int = 3
_LABEL_HORIZON: int = 60
_EMBARGO: int = 5
_N_FOLDS: int = 6
_N_THRESHOLD_GRID: int = 50

_POSITIVE_TIERS: frozenset[str] = frozenset({"strong", "regular"})

# The famous-lows-validated STRUCTURAL oracle (identical to the GBDT baseline).
_ORACLE_CFG = OracleConfig(
    scale_nest=(20, 50, 100, 200),
    weight_curve="linear",
    drawdown_pct=0.10,
    horizon=60,
    tau_strong=0.60,
    tau_regular=0.30,
)

# Shallow, Pine-exportable student capacity (the brief's defaults).
_STUDENT_CFG = StudentConfig(max_depth=4, max_leaves=16, min_samples_leaf=30, seed=_SEED)

# The GBDT teacher config (mirrors pipeline.fit_pooled), used both for the
# fidelity comparison and so the student OOS shares the teacher's folds.
_GBDT_CFG = GBDTConfig(
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=400,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    tolerance=_TOLERANCE,
    label_horizon=_LABEL_HORIZON,
    embargo=_EMBARGO,
    n_folds=_N_FOLDS,
    n_threshold_grid=_N_THRESHOLD_GRID,
    seed=_SEED,
)


# --------------------------------------------------------------------------- #
# Result containers.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OosSlice:
    """Out-of-sample event PRF over the pooled test blocks.

    Attributes:
        precision / recall / f1: Event metrics in ``[0, 1]``.
        tp / fp / fn: Confusion counts.
        n_true / n_pred: True and predicted OOS event counts.
    """

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_true: int
    n_pred: int


@dataclass(frozen=True)
class SideReport:
    """One side's distilled student: capacity, features, OOS, and fidelity.

    Attributes:
        side: ``"high"`` (top turns) or ``"low"`` (bottom turns).
        n_pos: Pooled positive (strong/regular) count.
        positive_rate: Pooled positive fraction.
        depth: Final (all-rows) student tree depth.
        leaves: Final student leaf count.
        features: Feature names the final student splits on (drives Pine export).
        student_oos: Pooled OOS event slice for the per-fold student.
        gbdt_oos: Pooled OOS event slice for the GBDT teacher (same data / folds).
        fidelity: Student-vs-GBDT call agreement on the last fold's test block.
        rules_path / pkl_path / json_path: Saved artefact paths.
    """

    side: str
    n_pos: int
    positive_rate: float
    depth: int
    leaves: int
    features: list[str]
    student_oos: OosSlice
    gbdt_oos: OosSlice
    fidelity: float
    rules_path: Path
    pkl_path: Path
    json_path: Path


# --------------------------------------------------------------------------- #
# Side label / weight derivation (the sample-weight convention).              #
# --------------------------------------------------------------------------- #


def _side_columns(side: str) -> tuple[str, str]:
    """Return the ``(tier_col, weight_col)`` oracle columns for ``side``."""
    if side == "high":
        return "top_tier", "top_weight"
    if side == "low":
        return "bottom_tier", "bottom_weight"
    raise ValueError(f"distill_students: unknown side {side!r} (expected 'high'/'low')")


def _labels_and_weights(
    data: PooledDataset, side: str
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Derive the binary label and oracle score-weight for ``side``.

    Label: oracle tier ``in {strong, regular}`` -> ``y = 1``. Sample weight
    follows the project convention ``w = where(y == 1, 1 + oracle_score, 1)`` so
    the raw score (``0`` on non-turns) never zeroes the negative class and a
    big-scale pivot pulls harder than a small one.

    Args:
        data: The pooled corpus.
        side: ``"high"`` or ``"low"``.

    Returns:
        ``(y, sample_weight)`` aligned to the pooled rows.
    """
    tier_col, weight_col = _side_columns(side)
    tiers = data.labels[tier_col].to_numpy()
    y = np.isin(tiers, list(_POSITIVE_TIERS)).astype(np.int64)
    oracle_score = data.labels[weight_col].to_numpy(dtype=np.float64)
    sample_weight = np.where(y == 1, 1.0 + oracle_score, 1.0).astype(np.float64)
    return y, sample_weight


# --------------------------------------------------------------------------- #
# Per-fold student OOS (mirrors the GBDT harness: train-only threshold tuning).#
# --------------------------------------------------------------------------- #


def _student_proba(tree: DecisionTreeClassifier, X_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Positive-class probabilities from a fitted tree (``0`` if no positives seen)."""
    classes = list(tree.classes_)
    if 1 not in classes:
        return np.zeros(X_values.shape[0], dtype=np.float64)
    col = classes.index(1)
    return np.ascontiguousarray(tree.predict_proba(X_values)[:, col], dtype=np.float64)


def _student_fold_oos(
    X_values: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    fold_id: int,
    fold: Fold,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Fit a fresh student on the train block and flag OOS test bars.

    The decision threshold is tuned to maximise event-F1 on the *train* fold only
    (reusing the baseline's :func:`_tune_threshold`), then frozen for the held-out
    test block — identical leakage-safe protocol to the GBDT baseline.

    Args:
        X_values: Dense feature matrix as a ``float64`` array.
        y: Binary labels aligned to ``X_values``.
        sample_weight: Per-row oracle weights aligned to ``X_values``.
        fold_id: 0-based fold index (for logging).
        fold: ``(train_idx, test_idx)`` original-row indices.

    Returns:
        ``(test_pred_idx, test_true_idx)`` — predicted and true OOS bar indices.
    """
    train_idx, test_idx = fold
    tree = DecisionTreeClassifier(**_STUDENT_CFG.tree_kwargs())
    tree.fit(X_values[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])

    train_proba = _student_proba(tree, X_values[train_idx])
    test_proba = _student_proba(tree, X_values[test_idx])

    train_true_idx = np.ascontiguousarray(train_idx[y[train_idx] == 1], dtype=np.int64)
    test_true_idx = np.ascontiguousarray(test_idx[y[test_idx] == 1], dtype=np.int64)

    threshold, train_f1 = _tune_threshold(
        train_proba, train_idx, train_true_idx, _GBDT_CFG
    )
    test_pred_idx = _positive_indices(test_proba, test_idx, threshold)

    logger.info(
        "student fold %d: thr=%.4f train_f1=%.3f -> OOS pred=%d true=%d",
        fold_id,
        threshold,
        train_f1,
        test_pred_idx.size,
        test_true_idx.size,
    )
    return test_pred_idx, test_true_idx


def _pool_oos(
    pred_blocks: list[NDArray[np.int64]],
    true_blocks: list[NDArray[np.int64]],
) -> OosSlice:
    """Concatenate per-fold OOS indices and score one pooled event PRF.

    Test blocks are disjoint in the original index space, so a single global
    match over the concatenated indices is exact (no cross-fold double counting).

    Args:
        pred_blocks: Per-fold predicted OOS bar indices.
        true_blocks: Per-fold true OOS bar indices.

    Returns:
        The pooled :class:`OosSlice`.
    """
    all_pred = np.concatenate(pred_blocks) if pred_blocks else np.empty(0, np.int64)
    all_true = np.concatenate(true_blocks) if true_blocks else np.empty(0, np.int64)
    tp, fp, fn, _ = match_events(all_pred, all_true, _TOLERANCE)
    p, r, f1 = event_prf(all_pred, all_true, _TOLERANCE)
    return OosSlice(
        precision=p,
        recall=r,
        f1=f1,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        n_true=int(all_true.size),
        n_pred=int(all_pred.size),
    )


# --------------------------------------------------------------------------- #
# Fidelity: student-vs-GBDT call agreement on a held-out block.               #
# --------------------------------------------------------------------------- #


def _fidelity_last_fold(
    X_values: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    gbdt_cfg: GBDTConfig,
    fold: Fold,
) -> float:
    """Agreement between the student's and GBDT's per-bar calls on a test block.

    Both models are trained on the *same* train block (each tuning its own
    train-only threshold), then their binary flags over the held-out test rows
    are compared. Returns the fraction of test rows on which the two detectors
    agree — the shallow tree's fidelity to the teacher.

    Args:
        X_values: Dense feature matrix as a ``float64`` array.
        y: Binary labels aligned to ``X_values``.
        sample_weight: Per-row oracle weights aligned to ``X_values``.
        gbdt_cfg: The GBDT teacher config (already carries ``scale_pos_weight``).
        fold: The ``(train_idx, test_idx)`` block to compare on.

    Returns:
        Per-bar agreement in ``[0, 1]`` (``1.0`` if the test block is empty).
    """
    train_idx, test_idx = fold
    if test_idx.size == 0:
        return 1.0

    # GBDT teacher: reuse the baseline fold fitter to get its OOS predicted bars.
    _result, gbdt_pred_idx, _true = _fit_one_fold(
        X_values, y, sample_weight, 0, fold, gbdt_cfg
    )

    # Student: fit + tune threshold on the same train block.
    student_pred_idx, _student_true = _student_fold_oos(
        X_values, y, sample_weight, 0, fold
    )

    # Compare the two flag vectors over the test rows.
    gbdt_flag = np.isin(test_idx, gbdt_pred_idx)
    student_flag = np.isin(test_idx, student_pred_idx)
    return float((gbdt_flag == student_flag).mean())


# --------------------------------------------------------------------------- #
# Final (all-rows) student fit + artefact export.                             #
# --------------------------------------------------------------------------- #


def _tree_to_json(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict[str, object]:
    """Serialise a fitted tree's node array to a plain, export-friendly dict.

    Captures everything a Pine generator needs: per-node split feature (name +
    index), threshold, child links, and leaf positive-class probability. Leaf
    nodes carry ``feature: null`` and ``threshold: null``.

    Args:
        tree: A fitted decision tree.
        feature_names: Column names aligned to the matrix the tree was fit on.

    Returns:
        A JSON-serialisable dict with ``n_nodes``, ``max_depth``, ``features``
        (used, in first-use order), and the ``nodes`` list.
    """
    inner = tree.tree_
    classes = list(tree.classes_)
    pos_col = classes.index(1) if 1 in classes else -1

    nodes: list[dict[str, object]] = []
    for node in range(inner.node_count):
        feat = int(inner.feature[node])
        is_leaf = feat == _tree.TREE_UNDEFINED
        counts = inner.value[node][0]
        total = float(counts.sum())
        pos_mass = float(counts[pos_col]) if pos_col >= 0 else 0.0
        prob = pos_mass / total if total > 0.0 else 0.0
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
        "features": student_features(tree, feature_names),
        "nodes": nodes,
    }


def _save_artefacts(
    side: str,
    tree: DecisionTreeClassifier,
    feature_names: list[str],
) -> tuple[Path, Path, Path]:
    """Persist the deployable tree as pickle + node JSON + readable rules.

    Args:
        side: ``"high"`` or ``"low"`` (names the files).
        tree: The fitted final student.
        feature_names: Column names aligned to the matrix the tree was fit on.

    Returns:
        ``(pkl_path, json_path, rules_path)``.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = _OUT_DIR / f"student_{side}.pkl"
    json_path = _OUT_DIR / f"student_{side}.json"
    rules_path = _OUT_DIR / f"student_{side}_rules.txt"

    with pkl_path.open("wb") as fh:
        pickle.dump(tree, fh)
    json_path.write_text(
        json.dumps(_tree_to_json(tree, feature_names), indent=2), encoding="utf-8"
    )

    header = (
        f"# cfd10 student ruleset -- {side.upper()} "
        f"({'top' if side == 'high' else 'bottom'} turns)\n"
        f"# Shallow DecisionTreeClassifier distilled from the STRUCTURAL oracle.\n"
        f"# depth={tree.get_depth()} leaves={tree.get_n_leaves()} "
        f"features_used={student_features(tree, feature_names)}\n"
        f"# scikit-learn routes 'feature <= threshold' to the if-branch.\n\n"
    )
    rules_path.write_text(
        header + tree_to_rules(tree, feature_names) + "\n", encoding="utf-8"
    )
    return pkl_path, json_path, rules_path


# --------------------------------------------------------------------------- #
# Per-side orchestration.                                                      #
# --------------------------------------------------------------------------- #


def distil_side(
    data: PooledDataset,
    splits: list[Fold],
    X_values: NDArray[np.float64],
    side: str,
) -> SideReport:
    """Fit the final student, score per-fold OOS, and measure GBDT fidelity.

    Args:
        data: The pooled corpus.
        splits: Shared purged walk-forward folds over the pooled time axis.
        X_values: Dense pooled feature matrix as a ``float64`` array.
        side: ``"high"`` or ``"low"``.

    Returns:
        The assembled :class:`SideReport`.
    """
    from dataclasses import replace

    feature_names = list(data.X.columns)
    y, sample_weight = _labels_and_weights(data, side)
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    spw = float(n_neg / max(n_pos, 1))
    gbdt_cfg = replace(_GBDT_CFG, scale_pos_weight=spw)

    logger.info(
        "distil_side[%s]: %d positives / %d rows (%.3f%%); scale_pos_weight=%.1f",
        side,
        n_pos,
        y.shape[0],
        100.0 * n_pos / max(y.shape[0], 1),
        spw,
    )

    # 1) FINAL deployable student on ALL pooled rows.
    final_tree = fit_student(data.X, y, sample_weight, _STUDENT_CFG)
    features = student_features(final_tree, feature_names)
    pkl_path, json_path, rules_path = _save_artefacts(side, final_tree, feature_names)

    # 2) Per-fold OOS event-F1 (fresh student per fold, train-only thresholding).
    pred_blocks: list[NDArray[np.int64]] = []
    true_blocks: list[NDArray[np.int64]] = []
    for fold_id, fold in enumerate(splits):
        pred_idx, true_idx = _student_fold_oos(
            X_values, y, sample_weight, fold_id, fold
        )
        pred_blocks.append(pred_idx)
        true_blocks.append(true_idx)
    student_oos = _pool_oos(pred_blocks, true_blocks)

    # 3) GBDT teacher OOS on the same folds (for the not-worse-than gate).
    gbdt_pred: list[NDArray[np.int64]] = []
    gbdt_true: list[NDArray[np.int64]] = []
    for fold_id, fold in enumerate(splits):
        _r, p_idx, t_idx = _fit_one_fold(X_values, y, sample_weight, fold_id, fold, gbdt_cfg)
        gbdt_pred.append(p_idx)
        gbdt_true.append(t_idx)
    gbdt_oos = _pool_oos(gbdt_pred, gbdt_true)

    # 4) Fidelity on the last fold's test block.
    fidelity = _fidelity_last_fold(X_values, y, sample_weight, gbdt_cfg, splits[-1])

    logger.info(
        "distil_side[%s]: student OOS f1=%.4f | GBDT OOS f1=%.4f | fidelity=%.4f | "
        "features=%s",
        side,
        student_oos.f1,
        gbdt_oos.f1,
        fidelity,
        features,
    )

    return SideReport(
        side=side,
        n_pos=n_pos,
        positive_rate=float(y.mean()) if y.size else 0.0,
        depth=int(final_tree.get_depth()),
        leaves=int(final_tree.get_n_leaves()),
        features=features,
        student_oos=student_oos,
        gbdt_oos=gbdt_oos,
        fidelity=fidelity,
        rules_path=rules_path,
        pkl_path=pkl_path,
        json_path=json_path,
    )


# --------------------------------------------------------------------------- #
# Top-level run + reporting.                                                    #
# --------------------------------------------------------------------------- #


def run(
    seed: int = _SEED, max_assets: int | None = None
) -> tuple[PooledDataset, list[SideReport]]:
    """Build the pool, distil both sides, save artefacts, and return the reports.

    Args:
        seed: Global RNG seed for reproducibility.
        max_assets: Optional cap on the pooled asset count (``None`` = all).

    Returns:
        ``(dataset, reports)``.

    Raises:
        FileNotFoundError: If the data root is missing.
    """
    set_seed(seed)
    if not _DATA_ROOT.is_dir():
        raise FileNotFoundError(f"distill_students: missing data root {_DATA_ROOT}")

    dataset, _issues = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=FeatureConfig(),
        oracle_cfg=_ORACLE_CFG,
        timeframe=_TIMEFRAME,
        max_assets=max_assets,
    )

    splits = purged_walk_forward(
        dataset.timestamps,
        label_horizon=_LABEL_HORIZON,
        embargo=_EMBARGO,
        n_folds=_N_FOLDS,
        pooled_groups=dataset.groups,
    )

    X_values = np.ascontiguousarray(dataset.X.to_numpy(dtype=np.float64))
    reports = [distil_side(dataset, splits, X_values, side) for side in ("low", "high")]
    return dataset, reports


def _print_report(dataset: PooledDataset, reports: list[SideReport]) -> None:
    """Print the per-side summary: capacity, features, OOS vs GBDT, fidelity."""
    print()
    print("=" * 72)
    print("cfd10 STUDENT distillation -- shallow Pine-exportable trees")
    print("=" * 72)
    print(f"  pool: {dataset.n_assets} assets, {dataset.n_rows} rows, "
          f"{dataset.X.shape[1]} features")
    print(f"  student cap: depth<={_STUDENT_CFG.max_depth}, "
          f"leaves<={_STUDENT_CFG.max_leaves}, "
          f"min_samples_leaf={_STUDENT_CFG.min_samples_leaf}")
    print("-" * 72)
    for rep in reports:
        turn = "top" if rep.side == "high" else "bottom"
        verdict = "NOT WORSE" if rep.student_oos.f1 >= rep.gbdt_oos.f1 else "below GBDT"
        print(f"  {rep.side.upper()} ({turn} turns):")
        print(f"    positives      : {rep.n_pos} (rate {rep.positive_rate * 100:.3f}%)")
        print(f"    tree           : depth={rep.depth}, leaves={rep.leaves}")
        print(f"    FEATURES USED  : {rep.features}")
        print(
            f"    student OOS    : F1={rep.student_oos.f1:.4f} "
            f"(P={rep.student_oos.precision:.4f}, R={rep.student_oos.recall:.4f}, "
            f"tp={rep.student_oos.tp}, fp={rep.student_oos.fp}, fn={rep.student_oos.fn}, "
            f"n_pred={rep.student_oos.n_pred})"
        )
        print(
            f"    GBDT OOS       : F1={rep.gbdt_oos.f1:.4f} "
            f"(P={rep.gbdt_oos.precision:.4f}, R={rep.gbdt_oos.recall:.4f}) [{verdict}]"
        )
        print(f"    fidelity       : {rep.fidelity:.4f} (student vs GBDT call agreement)")
        print(f"    artefacts      : {rep.pkl_path.name}, {rep.json_path.name}, "
              f"{rep.rules_path.name}")
        print("-" * 72)


def main() -> None:
    """Entry point: run the distillation and print the per-side summary."""
    dataset, reports = run()
    _print_report(dataset, reports)
    print(f"  artefacts dir -> {_OUT_DIR}")


if __name__ == "__main__":
    main()
