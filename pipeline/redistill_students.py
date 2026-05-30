"""Re-distil DEEPER, SMALL Pine-portable students that recover the GBDT ranking.

Run with::

    uv run python pipeline/redistill_students.py

The shallow depth-4 student (``pipeline/distill_students.py``) recovered far too
little of the GBDT teacher's ranking (LOW event-F1 ~0.024 vs GBDT ~0.126). This
pipeline searches a small family of *deeper but still Pine-portable* students and
picks, per side, the one that best recovers the teacher's threshold-free ranking
(n-weighted Average Precision) while staying small enough to hand-transcribe into
Pine.

Pipeline (per side ``low`` / ``high``)
-------------------------------------
1. Build the multi-asset pooled corpus with ``FeatureConfig(include_overextension
   =True)`` (the AP-ablation-selected 45-feature bank) and the famous-lows-
   validated STRUCTURAL oracle (nest ``(20, 50, 100, 200)``, linear weight curve,
   10% drawdown, horizon 60, tau 0.60/0.30).
2. Derive the binary label (tier ``in {strong, regular}`` -> ``y = 1``) and the
   n-weighted sample weight ``w = where(y == 1, pos_boost * (1 + score), 1)`` with
   a ``pos_boost`` tuned per side (NO ``class_weight='balanced'``: balancing
   floods leaves and kills the sparse high-precision cut).
3. For each VARIANT — ``tree@depth6``, ``tree@depth8``, ``gboost@(40, d3)``,
   ``gboost@(80, d2)`` — fit a fresh model per purged walk-forward fold (6 folds)
   and score it out-of-sample two ways. The pooled corpus is ~0.47M rows and
   ~99.8% negative, so each FIT keeps every positive plus a capped multiple of
   negatives (``_NEG_PER_POS``); the held-out test block is always scored in full,
   so only the training matrix is shrunk, never the evaluation:
   * event-F1 at 3-bar tolerance, threshold tuned on the *train* fold only
     (identical leakage-safe protocol to the GBDT baseline);
   * n-weighted Average Precision (threshold-free): the pooled OOS scores ranked
     against the labels with ``sample_weight = 1 + score`` on positives.
4. PICK the best variant: highest OOS AP(n-wt) that recovers ``>= 70%`` of the
   reference GBDT AP (HIGH ``0.0597`` / LOW ``0.0692``), breaking near-ties toward
   the SMALLEST / shallowest model (Pine portability).
5. Refit the CHOSEN variant on ALL pooled rows (the deployable student) and save
   ``outputs/student_{side}_v2.{pkl,json}`` plus readable rules to
   ``outputs/student_{side}_v2_rules.txt``.

This is an oracle-vs-detector OOS evaluation only -- no Pine-signal comparison and
no PnL. The 400-tree GBDT is NOT refit here; its reference AP is the known value
from the full eval. Import-safe (nothing runs on import).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, _tree

from cfd10.cv_module import Fold, purged_walk_forward
from cfd10.data_module import PooledDataset, build_pooled_dataset
from cfd10.eval_module import average_precision, event_prf
from cfd10.feature_module import FeatureConfig
from cfd10.label_module import OracleConfig
from cfd10.student_module import (
    StudentConfig,
    ensemble_features,
    ensemble_to_rules,
    fit_student,
    student_features,
    tree_to_rules,
)
from cfd10.teacher_module.baseline import _positive_indices, _tune_threshold
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DATA_ROOT: Path = _REPO_ROOT / "data" / "raw_v16"
_OUT_DIR: Path = _REPO_ROOT / "outputs"

_SEED: int = 42
_TIMEFRAME: str = "1D"
_TOLERANCE: int = 3
_LABEL_HORIZON: int = 60
_EMBARGO: int = 5
_N_FOLDS: int = 6
_N_THRESHOLD_GRID: int = 50

# Negative-class subsampling ratio for the FIT only (the pooled corpus is ~0.47M
# rows, ~99.8% negative; a single-threaded GradientBoostingClassifier on the full
# expanding train fold costs minutes per fit). We keep EVERY positive and a
# bounded multiple of negatives, sampled deterministically per fold. This is a
# standard remedy for an extreme class imbalance and leaves the supervision
# unchanged in spirit (the oracle weight still scales the kept positives). All
# OOS scoring (event-F1 AND n-weighted AP) is still done on the FULL test block,
# so the evaluation is not subsampled -- only the training matrix is shrunk.
_NEG_PER_POS: int = 40

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

# The AP-ablation-selected feature bank: 45 features incl. the overextension /
# vol-regime top tells. (Per-asset z-debias is OFF -- it hurt AP.)
_FEATURE_CFG = FeatureConfig(include_overextension=True)

# Reference GBDT n-weighted AP from the full pooled eval (NOT recomputed here).
_GBDT_AP_REF: dict[str, float] = {"low": 0.0692, "high": 0.0597}

# A variant must recover at least this fraction of the reference GBDT AP to be
# eligible as the deployable student.
_RECOVERY_FLOOR: float = 0.70

# Functional deployability gate: a chosen student's OOS event-F1 must be at least
# this fraction of the best F1 across the variants. A model that ranks well (high
# AP) but whose train-tuned threshold flags ~no OOS events (F1 ~0) is not a usable
# detector -- this floor excludes that degenerate gradient-boost collapse.
_DEPLOY_F1_FRAC: float = 0.50

# Positive-boost grid (the n-weight multiplier on turns). Tuned per side by OOS
# AP on the cheap reference variant; kept in the brief's ~5..40 range so leaves
# stay separable (avoid both the flood and the all-negative collapse).
_POS_BOOST_GRID: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

# The cheap variant used to tune ``pos_boost`` (a single depth-6 tree).
_BOOST_TUNE_VARIANT = StudentConfig(
    kind="tree", max_depth=6, max_leaves=32, min_samples_leaf=30, seed=_SEED
)

# The candidate students, smallest/shallowest first so ties break toward Pine
# portability. Trees carry an explicit leaf cap; gboost stays tiny.
_VARIANTS: tuple[tuple[str, StudentConfig], ...] = (
    ("tree@d6", StudentConfig(kind="tree", max_depth=6, max_leaves=32, min_samples_leaf=30, seed=_SEED)),
    ("tree@d8", StudentConfig(kind="tree", max_depth=8, max_leaves=64, min_samples_leaf=30, seed=_SEED)),
    ("gboost@(40,d3)", StudentConfig(kind="gboost", n_estimators=40, gb_max_depth=3, learning_rate=0.1, min_samples_leaf=30, seed=_SEED)),
    ("gboost@(80,d2)", StudentConfig(kind="gboost", n_estimators=80, gb_max_depth=2, learning_rate=0.1, min_samples_leaf=30, seed=_SEED)),
)


# --------------------------------------------------------------------------- #
# Result containers.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VariantReport:
    """One variant's OOS ranking + event scores on a side.

    Attributes:
        name: Human label (e.g. ``"tree@d6"``).
        kind: ``"tree"`` or ``"gboost"``.
        size: Compact size descriptor (e.g. ``"depth=7 leaves=41"`` or
            ``"n_estimators=40 depth=3"``).
        oos_ap_nwt: Pooled OOS n-weighted Average Precision.
        oos_f1: Pooled OOS event-F1 at the tolerance.
        oos_precision / oos_recall: Pooled OOS event precision / recall.
        recovery: ``oos_ap_nwt / gbdt_ap`` (fraction of the teacher AP recovered).
        eligible: ``True`` iff ``recovery >= _RECOVERY_FLOOR``.
        n_pred: Pooled OOS predicted event count.
    """

    name: str
    kind: str
    size: str
    oos_ap_nwt: float
    oos_f1: float
    oos_precision: float
    oos_recall: float
    recovery: float
    eligible: bool
    n_pred: int


@dataclass(frozen=True)
class SideReport:
    """One side's chosen deployable student and the full variant comparison.

    Attributes:
        side: ``"high"`` (top turns) or ``"low"`` (bottom turns).
        n_pos: Pooled positive (strong/regular) count.
        positive_rate: Pooled positive fraction.
        pos_boost: The tuned positive-boost used for the n-weight.
        gbdt_ap: The reference GBDT n-weighted AP for this side.
        variants: Every variant's :class:`VariantReport` (smallest-first order).
        chosen: The winning variant's name.
        chosen_kind: The winning variant's kind.
        chosen_size: The winning (all-rows refit) model's size descriptor.
        features: Feature names the chosen all-rows model uses (drives Pine).
        pkl_path / json_path / rules_path: Saved artefact paths.
    """

    side: str
    n_pos: int
    positive_rate: float
    pos_boost: float
    gbdt_ap: float
    variants: list[VariantReport]
    chosen: str
    chosen_kind: str
    chosen_size: str
    features: list[str]
    pkl_path: Path
    json_path: Path
    rules_path: Path

    def chosen_report(self) -> VariantReport:
        """Return the :class:`VariantReport` of the chosen variant."""
        return next(v for v in self.variants if v.name == self.chosen)


# --------------------------------------------------------------------------- #
# Side label / weight derivation (the n-weighted sample-weight convention).   #
# --------------------------------------------------------------------------- #


def _side_columns(side: str) -> tuple[str, str]:
    """Return the ``(tier_col, score_col)`` oracle columns for ``side``."""
    if side == "high":
        return "top_tier", "top_score"
    if side == "low":
        return "bottom_tier", "bottom_score"
    raise ValueError(f"redistill: unknown side {side!r} (expected 'high'/'low')")


def _label_and_score(
    data: PooledDataset, side: str
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Derive the binary label and the raw oracle score for ``side``.

    Args:
        data: The pooled corpus.
        side: ``"high"`` or ``"low"``.

    Returns:
        ``(y, oracle_score)`` aligned to the pooled rows; ``oracle_score`` is the
        oracle pivot score in ``[0, 1]`` (``0`` on non-turns).
    """
    tier_col, score_col = _side_columns(side)
    tiers = data.labels[tier_col].to_numpy()
    y = np.isin(tiers, list(_POSITIVE_TIERS)).astype(np.int64)
    oracle_score = data.labels[score_col].to_numpy(dtype=np.float64)
    return y, oracle_score


def _sample_weight(
    y: NDArray[np.int64], oracle_score: NDArray[np.float64], pos_boost: float
) -> NDArray[np.float64]:
    """The n-weighted training weight ``w = where(y==1, pos_boost*(1+score), 1)``.

    Args:
        y: Binary labels.
        oracle_score: Oracle pivot score in ``[0, 1]`` (``0`` off-turn).
        pos_boost: Positive-class boost multiplier.

    Returns:
        Per-row ``float64`` sample weight; negatives weight ``1.0``.
    """
    return np.where(y == 1, pos_boost * (1.0 + oracle_score), 1.0).astype(np.float64)


def _ap_weight(
    y: NDArray[np.int64], oracle_score: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The n-weighted AP weight ``w = where(y==1, 1+score, 1)`` (threshold-free).

    The brief's AP convention: positives carry ``1 + score`` so nailing a heavy
    pivot lifts the score more than a light one; negatives carry ``1.0``.

    Args:
        y: Binary labels.
        oracle_score: Oracle pivot score in ``[0, 1]``.

    Returns:
        Per-row ``float64`` AP sample weight.
    """
    return np.where(y == 1, 1.0 + oracle_score, 1.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# Unified student scoring.                                                     #
# --------------------------------------------------------------------------- #


def _student_proba(
    model: DecisionTreeClassifier | GradientBoostingClassifier,
    X_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Positive-class probabilities from a fitted student (``0`` if no positives).

    Works for both kinds: a tree whose train fold saw only negatives has no
    class ``1`` column (return zeros); otherwise ``predict_proba`` is sliced at
    the positive column.

    Args:
        model: A fitted tree or gradient-boosting student.
        X_values: Dense rows to score.

    Returns:
        The class-1 probability per row as a ``float64`` array.
    """
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X_values.shape[0], dtype=np.float64)
    col = classes.index(1)
    return np.ascontiguousarray(model.predict_proba(X_values)[:, col], dtype=np.float64)


def _subsample_fit_rows(
    rows: NDArray[np.int64],
    y: NDArray[np.int64],
    seed: int,
) -> NDArray[np.int64]:
    """Keep all positive rows + a bounded multiple of negatives (fit only).

    The pooled corpus is overwhelmingly negative; fitting on every negative is
    both slow and unnecessary for a sparse-positive detector. This keeps EVERY
    positive in ``rows`` and a deterministic random sample of at most
    ``_NEG_PER_POS`` negatives per positive. The returned indices are a subset of
    ``rows`` (sorted) used to build the training matrix; evaluation is unaffected
    because the test block is scored in full elsewhere.

    Args:
        rows: Candidate fit-row indices (a fold's train block, or all rows).
        y: Binary labels aligned to the full matrix (indexed by ``rows``).
        seed: RNG seed for the deterministic negative draw.

    Returns:
        Sorted ``int64`` subset of ``rows`` (all positives + capped negatives). If
        there are no positives the rows are returned unchanged (nothing to anchor
        a subsample on).
    """
    labels = y[rows]
    pos = rows[labels == 1]
    neg = rows[labels == 0]
    if pos.size == 0:
        return np.ascontiguousarray(np.sort(rows), dtype=np.int64)
    keep_neg = min(neg.size, _NEG_PER_POS * int(pos.size))
    if keep_neg < neg.size:
        rng = np.random.default_rng(seed)
        neg = rng.choice(neg, size=keep_neg, replace=False)
    out = np.concatenate([pos, neg])
    return np.ascontiguousarray(np.sort(out), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Per-fold OOS (mirrors the GBDT harness: train-only threshold tuning).        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FoldOos:
    """One fold's OOS predictions / scores in original-row index space."""

    pred_idx: NDArray[np.int64]  # flagged bars (thresholded) for event-F1.
    true_idx: NDArray[np.int64]  # true pivot bars in the test block.
    test_idx: NDArray[np.int64]  # all test-row indices (score alignment).
    test_score: NDArray[np.float64]  # class-1 proba per test row (for AP).


def _variant_fold_oos(
    X_values: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    fold: Fold,
    cfg: StudentConfig,
) -> _FoldOos:
    """Fit a fresh student on the train block; return OOS preds + scores.

    The decision threshold is tuned to maximise event-F1 on the *train* fold only
    (reusing the baseline's :func:`_tune_threshold`), then frozen for the held-out
    test block. Scores (for the threshold-free AP) are the raw class-1
    probabilities on the test rows.

    Args:
        X_values: Dense feature matrix as a ``float64`` array.
        y: Binary labels aligned to ``X_values``.
        sample_weight: n-weighted train weights aligned to ``X_values``.
        fold: ``(train_idx, test_idx)`` original-row indices.
        cfg: The variant's student configuration.

    Returns:
        The fold's :class:`_FoldOos`.
    """
    train_idx, test_idx = fold
    # Fit on all positives + a capped negative sample (bounds the cost on the
    # ~0.47M-row pool); seed by the fold's first row so each fold draws stably.
    fit_idx = _subsample_fit_rows(
        train_idx, y, seed=cfg.seed + int(train_idx[0]) if train_idx.size else cfg.seed
    )
    model = fit_student(X_values[fit_idx], y[fit_idx], sample_weight[fit_idx], cfg)

    # Tune the decision threshold to maximise event-F1 on the FULL train block
    # (only the *fit* matrix was subsampled, never the threshold tuning or the
    # OOS test scoring). This mirrors the GBDT baseline harness exactly, so the
    # student's OOS event-F1 is directly comparable to the GBDT reference.
    train_proba = _student_proba(model, X_values[train_idx])
    test_proba = _student_proba(model, X_values[test_idx])

    train_true_idx = np.ascontiguousarray(train_idx[y[train_idx] == 1], dtype=np.int64)
    test_true_idx = np.ascontiguousarray(test_idx[y[test_idx] == 1], dtype=np.int64)

    threshold, _train_f1 = _tune_threshold(
        train_proba, train_idx, train_true_idx, _threshold_cfg()
    )
    pred_idx = _positive_indices(test_proba, test_idx, threshold)

    return _FoldOos(
        pred_idx=pred_idx,
        true_idx=test_true_idx,
        test_idx=np.ascontiguousarray(test_idx, dtype=np.int64),
        test_score=test_proba,
    )


def _threshold_cfg():  # type: ignore[no-untyped-def]  # tiny shim, see below.
    """Return a minimal config object exposing the two fields ``_tune_threshold`` reads.

    :func:`cfd10.teacher_module.baseline._tune_threshold` only touches
    ``cfg.tolerance`` and ``cfg.n_threshold_grid``; rather than build a full
    ``GBDTConfig`` we hand it a frozen stand-in carrying just those two knobs.

    Returns:
        A frozen object with ``tolerance`` and ``n_threshold_grid`` attributes.
    """
    return _ThresholdCfg(tolerance=_TOLERANCE, n_threshold_grid=_N_THRESHOLD_GRID)


@dataclass(frozen=True)
class _ThresholdCfg:
    """Minimal stand-in carrying only the fields ``_tune_threshold`` reads."""

    tolerance: int
    n_threshold_grid: int


def _eval_variant(
    X_values: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    ap_weight: NDArray[np.float64],
    splits: list[Fold],
    name: str,
    cfg: StudentConfig,
    gbdt_ap: float,
) -> VariantReport:
    """Fit + OOS-score one variant across all folds (event-F1 and n-weighted AP).

    Test blocks are disjoint in original-row space, so the per-fold OOS bars /
    scores concatenate into one global slice that is matched / ranked once (no
    cross-fold double counting).

    Args:
        X_values: Dense feature matrix as a ``float64`` array.
        y: Binary labels aligned to ``X_values``.
        sample_weight: n-weighted train weights.
        ap_weight: n-weighted AP weights (``1 + score`` on positives).
        splits: Purged walk-forward folds.
        name: The variant label.
        cfg: The variant's student configuration.
        gbdt_ap: Reference GBDT AP for the side (for the recovery fraction).

    Returns:
        The variant's :class:`VariantReport`.
    """
    pred_blocks: list[NDArray[np.int64]] = []
    true_blocks: list[NDArray[np.int64]] = []
    score_rows: list[NDArray[np.int64]] = []
    score_vals: list[NDArray[np.float64]] = []
    for fold in splits:
        oos = _variant_fold_oos(X_values, y, sample_weight, fold, cfg)
        pred_blocks.append(oos.pred_idx)
        true_blocks.append(oos.true_idx)
        score_rows.append(oos.test_idx)
        score_vals.append(oos.test_score)

    all_pred = np.concatenate(pred_blocks) if pred_blocks else np.empty(0, np.int64)
    all_true = np.concatenate(true_blocks) if true_blocks else np.empty(0, np.int64)
    precision, recall, f1 = event_prf(all_pred, all_true, _TOLERANCE)

    # Threshold-free, n-weighted AP over the pooled OOS rows.
    rows = np.concatenate(score_rows) if score_rows else np.empty(0, np.int64)
    scores = np.concatenate(score_vals) if score_vals else np.empty(0, np.float64)
    ap = average_precision(scores, y[rows], sample_weight=ap_weight[rows])

    recovery = ap / gbdt_ap if gbdt_ap > 0.0 else 0.0
    report = VariantReport(
        name=name,
        kind=cfg.kind,
        size=_size_descriptor_cfg(cfg),
        oos_ap_nwt=ap,
        oos_f1=f1,
        oos_precision=precision,
        oos_recall=recall,
        recovery=recovery,
        eligible=recovery >= _RECOVERY_FLOOR,
        n_pred=int(all_pred.size),
    )
    logger.info(
        "variant %-14s [%s]: OOS AP(n-wt)=%.4f (%.0f%% of GBDT) | F1=%.4f "
        "(P=%.4f R=%.4f n_pred=%d)",
        name,
        report.size,
        ap,
        100.0 * recovery,
        f1,
        precision,
        recall,
        report.n_pred,
    )
    return report


# --------------------------------------------------------------------------- #
# pos_boost tuning + variant selection.                                        #
# --------------------------------------------------------------------------- #


def _tune_pos_boost(
    X_values: NDArray[np.float64],
    y: NDArray[np.int64],
    oracle_score: NDArray[np.float64],
    ap_weight: NDArray[np.float64],
    splits: list[Fold],
    side: str,
) -> float:
    """Pick the ``pos_boost`` maximising OOS AP on the cheap reference variant.

    A single depth-6 tree is fit per fold at each candidate boost; the boost with
    the highest pooled OOS n-weighted AP wins. This keeps tuning cheap (trees are
    fast) while choosing a weighting that actually helps the ranking.

    Args:
        X_values: Dense feature matrix.
        y: Binary labels.
        oracle_score: Oracle pivot score in ``[0, 1]``.
        ap_weight: n-weighted AP weights.
        splits: Purged walk-forward folds.
        side: ``"high"`` / ``"low"`` (logging only).

    Returns:
        The selected ``pos_boost``.
    """
    gbdt_ap = _GBDT_AP_REF[side]
    best_boost = _POS_BOOST_GRID[0]
    best_ap = -1.0
    for boost in _POS_BOOST_GRID:
        weight = _sample_weight(y, oracle_score, boost)
        rep = _eval_variant(
            X_values, y, weight, ap_weight, splits,
            f"boost={boost:g}", _BOOST_TUNE_VARIANT, gbdt_ap,
        )
        if rep.oos_ap_nwt > best_ap:
            best_ap = rep.oos_ap_nwt
            best_boost = boost
    logger.info(
        "side[%s]: tuned pos_boost=%g (OOS AP=%.4f on the depth-6 probe)",
        side,
        best_boost,
        best_ap,
    )
    return best_boost


def _select_variant(variants: list[VariantReport]) -> VariantReport:
    """Pick the best DEPLOYABLE variant: best eligible AP, ties -> smallest model.

    A student that ranks superbly (high AP) but, with its train-tuned threshold,
    flags essentially no OOS events is useless as the deployable Pine detector: it
    would never (or almost never) emit a marker. In practice the small gradient
    boosters here do exactly that -- their F1-optimal train threshold lands above
    nearly every test probability, collapsing OOS event-F1 to ~0 with a handful
    of flagged bars despite a top AP. So selection first applies a *functional*
    deployability gate on OOS event-F1, then maximises the threshold-free
    n-weighted AP within that set.

    Selection rule (in order):

    1. Deployability gate -- keep only variants whose OOS event-F1 is at least
       :data:`_DEPLOY_F1_FRAC` of the best F1 across all variants (i.e. genuinely
       usable detectors, not a 2-event degenerate). If *none* qualify (every
       variant collapsed), drop the gate so the run still returns the best-AP
       model, flagged downstream as a non-functional fallback.
    2. Recovery floor -- among the gated set prefer those recovering
       ``>= _RECOVERY_FLOOR`` of the GBDT AP; if none clear it, keep the gated set.
    3. Best AP with a smallest-model tie-break -- take the highest AP, treat AP
       within 2% (relative) of it as a near-tie, and among the near-ties pick the
       SMALLEST / shallowest model (``_VARIANTS`` is smallest-first, so the
       earliest index wins). Pine portability wins ties.

    Args:
        variants: The per-variant reports (in ``_VARIANTS`` smallest-first order).

    Returns:
        The chosen :class:`VariantReport`.
    """
    order = {v.name: i for i, v in enumerate(variants)}

    # 1) Functional deployability gate: the detector must actually emit events
    #    OOS, measured by event-F1 relative to the best variant (a model with F1
    #    ~0 / a couple of flagged bars is not a usable detector, however well it
    #    ranks). This is what excludes the high-AP gradient-boost collapse.
    best_f1 = max(v.oos_f1 for v in variants)
    f1_floor = _DEPLOY_F1_FRAC * best_f1
    deployable = [v for v in variants if v.oos_f1 >= f1_floor and v.oos_f1 > 0.0]
    pool = deployable if deployable else variants

    # 2) Prefer variants clearing the recovery floor, else keep the gated pool.
    eligible = [v for v in pool if v.eligible]
    pool = eligible if eligible else pool

    # 3) Best AP, near-ties broken toward the smallest model.
    best_ap = max(v.oos_ap_nwt for v in pool)
    near_ties = [v for v in pool if v.oos_ap_nwt >= best_ap * 0.98]
    return min(near_ties, key=lambda v: order[v.name])


# --------------------------------------------------------------------------- #
# Size descriptors + artefact serialisation (tree OR gboost).                  #
# --------------------------------------------------------------------------- #


def _size_descriptor_cfg(cfg: StudentConfig) -> str:
    """Compact size label for a variant's *configuration* (pre-fit caps)."""
    if cfg.kind == "gboost":
        return f"n_estimators={cfg.n_estimators} depth={cfg.gb_max_depth}"
    return f"max_depth={cfg.max_depth} max_leaves={cfg.max_leaves}"


def _size_descriptor_model(
    model: DecisionTreeClassifier | GradientBoostingClassifier,
) -> str:
    """Compact size label for a *fitted* model (realised depth / leaves)."""
    if isinstance(model, GradientBoostingClassifier):
        depths = [int(model.estimators_[s, 0].get_depth()) for s in range(model.n_estimators_)]
        return f"n_estimators={model.n_estimators_} max_stage_depth={max(depths)}"
    return f"depth={int(model.get_depth())} leaves={int(model.get_n_leaves())}"


def _tree_nodes(tree, feature_names: list[str], classifier: bool) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    """Serialise a tree's node array to plain dicts (classifier or regressor).

    Args:
        tree: A fitted ``DecisionTreeClassifier`` (leaf -> pos_proba / predict) or
            stage ``DecisionTreeRegressor`` (leaf -> raw value).
        feature_names: Column names aligned to the fit matrix.
        classifier: ``True`` for the classification tree, ``False`` for a stage
            regressor.

    Returns:
        The node list (each a JSON-serialisable dict).
    """
    inner = tree.tree_
    pos_col = -1
    if classifier:
        classes = list(tree.classes_)
        pos_col = classes.index(1) if 1 in classes else -1

    nodes: list[dict[str, object]] = []
    for node in range(inner.node_count):
        feat = int(inner.feature[node])
        is_leaf = feat == _tree.TREE_UNDEFINED
        entry: dict[str, object] = {
            "node_id": node,
            "is_leaf": is_leaf,
            "feature": None if is_leaf else feature_names[feat],
            "feature_index": None if is_leaf else feat,
            "threshold": None if is_leaf else float(inner.threshold[node]),
            "left": None if is_leaf else int(inner.children_left[node]),
            "right": None if is_leaf else int(inner.children_right[node]),
            "n_samples": int(inner.n_node_samples[node]),
        }
        if classifier:
            counts = inner.value[node][0]
            total = float(counts.sum())
            pos_mass = float(counts[pos_col]) if pos_col >= 0 else 0.0
            prob = pos_mass / total if total > 0.0 else 0.0
            entry["pos_proba"] = round(prob, 6)
            entry["predict"] = int(prob >= 0.5)
        else:
            entry["value"] = float(inner.value[node][0][0])
        nodes.append(entry)
    return nodes


def _model_to_json(
    model: DecisionTreeClassifier | GradientBoostingClassifier,
    feature_names: list[str],
) -> dict[str, object]:
    """Serialise a fitted student (tree OR gboost) to an export-friendly dict.

    A tree keeps the legacy schema (``model_kind="tree"``, single ``nodes`` array)
    so the existing Pine export / parity path is unaffected. A gboost adds
    ``model_kind="gboost"``, the ``init`` log-odds, ``learning_rate`` and a
    ``stages`` list of per-estimator regressor node arrays; the Pine score is
    ``init + learning_rate * sum(stage leaf value)`` with a positive call iff
    ``raw > 0``.

    Args:
        model: The fitted student.
        feature_names: Column names aligned to the fit matrix.

    Returns:
        A JSON-serialisable dict.
    """
    if isinstance(model, GradientBoostingClassifier):
        from cfd10.student_module.distill import _gboost_init_raw

        stages = [
            _tree_nodes(model.estimators_[s, 0], feature_names, classifier=False)
            for s in range(model.n_estimators_)
        ]
        return {
            "model_kind": "gboost",
            "n_estimators": int(model.n_estimators_),
            "learning_rate": float(model.learning_rate),
            "init": round(_gboost_init_raw(model), 8),
            "decision": "positive iff init + learning_rate * sum(stage leaf) > 0",
            "features": ensemble_features(model, feature_names),
            "stages": stages,
        }

    nodes = _tree_nodes(model, feature_names, classifier=True)
    return {
        "model_kind": "tree",
        "n_nodes": int(model.tree_.node_count),
        "max_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "features": student_features(model, feature_names),
        "nodes": nodes,
    }


def _readable_rules(
    model: DecisionTreeClassifier | GradientBoostingClassifier,
    feature_names: list[str],
) -> str:
    """Render the chosen model's rules (tree or additive gboost) as ASCII text."""
    if isinstance(model, GradientBoostingClassifier):
        return ensemble_to_rules(model, feature_names)
    return tree_to_rules(model, feature_names)


def _save_artefacts(
    side: str,
    model: DecisionTreeClassifier | GradientBoostingClassifier,
    feature_names: list[str],
) -> tuple[Path, Path, Path]:
    """Persist the chosen deployable model as pickle + JSON + readable rules.

    Args:
        side: ``"high"`` or ``"low"`` (names the ``*_v2`` files).
        model: The all-rows refit deployable student.
        feature_names: Column names aligned to the fit matrix.

    Returns:
        ``(pkl_path, json_path, rules_path)``.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = _OUT_DIR / f"student_{side}_v2.pkl"
    json_path = _OUT_DIR / f"student_{side}_v2.json"
    rules_path = _OUT_DIR / f"student_{side}_v2_rules.txt"

    with pkl_path.open("wb") as fh:
        pickle.dump(model, fh)
    json_path.write_text(
        json.dumps(_model_to_json(model, feature_names), indent=2), encoding="utf-8"
    )

    kind = "gboost" if isinstance(model, GradientBoostingClassifier) else "tree"
    used = (
        ensemble_features(model, feature_names)
        if isinstance(model, GradientBoostingClassifier)
        else student_features(model, feature_names)
    )
    header = (
        f"# cfd10 student v2 ruleset -- {side.upper()} "
        f"({'top' if side == 'high' else 'bottom'} turns)\n"
        f"# kind={kind}  size={_size_descriptor_model(model)}\n"
        f"# features_used={used}\n"
        f"# scikit-learn routes 'feature <= threshold' to the if-branch.\n\n"
    )
    rules_path.write_text(
        header + _readable_rules(model, feature_names) + "\n", encoding="utf-8"
    )
    return pkl_path, json_path, rules_path


# --------------------------------------------------------------------------- #
# Per-side orchestration.                                                      #
# --------------------------------------------------------------------------- #


def redistil_side(
    data: PooledDataset,
    splits: list[Fold],
    X_values: NDArray[np.float64],
    side: str,
) -> SideReport:
    """Tune the boost, sweep the variants, pick + refit the deployable student.

    Args:
        data: The pooled corpus.
        splits: Shared purged walk-forward folds over the pooled time axis.
        X_values: Dense pooled feature matrix as a ``float64`` array.
        side: ``"high"`` or ``"low"``.

    Returns:
        The assembled :class:`SideReport`.
    """
    feature_names = list(data.X.columns)
    y, oracle_score = _label_and_score(data, side)
    ap_weight = _ap_weight(y, oracle_score)
    n_pos = int(y.sum())
    gbdt_ap = _GBDT_AP_REF[side]

    logger.info(
        "redistil_side[%s]: %d positives / %d rows (%.3f%%); GBDT AP ref=%.4f",
        side, n_pos, y.shape[0], 100.0 * n_pos / max(y.shape[0], 1), gbdt_ap,
    )

    # 1) Tune the positive boost on the cheap depth-6 probe.
    pos_boost = _tune_pos_boost(X_values, y, oracle_score, ap_weight, splits, side)
    sample_weight = _sample_weight(y, oracle_score, pos_boost)

    # 2) Sweep all variants at the tuned boost.
    variants = [
        _eval_variant(
            X_values, y, sample_weight, ap_weight, splits, name, cfg, gbdt_ap
        )
        for name, cfg in _VARIANTS
    ]

    # 3) Pick the deployable variant.
    chosen = _select_variant(variants)
    chosen_cfg = next(cfg for name, cfg in _VARIANTS if name == chosen.name)

    # 4) Refit the chosen variant on the full pool (all positives + capped
    #    negatives, consistent with how every variant was fit during selection).
    all_rows = np.arange(y.shape[0], dtype=np.int64)
    fit_idx = _subsample_fit_rows(all_rows, y, seed=chosen_cfg.seed)
    final_model = fit_student(
        data.X.iloc[fit_idx], y[fit_idx], sample_weight[fit_idx], chosen_cfg
    )
    if isinstance(final_model, GradientBoostingClassifier):
        features = ensemble_features(final_model, feature_names)
    else:
        features = student_features(final_model, feature_names)
    pkl_path, json_path, rules_path = _save_artefacts(side, final_model, feature_names)

    logger.info(
        "redistil_side[%s]: CHOSE %s [%s] -> OOS AP=%.4f (%.0f%% of GBDT), "
        "F1=%.4f, features=%s",
        side, chosen.name, _size_descriptor_model(final_model),
        chosen.oos_ap_nwt, 100.0 * chosen.recovery, chosen.oos_f1, features,
    )

    return SideReport(
        side=side,
        n_pos=n_pos,
        positive_rate=float(y.mean()) if y.size else 0.0,
        pos_boost=pos_boost,
        gbdt_ap=gbdt_ap,
        variants=variants,
        chosen=chosen.name,
        chosen_kind=chosen.kind,
        chosen_size=_size_descriptor_model(final_model),
        features=features,
        pkl_path=pkl_path,
        json_path=json_path,
        rules_path=rules_path,
    )


# --------------------------------------------------------------------------- #
# Top-level run + reporting.                                                    #
# --------------------------------------------------------------------------- #


def run(
    seed: int = _SEED, max_assets: int | None = None
) -> tuple[PooledDataset, list[SideReport]]:
    """Build the pool, re-distil both sides, save artefacts, return the reports.

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
        raise FileNotFoundError(f"redistill_students: missing data root {_DATA_ROOT}")

    dataset, _issues = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=_FEATURE_CFG,
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
    reports = [redistil_side(dataset, splits, X_values, side) for side in ("low", "high")]
    return dataset, reports


def _print_report(dataset: PooledDataset, reports: list[SideReport]) -> None:
    """Print the per-side variant sweep and the chosen deployable student."""
    print()
    print("=" * 76)
    print("cfd10 STUDENT RE-DISTILLATION v2 -- deeper, small, Pine-portable")
    print("=" * 76)
    print(f"  pool: {dataset.n_assets} assets, {dataset.n_rows} rows, "
          f"{dataset.X.shape[1]} features (+overext)")
    print(f"  CV: {_N_FOLDS} purged folds, tol={_TOLERANCE}, horizon={_LABEL_HORIZON}, "
          f"embargo={_EMBARGO}")
    print(f"  recovery floor: >= {_RECOVERY_FLOOR * 100:.0f}% of the GBDT n-wt AP; "
          f"deploy gate: OOS F1 >= {_DEPLOY_F1_FRAC * 100:.0f}% of best F1")
    print("-" * 76)
    for rep in reports:
        turn = "top" if rep.side == "high" else "bottom"
        print(f"  {rep.side.upper()} ({turn} turns):  "
              f"positives={rep.n_pos} (rate {rep.positive_rate * 100:.3f}%), "
              f"pos_boost={rep.pos_boost:g}")
        print(f"    GBDT reference AP(n-wt) = {rep.gbdt_ap:.4f}")
        print(f"    {'variant':<16}{'size':<34}{'AP(n-wt)':>10}{'%GBDT':>8}"
              f"{'F1':>8}{'':>4}")
        best_f1 = max(v.oos_f1 for v in rep.variants)
        for v in rep.variants:
            star = " <=" if v.name == rep.chosen else ""
            # Flag why a high-AP variant might be skipped: non-functional detector
            # (F1 collapse) or below the AP recovery floor.
            if v.oos_f1 < _DEPLOY_F1_FRAC * best_f1 or v.oos_f1 <= 0.0:
                flag = " (F1 collapse: not deployable)"
            elif not v.eligible:
                flag = " (below AP floor)"
            else:
                flag = ""
            print(f"    {v.name:<16}{v.size:<34}{v.oos_ap_nwt:>10.4f}"
                  f"{100.0 * v.recovery:>7.0f}%{v.oos_f1:>8.4f}{star}{flag}")
        cr = rep.chosen_report()
        print(f"    CHOSEN -> {rep.chosen} [{rep.chosen_kind}], "
              f"deployed size {rep.chosen_size}")
        print(f"      OOS AP(n-wt)={cr.oos_ap_nwt:.4f} "
              f"({100.0 * cr.recovery:.0f}% of GBDT {rep.gbdt_ap:.4f}); "
              f"OOS event-F1={cr.oos_f1:.4f} "
              f"(P={cr.oos_precision:.4f}, R={cr.oos_recall:.4f})")
        print(f"      FEATURES USED ({len(rep.features)}): {rep.features}")
        print(f"      artefacts: {rep.pkl_path.name}, {rep.json_path.name}, "
              f"{rep.rules_path.name}")
        print("-" * 76)


def main() -> None:
    """Entry point: run the re-distillation and print the per-side summary."""
    dataset, reports = run()
    _print_report(dataset, reports)
    print(f"  artefacts dir -> {_OUT_DIR}")


if __name__ == "__main__":
    main()
