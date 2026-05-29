"""GBDT (LightGBM) baseline: the yardstick + first out-of-sample turn result.

This module trains a per-side LightGBM binary classifier on the dense feature
bank to detect oracle turns, scored *out-of-sample* with leakage-safe purged
walk-forward folds and the event-level precision/recall/F1 metric.

Two design rules keep the score honest:

* **No test leakage in thresholding.** A probability classifier needs a decision
  threshold to emit events. We tune that threshold to maximise *event-F1 on the
  training fold only* (:func:`_tune_threshold`), then apply the frozen threshold
  to the held-out test fold. The test labels never influence the threshold.
* **Event scoring, not per-bar accuracy.** A flagged bar counts as a hit when it
  lands within ``cfg.tolerance`` bars of a true oracle turn
  (:func:`cfd10.eval_module.metrics.event_prf`), so near-misses are not punished
  as hard false positives — the right notion for sparse turn detection.

Labels are derived from the oracle tiers by the caller: tier ``in {strong,
regular}`` -> ``y = 1``, ``none`` -> ``0``; the matching ``*_weight`` column is
passed as the LightGBM sample weight so stronger turns dominate the fit.

Public API
----------
:class:`GBDTConfig` (frozen) — LightGBM hyper-parameters plus the event /
walk-forward knobs. :func:`fit_gbdt_cv` — fit + OOS-evaluate across folds,
returning per-fold and aggregated event metrics. The teacher registry / factory
that expose named baseline fitters live in :mod:`cfd10.teacher_module`.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfd10.cv_module import Fold
from cfd10.eval_module import event_prf, match_events
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

__all__ = [
    "GBDTConfig",
    "FoldResult",
    "fit_gbdt_cv",
]


# LightGBM 4.6 auto-assigns ``Column_N`` feature names when fitted on a bare
# ndarray, after which scikit-learn 1.8's ``predict`` warns that the prediction
# array carries no names. We deliberately work in dense ndarray space (the splits
# index rows by position), so this specific warning is pure library noise with no
# correctness impact.
_FEATURE_NAME_WARNING = (
    "X does not have valid feature names, but LGBMClassifier was fitted"
)


def _predict_proba(model: lgb.LGBMClassifier, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Class-1 probabilities, silencing the benign LightGBM feature-name warning.

    Args:
        model: A fitted LightGBM classifier.
        X: Dense feature rows to score.

    Returns:
        The positive-class probability per row as a ``float64`` array.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_FEATURE_NAME_WARNING, category=UserWarning
        )
        proba = model.predict_proba(X)[:, 1]
    return np.ascontiguousarray(proba, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GBDTConfig:
    """Immutable LightGBM baseline configuration.

    Attributes:
        num_leaves: Max leaves per tree (the dominant LightGBM capacity knob).
        max_depth: Max tree depth; ``-1`` means unlimited (capped by num_leaves).
        learning_rate: Boosting shrinkage.
        n_estimators: Number of boosting rounds (trees).
        min_child_samples: Minimum samples per leaf (regularisation; matters for
            the sparse positive class).
        subsample: Row subsampling fraction per tree (bagging).
        colsample_bytree: Feature subsampling fraction per tree.
        reg_lambda: L2 regularisation on leaf weights.
        scale_pos_weight: Extra multiplier on positive-class gradients. Combined
            with the oracle sample weights this counteracts the heavy class
            imbalance of turn labels.
        tolerance: Event-match tolerance in bars for
            :func:`cfd10.eval_module.metrics.event_prf`.
        label_horizon: Forward label span (bars) passed to the purged splitter;
            should equal the oracle horizon used to build the labels.
        embargo: Post-test-block embargo (bars) for the purged splitter.
        n_folds: Number of walk-forward test blocks.
        n_threshold_grid: Number of candidate decision thresholds swept on the
            train fold when tuning for event-F1.
        seed: Global RNG seed (also LightGBM ``random_state``).
    """

    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0

    tolerance: int = 3
    label_horizon: int = 40
    embargo: int = 5
    n_folds: int = 6

    n_threshold_grid: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the capacity, event and walk-forward knobs."""
        if self.num_leaves < 2:
            raise ValueError(f"GBDTConfig: num_leaves must be >= 2, got {self.num_leaves}")
        if self.n_estimators < 1:
            raise ValueError(
                f"GBDTConfig: n_estimators must be >= 1, got {self.n_estimators}"
            )
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                f"GBDTConfig: learning_rate must be in (0, 1], got {self.learning_rate}"
            )
        if self.tolerance < 0:
            raise ValueError(f"GBDTConfig: tolerance must be >= 0, got {self.tolerance}")
        if self.label_horizon < 0:
            raise ValueError(
                f"GBDTConfig: label_horizon must be >= 0, got {self.label_horizon}"
            )
        if self.embargo < 0:
            raise ValueError(f"GBDTConfig: embargo must be >= 0, got {self.embargo}")
        if self.n_folds < 2:
            raise ValueError(f"GBDTConfig: n_folds must be >= 2, got {self.n_folds}")
        if self.n_threshold_grid < 1:
            raise ValueError(
                f"GBDTConfig: n_threshold_grid must be >= 1, got {self.n_threshold_grid}"
            )

    def lgb_params(self) -> dict[str, object]:
        """Return the LightGBM ``LGBMClassifier`` keyword arguments.

        Returns:
            A kwargs dict spanning only the LightGBM-relevant fields (the event /
            walk-forward knobs are excluded), with fixed ``objective='binary'`` and
            silent verbosity.
        """
        return {
            "objective": "binary",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbosity": -1,
        }


# --------------------------------------------------------------------------- #
# Per-fold result container.                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldResult:
    """One fold's tuned threshold and out-of-sample event scores.

    Attributes:
        fold: 0-based fold index.
        threshold: Decision threshold tuned on the *train* fold.
        train_f1: Event-F1 achieved on the train fold at ``threshold`` (the tuning
            objective; reported for diagnostics, not as a result).
        precision: Out-of-sample event precision on the test fold.
        recall: Out-of-sample event recall on the test fold.
        f1: Out-of-sample event F1 on the test fold.
        n_true: Number of true oracle events in the test fold.
        n_pred: Number of events the model flagged in the test fold.
        n_train: Number of training rows in the fold.
        n_test: Number of test rows in the fold.
    """

    fold: int
    threshold: float
    train_f1: float
    precision: float
    recall: float
    f1: float
    n_true: int
    n_pred: int
    n_train: int
    n_test: int


# --------------------------------------------------------------------------- #
# Helpers.                                                                     #
# --------------------------------------------------------------------------- #


def _positive_indices(
    proba: NDArray[np.float64],
    base_idx: NDArray[np.int64],
    threshold: float,
) -> NDArray[np.int64]:
    """Map class-1 probabilities to original bar indices at/above ``threshold``.

    Args:
        proba: Class-1 probability per row (aligned to ``base_idx``).
        base_idx: Original bar index for each row.
        threshold: Inclusive decision threshold.

    Returns:
        Sorted original bar indices whose probability is ``>= threshold``.
    """
    flagged = base_idx[proba >= threshold]
    return np.ascontiguousarray(np.sort(flagged), dtype=np.int64)


def _tune_threshold(
    train_proba: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    train_true_idx: NDArray[np.int64],
    cfg: GBDTConfig,
) -> tuple[float, float]:
    """Pick the decision threshold maximising event-F1 on the *train* fold.

    Candidate thresholds are the quantiles of the train probabilities (so every
    threshold actually changes the flagged set), swept with the event metric at
    ``cfg.tolerance``. Ties are broken toward the **higher** threshold (fewer, more
    confident flags), and a degenerate empty train side falls back to ``0.5``.

    Args:
        train_proba: Class-1 probabilities on the train fold.
        train_idx: Original bar indices of the train rows.
        train_true_idx: Original bar indices of true oracle events in the train
            fold.
        cfg: Baseline configuration (``tolerance`` and ``n_threshold_grid``).

    Returns:
        ``(threshold, train_f1)`` — the chosen threshold and the train event-F1 it
        attains.
    """
    if train_proba.size == 0 or train_true_idx.size == 0:
        # No probabilities or no positives to match: nothing to tune against.
        return 0.5, 0.0

    quantiles = np.linspace(0.0, 1.0, cfg.n_threshold_grid)
    candidates = np.unique(np.quantile(train_proba, quantiles))
    # Include a hair below the max so the single most-confident bar can still flag.
    top = float(np.nextafter(float(train_proba.max()), 0.0))
    candidates = np.unique(np.concatenate([candidates, np.array([top])]))

    # Sparse-event detection: the F1 optimum flags only a small multiple of the
    # true-event count. Candidates that flag far more bars have hopeless precision
    # AND blow up the event match (O(n_pred * n_true) Hungarian) on large pooled
    # train folds. We descend from the most confident threshold and stop once the
    # flagged set exceeds a sane multiple of the true-event count: the flagged set
    # grows monotonically as the threshold falls, so every lower candidate is also
    # over budget. This bounds tuning cost without discarding the real optimum.
    max_pred = max(512, 6 * int(train_true_idx.size))

    best_threshold = float(candidates[-1])
    best_f1 = -1.0
    for thr in candidates[::-1]:
        pred_idx = _positive_indices(train_proba, train_idx, float(thr))
        if pred_idx.size > max_pred:
            break
        _p, _r, f1 = event_prf(pred_idx, train_true_idx, cfg.tolerance)
        # Descending sweep + strict ">" keeps the highest threshold among ties
        # (fewer false positives at equal F1).
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
    return best_threshold, float(max(best_f1, 0.0))


def _validate_inputs(
    X: pd.DataFrame,
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    splits: list[Fold],
) -> None:
    """Validate array shapes and the split index domain.

    Args:
        X: Feature matrix.
        y: Binary label array.
        sample_weight: Per-row sample weight.
        splits: Walk-forward folds.

    Raises:
        ValueError: If lengths disagree, ``splits`` is empty, ``y`` is not binary,
            or any fold index is out of range.
    """
    n = len(X)
    if y.shape[0] != n:
        raise ValueError(f"fit_gbdt_cv: len(y)={y.shape[0]} != len(X)={n}")
    if sample_weight.shape[0] != n:
        raise ValueError(
            f"fit_gbdt_cv: len(sample_weight)={sample_weight.shape[0]} != len(X)={n}"
        )
    if not splits:
        raise ValueError("fit_gbdt_cv: splits is empty")
    uniq = set(np.unique(y).tolist()) if y.size else set()
    if not uniq <= {0, 1}:
        raise ValueError(f"fit_gbdt_cv: y must be binary 0/1, got values {sorted(uniq)}")
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        for name, idx in (("train", train_idx), ("test", test_idx)):
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
                raise ValueError(
                    f"fit_gbdt_cv: fold {fold_id} {name} index out of range [0, {n})"
                )


def _fit_one_fold(
    X_values: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    fold_id: int,
    fold: Fold,
    cfg: GBDTConfig,
) -> tuple[FoldResult, NDArray[np.int64], NDArray[np.int64]]:
    """Train on one fold's train block and score its test block out-of-sample.

    Args:
        X_values: Dense feature matrix as a ``float64`` array.
        y: Binary labels aligned to ``X_values``.
        sample_weight: Per-row weights aligned to ``X_values``.
        fold_id: 0-based fold index.
        fold: ``(train_idx, test_idx)`` original-row indices.
        cfg: Baseline configuration.

    Returns:
        ``(result, test_pred_idx, test_true_idx)`` — the fold's scores plus the
        predicted and true OOS bar indices (for pooled aggregation).
    """
    train_idx, test_idx = fold
    model = lgb.LGBMClassifier(**cfg.lgb_params())
    model.fit(
        X_values[train_idx],
        y[train_idx],
        sample_weight=sample_weight[train_idx],
    )

    # Class-1 probabilities on train (for thresholding) and test (for scoring).
    train_proba = _predict_proba(model, X_values[train_idx])
    test_proba = _predict_proba(model, X_values[test_idx])

    # Guard the classic sparse-label trap: if every training row carries the
    # positive class (or negatives were given zero sample weight) LightGBM emits a
    # near-constant probability and thresholding degenerates to "flag everything".
    # Warn loudly rather than silently report a meaningless score.
    if train_proba.size and float(np.ptp(train_proba)) < 1e-9:
        logger.warning(
            "fold %d: train probabilities are near-constant (ptp=%.2e); the model "
            "is not discriminating — check that negatives carry non-zero weight",
            fold_id,
            float(np.ptp(train_proba)),
        )

    train_true_idx = np.ascontiguousarray(
        train_idx[y[train_idx] == 1], dtype=np.int64
    )
    test_true_idx = np.ascontiguousarray(test_idx[y[test_idx] == 1], dtype=np.int64)

    threshold, train_f1 = _tune_threshold(train_proba, train_idx, train_true_idx, cfg)

    test_pred_idx = _positive_indices(test_proba, test_idx, threshold)
    precision, recall, f1 = event_prf(test_pred_idx, test_true_idx, cfg.tolerance)

    logger.info(
        "fold %d: thr=%.4f train_f1=%.3f | OOS p=%.3f r=%.3f f1=%.3f "
        "(n_true=%d n_pred=%d, train=%d test=%d)",
        fold_id,
        threshold,
        train_f1,
        precision,
        recall,
        f1,
        test_true_idx.size,
        test_pred_idx.size,
        train_idx.size,
        test_idx.size,
    )

    result = FoldResult(
        fold=fold_id,
        threshold=threshold,
        train_f1=train_f1,
        precision=precision,
        recall=recall,
        f1=f1,
        n_true=int(test_true_idx.size),
        n_pred=int(test_pred_idx.size),
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
    )
    return result, test_pred_idx, test_true_idx


# --------------------------------------------------------------------------- #
# Public API.                                                                 #
# --------------------------------------------------------------------------- #


def fit_gbdt_cv(
    X: pd.DataFrame,
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
    splits: list[Fold],
    cfg: GBDTConfig,
) -> dict[str, object]:
    """Fit a LightGBM turn detector per fold and report out-of-sample event PRF.

    For every walk-forward fold the model is trained on the train block (with the
    oracle sample weights), the decision threshold is tuned to maximise event-F1
    **on that train block only**, and the frozen threshold scores the held-out
    test block with :func:`cfd10.eval_module.metrics.event_prf`. The per-fold OOS
    predicted / true bars are then pooled (test blocks are disjoint in the original
    index space) and scored once to give the aggregated event PRF — the headline
    OOS number — alongside the mean per-fold F1.

    Args:
        X: Dense feature matrix (warm-up NaN rows must already be dropped); rows
            are addressed by the original integer positions used in ``splits``.
        y: Binary label array (``1`` = oracle turn) aligned to ``X``.
        sample_weight: Per-row LightGBM weight (the oracle ``*_weight`` column)
            aligned to ``X``.
        splits: Purged walk-forward folds from
            :func:`cfd10.cv_module.purged_walk_forward`; indices address rows of
            ``X``.
        cfg: Baseline configuration.

    Returns:
        A dict with:
            ``folds``: list of :class:`FoldResult`;
            ``oos_precision`` / ``oos_recall`` / ``oos_f1``: pooled OOS event PRF;
            ``oos_tp`` / ``oos_fp`` / ``oos_fn``: pooled OOS confusion counts;
            ``mean_fold_f1``: mean of the per-fold OOS F1;
            ``n_true`` / ``n_pred``: pooled OOS true / predicted event counts;
            ``positive_rate``: overall label positive rate;
            ``config``: the resolved ``cfg`` as a plain dict.

    Raises:
        ValueError: If array lengths disagree, ``splits`` is empty, ``y`` is not
            binary, or a fold index is out of range.
    """
    set_seed(cfg.seed)
    y = np.ascontiguousarray(y, dtype=np.int64)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    _validate_inputs(X, y, sample_weight, splits)

    X_values = np.ascontiguousarray(X.to_numpy(dtype=np.float64))

    fold_results: list[FoldResult] = []
    pred_blocks: list[NDArray[np.int64]] = []
    true_blocks: list[NDArray[np.int64]] = []
    for fold_id, fold in enumerate(splits):
        result, test_pred_idx, test_true_idx = _fit_one_fold(
            X_values, y, sample_weight, fold_id, fold, cfg
        )
        fold_results.append(result)
        pred_blocks.append(test_pred_idx)
        true_blocks.append(test_true_idx)

    all_pred = (
        np.concatenate(pred_blocks) if pred_blocks else np.empty(0, dtype=np.int64)
    )
    all_true = (
        np.concatenate(true_blocks) if true_blocks else np.empty(0, dtype=np.int64)
    )
    # Pooled OOS event PRF. Test blocks are disjoint so a single global match over
    # the concatenated indices is exact (no cross-fold double counting).
    tp, fp, fn, _ = match_events(all_pred, all_true, cfg.tolerance)
    oos_p, oos_r, oos_f1 = event_prf(all_pred, all_true, cfg.tolerance)

    fold_f1s = np.array([r.f1 for r in fold_results], dtype=np.float64)
    mean_fold_f1 = float(fold_f1s.mean()) if fold_f1s.size else 0.0
    positive_rate = float(y.mean()) if y.size else 0.0

    logger.info(
        "fit_gbdt_cv: pooled OOS p=%.4f r=%.4f f1=%.4f (tp=%d fp=%d fn=%d), "
        "mean_fold_f1=%.4f, pos_rate=%.4f",
        oos_p,
        oos_r,
        oos_f1,
        tp,
        fp,
        fn,
        mean_fold_f1,
        positive_rate,
    )

    return {
        "folds": fold_results,
        "oos_precision": oos_p,
        "oos_recall": oos_r,
        "oos_f1": oos_f1,
        "oos_tp": int(tp),
        "oos_fp": int(fp),
        "oos_fn": int(fn),
        "mean_fold_f1": mean_fold_f1,
        "n_true": int(all_true.size),
        "n_pred": int(all_pred.size),
        "positive_rate": positive_rate,
        "config": asdict(cfg),
    }
