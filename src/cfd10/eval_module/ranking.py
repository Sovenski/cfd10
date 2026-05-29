"""Threshold-free, n-weighted ranking metrics for the pivot-oracle detector.

The event precision / recall / F1 in :mod:`cfd10.eval_module.metrics` collapse to
zero the moment the detector flags nothing, which hides a model that *ranks* bars
well but is simply mis-thresholded. Under the weighted multi-scale pivot oracle the
positive class is rare and wildly heterogeneous -- a 200-scale pivot matters far
more than a 2-scale one -- so a single F1 number is doubly misleading. These metrics
sidestep the threshold entirely and ask a sharper, n-weighted question: does a
higher score concentrate the true pivots, and the *heavy* pivots most of all?

There are no forward-return or PnL metrics here, by design: the label is the
oracle's structural pivot tier, and these scores judge ranking against that label
only.

Metric families
---------------
Ranking quality
    :func:`average_precision` -- area under the precision-recall curve
    (scikit-learn's ``average_precision_score``). This is the headline,
    threshold-free score: ``1.0`` iff every positive outranks every negative. Its
    optional ``sample_weight`` carries the oracle n-score, so nailing a heavy pivot
    counts for more than a light one.
Top-of-book hit rate
    :func:`precision_at_k` / :func:`recall_at_k` -- of the ``k`` highest-scoring
    bars, the fraction that are true positives, and the fraction of all true
    positives those picks recover (exact bar match). :func:`lift_at_k` divides
    precision-at-k by the base positive rate, so ``1.0`` means "no better than
    chance" and ``> 1`` means the ranking concentrates positives.
Event-tolerant hit rate
    :func:`precision_at_k_event` / :func:`recall_at_k_event` -- relax the hit test
    to a ``tolerance`` window around any true pivot, reusing the one-to-one
    :func:`cfd10.eval_module.events.match_events` matcher (no true pivot is claimed
    twice), with a ``bar_index`` mapping each scored row to its bar position.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from cfd10.eval_module.events import match_events
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "average_precision",
    "precision_at_k",
    "recall_at_k",
    "lift_at_k",
    "precision_at_k_event",
    "recall_at_k_event",
]


def _as_float_1d(arr: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return ``arr`` as a contiguous 1-D ``float64`` array.

    Args:
        arr: Candidate numeric array.
        name: Argument name, used only for error messages.

    Returns:
        A C-contiguous ``float64`` copy of ``arr``.

    Raises:
        ValueError: If ``arr`` is not one-dimensional.
    """
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {out.shape}")
    return out


def _as_int_1d(arr: NDArray[np.int64], name: str) -> NDArray[np.int64]:
    """Return ``arr`` as a contiguous 1-D ``int64`` array.

    Args:
        arr: Candidate index / label array.
        name: Argument name, used only for error messages.

    Returns:
        A C-contiguous ``int64`` copy of ``arr``.

    Raises:
        ValueError: If ``arr`` is not one-dimensional.
    """
    out = np.ascontiguousarray(arr, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {out.shape}")
    return out


def _check_binary(y: NDArray[np.int64]) -> None:
    """Raise if ``y`` holds any value other than ``0`` / ``1``.

    Args:
        y: 1-D integer label array.

    Raises:
        ValueError: If ``y`` contains a value outside ``{0, 1}``.
    """
    unique = np.unique(y)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"y_true must be binary 0/1, got values {unique.tolist()}")


def _top_k_indices(scores: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Return the positions of the ``k`` highest scores, ties broken by position.

    Sorting is deterministic: scores are ranked descending and ties are resolved by
    ascending original position (a stable sort on the negated scores), so the same
    inputs always select the same ``k`` rows.

    Args:
        scores: 1-D array of scores.
        k: Number of top rows to select (already validated to ``1 <= k <= len``).

    Returns:
        A 1-D ``int64`` array of the selected row positions, in descending-score
        order.
    """
    order = np.argsort(-scores, kind="stable")
    return order[:k].astype(np.int64)


def average_precision(
    scores: NDArray[np.float64],
    y_true: NDArray[np.int64],
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    """Area under the precision-recall curve (a.k.a. average precision).

    A threshold-free summary of ranking quality: it equals ``1.0`` exactly when
    every positive bar receives a strictly higher score than every negative bar, and
    degrades smoothly as positives mix into the lower ranks. This is the headline
    metric for the pivot detector because it cannot be hidden by an F1 collapse from
    a badly-placed decision threshold.

    The optional ``sample_weight`` is the lever for n-weighting: pass the oracle
    pivot score (e.g. ``top_score`` / ``bottom_score``, or the
    ``w = 1 + oracle_score`` sample weight) so that correctly ranking a heavy
    (large-scale) pivot above the negatives lifts the score more than ranking a light
    pivot. With uniform weights this reduces to the ordinary average precision.

    Thin wrapper over :func:`sklearn.metrics.average_precision_score`.

    Args:
        scores: 1-D array of per-bar scores; larger means "more likely a pivot".
        y_true: 1-D array of binary labels (``1`` = true pivot, ``0`` = not) aligned
            with ``scores``.
        sample_weight: Optional 1-D array of per-bar weights aligned with ``scores``;
            typically the oracle n-score. ``None`` means uniform weights.

    Returns:
        The (optionally weighted) average precision as a plain ``float`` in
        ``[0, 1]``. Returns ``0.0`` for the degenerate case of no positive labels (no
        PR curve is defined).

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``y_true`` is not
            binary, or any weight is negative.
    """
    s = _as_float_1d(scores, "scores")
    y = _as_int_1d(y_true, "y_true")
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"scores and y_true must align, got {s.shape[0]} vs {y.shape[0]}"
        )
    _check_binary(y)

    w: NDArray[np.float64] | None = None
    if sample_weight is not None:
        w = _as_float_1d(sample_weight, "sample_weight")
        if w.shape[0] != y.shape[0]:
            raise ValueError(
                f"sample_weight and y_true must align, got {w.shape[0]} vs {y.shape[0]}"
            )
        if np.any(w < 0.0):
            raise ValueError("sample_weight must be non-negative")

    n_pos = int(y.sum())
    if n_pos == 0:
        logger.warning("average_precision: no positive labels; returning 0.0")
        return 0.0

    ap = float(average_precision_score(y, s, sample_weight=w))
    logger.debug(
        "average_precision: ap=%.4f (n=%d, n_pos=%d, weighted=%s)",
        ap,
        y.shape[0],
        n_pos,
        w is not None,
    )
    return ap


def precision_at_k(
    scores: NDArray[np.float64],
    y_true: NDArray[np.int64],
    k: int,
) -> float:
    """Fraction of the ``k`` highest-scoring bars that are true positives.

    The exact-match counterpart to :func:`precision_at_k_event`: a selected bar
    counts only if its own label is ``1``. Useful when only the very top of the
    ranked list will ever be acted on.

    Args:
        scores: 1-D array of per-bar scores; larger means "more likely a pivot".
        y_true: 1-D array of binary labels (``1``/``0``) aligned with ``scores``.
        k: Number of top-scoring bars to inspect. Must satisfy
            ``1 <= k <= len(scores)``.

    Returns:
        ``hits / k`` as a plain ``float`` in ``[0, 1]``, where ``hits`` is the number
        of selected bars whose label is ``1``.

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``y_true`` is not
            binary, or ``k`` is out of range.
    """
    s = _as_float_1d(scores, "scores")
    y = _as_int_1d(y_true, "y_true")
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"scores and y_true must align, got {s.shape[0]} vs {y.shape[0]}"
        )
    _check_binary(y)
    if not 1 <= k <= s.shape[0]:
        raise ValueError(f"k must be in [1, {s.shape[0]}], got {k}")

    top = _top_k_indices(s, k)
    hits = int(y[top].sum())
    precision = hits / k
    logger.debug("precision_at_k: %d/%d = %.4f", hits, k, precision)
    return float(precision)


def recall_at_k(
    scores: NDArray[np.float64],
    y_true: NDArray[np.int64],
    k: int,
) -> float:
    """Fraction of all true positives captured by the ``k`` highest-scoring bars.

    The recall counterpart to :func:`precision_at_k`: ``hits / n_pos``, where
    ``n_pos`` is the total number of positive labels. Answers "if we only act on the
    top ``k`` bars, how many of the real pivots do we catch?".

    Args:
        scores: 1-D array of per-bar scores; larger means "more likely a pivot".
        y_true: 1-D array of binary labels (``1``/``0``) aligned with ``scores``.
        k: Number of top-scoring bars to inspect. Must satisfy
            ``1 <= k <= len(scores)``.

    Returns:
        ``hits / n_pos`` as a plain ``float`` in ``[0, 1]``. Returns ``0.0`` when
        there are no positive labels (recall is undefined).

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``y_true`` is not
            binary, or ``k`` is out of range.
    """
    s = _as_float_1d(scores, "scores")
    y = _as_int_1d(y_true, "y_true")
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"scores and y_true must align, got {s.shape[0]} vs {y.shape[0]}"
        )
    _check_binary(y)
    if not 1 <= k <= s.shape[0]:
        raise ValueError(f"k must be in [1, {s.shape[0]}], got {k}")

    n_pos = int(y.sum())
    if n_pos == 0:
        logger.warning("recall_at_k: no positive labels; returning 0.0")
        return 0.0

    top = _top_k_indices(s, k)
    hits = int(y[top].sum())
    recall = hits / n_pos
    logger.debug("recall_at_k: %d/%d = %.4f", hits, n_pos, recall)
    return float(recall)


def lift_at_k(
    scores: NDArray[np.float64],
    y_true: NDArray[np.int64],
    k: int,
) -> float:
    """Precision-at-k relative to the base positive rate (``precision / base_rate``).

    Lift normalises :func:`precision_at_k` by the prevalence of positives in the
    whole sample (``base_rate = n_pos / n``): ``1.0`` means the top ``k`` bars are no
    richer in pivots than a random draw, and ``> 1`` means the ranking concentrates
    them. The theoretical maximum is ``1 / base_rate`` (a perfectly pure top ``k``).

    Args:
        scores: 1-D array of per-bar scores; larger means "more likely a pivot".
        y_true: 1-D array of binary labels (``1``/``0``) aligned with ``scores``.
        k: Number of top-scoring bars to inspect. Must satisfy
            ``1 <= k <= len(scores)``.

    Returns:
        ``precision_at_k / base_rate`` as a plain ``float`` in ``[0, 1 / base_rate]``.
        Returns ``0.0`` when there are no positive labels (base rate is zero).

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``y_true`` is not
            binary, or ``k`` is out of range.
    """
    # precision_at_k re-validates; we only need n_pos and n here for the base rate.
    s = _as_float_1d(scores, "scores")
    y = _as_int_1d(y_true, "y_true")
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"scores and y_true must align, got {s.shape[0]} vs {y.shape[0]}"
        )
    _check_binary(y)

    n = y.shape[0]
    n_pos = int(y.sum())
    if n_pos == 0:
        logger.warning("lift_at_k: no positive labels; returning 0.0")
        return 0.0

    base_rate = n_pos / n
    precision = precision_at_k(s, y, k)
    lift = precision / base_rate
    logger.debug(
        "lift_at_k: precision=%.4f base_rate=%.4f lift=%.4f", precision, base_rate, lift
    )
    return float(lift)


def _event_hits_at_k(
    pred_scores: NDArray[np.float64],
    true_idx: NDArray[np.int64],
    k: int,
    tolerance: int,
    bar_index: NDArray[np.int64],
) -> tuple[int, int]:
    """Shared core for the event-tolerant top-``k`` metrics.

    Selects the ``k`` highest-scoring rows, maps them to bar positions via
    ``bar_index`` and counts tolerance-window hits against ``true_idx`` with the
    one-to-one :func:`cfd10.eval_module.events.match_events` matcher.

    Args:
        pred_scores: 1-D array of per-row scores; larger means "more likely a pivot".
        true_idx: 1-D array of true pivot bar-indices.
        k: Number of top-scoring rows to select. Must satisfy
            ``1 <= k <= len(pred_scores)``.
        tolerance: Maximum absolute bar-distance for a hit (``>= 0``).
        bar_index: 1-D array mapping each scored row to its bar position, aligned
            with ``pred_scores``.

    Returns:
        ``(tp, n_true)`` where ``tp`` is the number of selected rows matched to a
        distinct true pivot within ``tolerance`` and ``n_true`` is the number of true
        pivots.

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``k`` is out of
            range, or ``tolerance`` is negative.
    """
    s = _as_float_1d(pred_scores, "pred_scores")
    true = _as_int_1d(true_idx, "true_idx")
    bars = _as_int_1d(bar_index, "bar_index")
    if bars.shape[0] != s.shape[0]:
        raise ValueError(
            f"bar_index and pred_scores must align, got {bars.shape[0]} vs {s.shape[0]}"
        )
    if not 1 <= k <= s.shape[0]:
        raise ValueError(f"k must be in [1, {s.shape[0]}], got {k}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")

    top_rows = _top_k_indices(s, k)
    pred_bars = bars[top_rows]
    tp, _fp, _fn, _matches = match_events(pred_bars, true, tolerance)
    return tp, true.shape[0]


def precision_at_k_event(
    pred_scores: NDArray[np.float64],
    true_idx: NDArray[np.int64],
    k: int,
    tolerance: int,
    bar_index: NDArray[np.int64],
) -> float:
    """Top-``k`` precision with a tolerance window around each true pivot.

    Selects the ``k`` highest-scoring rows, maps them to bar positions through
    ``bar_index``, and counts a pick as a hit if it lies within ``tolerance`` bars of
    a *distinct* true pivot, reusing the one-to-one matcher
    :func:`cfd10.eval_module.events.match_events` (no true pivot is claimed twice).
    The score is ``tp / k``, so unmatched picks still dilute precision.

    The ``bar_index`` indirection lets the scored rows be a subset or re-ordering of
    the raw bars (e.g. a pooled, multi-asset evaluation frame): row ``i`` carries
    score ``pred_scores[i]`` and sits at bar ``bar_index[i]``.

    Args:
        pred_scores: 1-D array of per-row scores; larger means "more likely a pivot".
        true_idx: 1-D array of true pivot bar-indices.
        k: Number of top-scoring rows to select. Must satisfy
            ``1 <= k <= len(pred_scores)``.
        tolerance: Maximum absolute bar-distance for a hit (``>= 0``).
        bar_index: 1-D array mapping each scored row to its bar position, aligned
            with ``pred_scores``.

    Returns:
        ``tp / k`` as a plain ``float`` in ``[0, 1]``, where ``tp`` is the number of
        selected rows matched to a true pivot within ``tolerance``.

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``k`` is out of
            range, or ``tolerance`` is negative.
    """
    tp, _n_true = _event_hits_at_k(pred_scores, true_idx, k, tolerance, bar_index)
    precision = tp / k
    logger.debug(
        "precision_at_k_event: tp=%d/%d = %.4f (tol=%d)", tp, k, precision, tolerance
    )
    return float(precision)


def recall_at_k_event(
    pred_scores: NDArray[np.float64],
    true_idx: NDArray[np.int64],
    k: int,
    tolerance: int,
    bar_index: NDArray[np.int64],
) -> float:
    """Top-``k`` recall with a tolerance window around each true pivot.

    The recall counterpart to :func:`precision_at_k_event`: of all true pivots, the
    fraction matched by the ``k`` highest-scoring rows within ``tolerance`` bars
    (``tp / n_true``), using the same one-to-one matcher so no true pivot is claimed
    twice.

    Args:
        pred_scores: 1-D array of per-row scores; larger means "more likely a pivot".
        true_idx: 1-D array of true pivot bar-indices.
        k: Number of top-scoring rows to select. Must satisfy
            ``1 <= k <= len(pred_scores)``.
        tolerance: Maximum absolute bar-distance for a hit (``>= 0``).
        bar_index: 1-D array mapping each scored row to its bar position, aligned
            with ``pred_scores``.

    Returns:
        ``tp / n_true`` as a plain ``float`` in ``[0, 1]``. Returns ``0.0`` when
        there are no true pivots (recall is undefined).

    Raises:
        ValueError: If an input is not 1-D, the lengths differ, ``k`` is out of
            range, or ``tolerance`` is negative.
    """
    tp, n_true = _event_hits_at_k(pred_scores, true_idx, k, tolerance, bar_index)
    if n_true == 0:
        logger.warning("recall_at_k_event: no true pivots; returning 0.0")
        return 0.0
    recall = tp / n_true
    logger.debug(
        "recall_at_k_event: tp=%d/%d = %.4f (tol=%d)", tp, n_true, recall, tolerance
    )
    return float(recall)
