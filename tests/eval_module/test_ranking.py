"""Tests for ``cfd10.eval_module.ranking`` -- threshold-free, n-weighted metrics.

The point of these metrics is to survive an F1 collapse on the rare pivot label, so
the tests pin the threshold-free guarantees: average precision is exactly ``1.0``
under a perfect ranking and *rises* once the oracle ``sample_weight`` emphasises a
correctly-ranked heavy pivot; top-k precision / recall / lift match hand-worked
cases; and the event variants honour the tolerance window (with a non-trivial
``bar_index`` mapping) without double-counting a true pivot.

No forward-return or PnL assertions appear here -- the label is the structural pivot
tier and these scores judge ranking against that label alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfd10.eval_module.ranking import (
    average_precision,
    lift_at_k,
    precision_at_k,
    precision_at_k_event,
    recall_at_k,
    recall_at_k_event,
)


# --------------------------------------------------------------------------- #
# average_precision
# --------------------------------------------------------------------------- #
def test_average_precision_perfect_ranking_is_one() -> None:
    """AP == 1.0 when every positive outranks every negative.

    Labels [0, 1, 0, 1, 0] with scores that place both positives strictly above
    every negative -> a perfect precision-recall curve.
    """
    y_true = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    scores = np.array([0.10, 0.90, 0.20, 0.80, 0.05], dtype=np.float64)
    ap = average_precision(scores, y_true)
    assert math.isclose(ap, 1.0, abs_tol=1e-12)


def test_average_precision_worst_ranking_below_one() -> None:
    """A ranking that buries the only positive last scores well under 1.0."""
    y_true = np.array([1, 0, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)  # positive ranked last
    ap = average_precision(scores, y_true)
    # Single positive recovered only at rank 4 -> precision 1/4.
    assert math.isclose(ap, 0.25, abs_tol=1e-12)
    assert ap < 1.0


def test_average_precision_weight_emphasising_heavy_positive_rises() -> None:
    """n-weighting a correctly-ranked heavy pivot raises AP above the flat score.

    Ranking (by descending score):
        bar 0  positive, *heavy*  -> ranked top   (good)
        bar 1  negative
        bar 2  negative
        bar 3  positive, *light*  -> ranked last  (bad)

    Unweighted AP is 0.75 (the buried light positive drags it down). Putting the
    oracle's large n-score on the well-ranked heavy positive makes that correct
    decision dominate the average, so the weighted AP must exceed the flat one.
    """
    y_true = np.array([1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float64)

    ap_flat = average_precision(scores, y_true)
    assert math.isclose(ap_flat, 0.75, abs_tol=1e-12)

    # Heavy weight on the correctly-ranked positive (oracle n-score convention:
    # w = 1 + score, with the large-scale pivot carrying the big score).
    weight = np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float64)
    ap_weighted = average_precision(scores, y_true, sample_weight=weight)
    assert ap_weighted > ap_flat
    assert ap_weighted < 1.0  # the light positive is still buried


def test_average_precision_uniform_weight_matches_unweighted() -> None:
    """A constant sample_weight reproduces the unweighted average precision."""
    y_true = np.array([1, 0, 1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.8, 0.6, 0.55, 0.4, 0.2, 0.1], dtype=np.float64)
    flat = average_precision(scores, y_true)
    weighted = average_precision(
        scores, y_true, sample_weight=np.full(6, 3.0, dtype=np.float64)
    )
    assert math.isclose(flat, weighted, abs_tol=1e-12)


def test_average_precision_no_positives_returns_zero() -> None:
    """No positive labels -> undefined PR curve, reported as 0.0."""
    y_true = np.zeros(5, dtype=np.int64)
    scores = np.arange(5, dtype=np.float64)
    assert average_precision(scores, y_true) == 0.0


def test_average_precision_validation() -> None:
    """Length mismatch, non-binary labels and bad weights raise ValueError."""
    with pytest.raises(ValueError):
        average_precision(np.array([0.1, 0.2]), np.array([1], dtype=np.int64))
    with pytest.raises(ValueError):
        average_precision(
            np.array([0.1, 0.2, 0.3]), np.array([0, 2, 1], dtype=np.int64)
        )
    # sample_weight length mismatch.
    with pytest.raises(ValueError):
        average_precision(
            np.array([0.1, 0.2, 0.3]),
            np.array([0, 1, 0], dtype=np.int64),
            sample_weight=np.array([1.0, 2.0]),
        )
    # Negative weight.
    with pytest.raises(ValueError):
        average_precision(
            np.array([0.1, 0.2, 0.3]),
            np.array([0, 1, 0], dtype=np.int64),
            sample_weight=np.array([1.0, -1.0, 1.0]),
        )


# --------------------------------------------------------------------------- #
# precision_at_k / recall_at_k / lift_at_k
# --------------------------------------------------------------------------- #
def test_precision_at_k_known_case() -> None:
    """Top-3 of a known ranking contains 2 of 3 true positives.

    scores  : [0.9, 0.8, 0.7, 0.2, 0.1]   ranked bars -> 0, 1, 2, 3, 4
    y_true  : [ 1 ,  0 ,  1 ,  1 ,  0 ]
    top-3 bars are {0, 1, 2} with labels {1, 0, 1} -> 2 hits / 3 = 2/3.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)
    y_true = np.array([1, 0, 1, 1, 0], dtype=np.int64)
    assert math.isclose(precision_at_k(scores, y_true, k=3), 2.0 / 3.0, abs_tol=1e-12)
    # k == 1 selects the single best-scoring bar, which is a true positive.
    assert precision_at_k(scores, y_true, k=1) == 1.0
    # k == len selects everything -> overall positive rate 3/5.
    assert math.isclose(precision_at_k(scores, y_true, k=5), 3.0 / 5.0, abs_tol=1e-12)


def test_recall_at_k_known_case() -> None:
    """Recall at k counts captured positives over the 3 true positives.

    Same ranking as above: top-3 catches 2 of the 3 positives -> 2/3; k == 5 (all
    bars) recovers every positive -> 1.0; k == 1 catches only 1 of 3 -> 1/3.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)
    y_true = np.array([1, 0, 1, 1, 0], dtype=np.int64)
    assert math.isclose(recall_at_k(scores, y_true, k=3), 2.0 / 3.0, abs_tol=1e-12)
    assert recall_at_k(scores, y_true, k=5) == 1.0
    assert math.isclose(recall_at_k(scores, y_true, k=1), 1.0 / 3.0, abs_tol=1e-12)


def test_recall_at_k_no_positives_returns_zero() -> None:
    """No positive labels -> recall undefined, reported as 0.0."""
    scores = np.arange(4, dtype=np.float64)
    y_true = np.zeros(4, dtype=np.int64)
    assert recall_at_k(scores, y_true, k=2) == 0.0


def test_lift_at_k_concentrates_positives() -> None:
    """Lift = precision_at_k / base_rate; > 1 when the top-k is enriched.

    base_rate = 3/5 = 0.6. Top-1 is a pure positive -> precision 1.0 ->
    lift = 1.0 / 0.6 = 5/3 (the theoretical max 1/base_rate). At k == 5 the top-k is
    the whole sample, so precision == base_rate and lift == 1.0 exactly.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)
    y_true = np.array([1, 0, 1, 1, 0], dtype=np.int64)
    assert math.isclose(lift_at_k(scores, y_true, k=1), 5.0 / 3.0, abs_tol=1e-12)
    assert math.isclose(lift_at_k(scores, y_true, k=5), 1.0, abs_tol=1e-12)


def test_lift_at_k_no_positives_returns_zero() -> None:
    """No positive labels -> base rate zero, reported as 0.0."""
    scores = np.arange(4, dtype=np.float64)
    y_true = np.zeros(4, dtype=np.int64)
    assert lift_at_k(scores, y_true, k=2) == 0.0


def test_precision_at_k_tie_break_is_deterministic() -> None:
    """Ties are broken by ascending index, so selection is reproducible."""
    scores = np.array([0.5, 0.5, 0.5, 0.1], dtype=np.float64)
    y_true = np.array([1, 0, 1, 0], dtype=np.int64)
    # Stable order picks bars 0 and 1 -> labels {1, 0} -> 1/2.
    assert math.isclose(precision_at_k(scores, y_true, k=2), 0.5, abs_tol=1e-12)


def test_at_k_out_of_range() -> None:
    """k outside [1, len] raises ValueError across the top-k family."""
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    y_true = np.array([0, 1, 0], dtype=np.int64)
    for fn in (precision_at_k, recall_at_k, lift_at_k):
        with pytest.raises(ValueError):
            fn(scores, y_true, k=0)
        with pytest.raises(ValueError):
            fn(scores, y_true, k=4)


# --------------------------------------------------------------------------- #
# precision_at_k_event / recall_at_k_event
# --------------------------------------------------------------------------- #
def test_precision_at_k_event_tolerance_window() -> None:
    """A near-miss within tolerance counts; a far pick does not.

    Four scored rows whose bar positions are a re-ordered, non-contiguous subset of
    the price series via ``bar_index``:
        row    : 0     1     2     3
        score  : 0.9   0.8   0.7   0.1
        bar    : 20    50    99    10
    The top-3 rows map to bars {20, 50, 99}; true pivots sit at {21, 50}.
        bar 20 -> within 1 of true 21   hit
        bar 50 -> exact true 50         hit
        bar 99 -> far from any true     miss
    So tp = 2, precision = 2 / 3, recall = 2 / 2 = 1.0.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
    bar_index = np.array([20, 50, 99, 10], dtype=np.int64)
    true_idx = np.array([21, 50], dtype=np.int64)

    p = precision_at_k_event(scores, true_idx, k=3, tolerance=1, bar_index=bar_index)
    assert math.isclose(p, 2.0 / 3.0, abs_tol=1e-12)
    r = recall_at_k_event(scores, true_idx, k=3, tolerance=1, bar_index=bar_index)
    assert math.isclose(r, 1.0, abs_tol=1e-12)

    # Tolerance 0 credits only the exact hit at bar 50 -> precision 1/3, recall 1/2.
    p0 = precision_at_k_event(scores, true_idx, k=3, tolerance=0, bar_index=bar_index)
    assert math.isclose(p0, 1.0 / 3.0, abs_tol=1e-12)
    r0 = recall_at_k_event(scores, true_idx, k=3, tolerance=0, bar_index=bar_index)
    assert math.isclose(r0, 1.0 / 2.0, abs_tol=1e-12)


def test_precision_at_k_event_no_double_count() -> None:
    """Two top picks near one lone true pivot yield only a single hit.

    Bars 4 and 5 both fall within tolerance 1 of the single true pivot at 4, but the
    one-to-one matcher credits only one of them -> tp = 1, precision 1/3.
    """
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    bar_index = np.array([4, 5, 0], dtype=np.int64)
    true_idx = np.array([4], dtype=np.int64)
    p = precision_at_k_event(scores, true_idx, k=3, tolerance=1, bar_index=bar_index)
    assert math.isclose(p, 1.0 / 3.0, abs_tol=1e-12)
    # Only one true pivot exists and it is matched -> recall 1.0.
    r = recall_at_k_event(scores, true_idx, k=3, tolerance=1, bar_index=bar_index)
    assert math.isclose(r, 1.0, abs_tol=1e-12)


def test_recall_at_k_event_no_true_pivots_returns_zero() -> None:
    """No true pivots -> recall undefined, reported as 0.0."""
    scores = np.array([0.9, 0.1], dtype=np.float64)
    bar_index = np.array([0, 1], dtype=np.int64)
    true_idx = np.array([], dtype=np.int64)
    assert (
        recall_at_k_event(scores, true_idx, k=1, tolerance=1, bar_index=bar_index)
        == 0.0
    )


def test_precision_at_k_event_validation() -> None:
    """Bad k, negative tolerance and a misaligned bar_index raise ValueError."""
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    bar_index = np.array([0, 1, 2], dtype=np.int64)
    true_idx = np.array([1], dtype=np.int64)
    with pytest.raises(ValueError):
        precision_at_k_event(scores, true_idx, k=5, tolerance=1, bar_index=bar_index)
    with pytest.raises(ValueError):
        precision_at_k_event(scores, true_idx, k=2, tolerance=-1, bar_index=bar_index)
    # bar_index must align with the scores.
    with pytest.raises(ValueError):
        precision_at_k_event(
            scores, true_idx, k=2, tolerance=1, bar_index=np.array([0, 1], dtype=np.int64)
        )
