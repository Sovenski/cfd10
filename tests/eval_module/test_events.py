"""Tests for ``cfd10.eval_module.events`` — tolerance-aware event matching.

These pin the bipartite-matching contract of :func:`match_events`:

* a prediction within ``tolerance`` bars of a true event is a true positive;
* a prediction beyond ``tolerance`` of every free true event is a false positive
  and leaves that true event a false negative;
* no true bar (and no predicted bar) is ever matched twice;
* the counts respect ``tp + fp == n_pred`` and ``tp + fn == n_true``.
"""

from __future__ import annotations

import numpy as np

from cfd10.eval_module.events import match_events


def test_exact_alignment_all_true_positive() -> None:
    """Identical predictions and truths give all TP, no FP/FN (delta 0)."""
    idx = np.array([10, 25, 40, 77], dtype=np.int64)
    tp, fp, fn, matches = match_events(idx, idx, tolerance=2)
    assert (tp, fp, fn) == (4, 0, 0)
    assert [m[2] for m in matches] == [0, 0, 0, 0]
    # Matches are returned sorted by the true bar.
    assert [m[1] for m in matches] == [10, 25, 40, 77]


def test_shift_within_tolerance_counts_as_tp() -> None:
    """A prediction shifted by exactly ``tolerance`` still matches (<= boundary)."""
    true = np.array([10, 20, 30], dtype=np.int64)
    pred = np.array([12, 18, 33], dtype=np.int64)  # deltas 2, 2, 3
    tp, fp, fn, matches = match_events(pred, true, tolerance=3)
    assert (tp, fp, fn) == (3, 0, 0)
    deltas = {m[1]: m[2] for m in matches}
    assert deltas == {10: 2, 20: 2, 30: 3}


def test_shift_beyond_tolerance_is_fp_plus_fn() -> None:
    """A prediction farther than tolerance from every truth is FP and leaves FN.

    One true/pred pair is aligned (delta 0); the other prediction is 5 bars away
    from its nearest truth with tolerance 2, so it cannot match: it becomes a
    false positive and its intended truth becomes a false negative.
    """
    true = np.array([10, 50], dtype=np.int64)
    pred = np.array([10, 55], dtype=np.int64)  # second delta is 5 > tol
    tp, fp, fn, matches = match_events(pred, true, tolerance=2)
    assert (tp, fp, fn) == (1, 1, 1)
    assert matches == [(10, 10, 0)]


def test_boundary_just_outside_tolerance_rejected() -> None:
    """A delta of ``tolerance + 1`` is rejected (strict ``<=`` gate)."""
    true = np.array([100], dtype=np.int64)
    pred = np.array([104], dtype=np.int64)  # delta 4
    tp, fp, fn, matches = match_events(pred, true, tolerance=3)
    assert (tp, fp, fn) == (0, 1, 1)
    assert matches == []


def test_no_double_match_two_preds_one_truth() -> None:
    """Two predictions near one truth: exactly one matches, the other is FP.

    Both predictions (9 and 11) are within tolerance of the single truth (10),
    but a true bar may be claimed only once, so tp == 1 and the surplus
    prediction is a false positive. No false negatives (the truth is covered).
    """
    true = np.array([10], dtype=np.int64)
    pred = np.array([9, 11], dtype=np.int64)
    tp, fp, fn, matches = match_events(pred, true, tolerance=2)
    assert (tp, fp, fn) == (1, 1, 0)
    assert len(matches) == 1
    # The matched prediction must be one of the two candidates, used once.
    assert matches[0][1] == 10
    assert matches[0][0] in (9, 11)


def test_no_double_match_one_pred_two_truths() -> None:
    """One prediction between two truths matches only one; the other is FN."""
    true = np.array([8, 12], dtype=np.int64)
    pred = np.array([10], dtype=np.int64)
    tp, fp, fn, matches = match_events(pred, true, tolerance=3)
    assert (tp, fp, fn) == (1, 0, 1)
    assert len(matches) == 1
    assert matches[0][0] == 10


def test_assignment_prefers_within_tolerance_pairing() -> None:
    """Global matching must not waste a near pair on a far one.

    Truths at [10, 20]; predictions at [21, 11]. A naive nearest-first or a
    sum-minimiser that ignores the tolerance gate could try (21->20, 11->10)
    which is correct here, but the key property is that BOTH within-tolerance
    pairings are recovered rather than a cross pairing (21->10, 11->20) that
    would exceed tolerance. Expect 2 TP with small deltas.
    """
    true = np.array([10, 20], dtype=np.int64)
    pred = np.array([21, 11], dtype=np.int64)
    tp, fp, fn, matches = match_events(pred, true, tolerance=2)
    assert (tp, fp, fn) == (2, 0, 0)
    # Each accepted delta is within tolerance.
    assert all(m[2] <= 2 for m in matches)
    # Pairing is the within-tolerance one: 11<->10 and 21<->20.
    deltas = {m[1]: m[2] for m in matches}
    assert deltas == {10: 1, 20: 1}


def test_crowded_far_pred_does_not_block_near_truth() -> None:
    """A far surplus prediction stays FP without stealing a valid match.

    Truths [10, 20]; predictions [10, 20, 100]. The first two align exactly; the
    third is far from everything. Expect 2 TP, 1 FP (the 100), 0 FN even though
    the optimiser sees a 3x2 matrix where one prediction must go unmatched.
    """
    true = np.array([10, 20], dtype=np.int64)
    pred = np.array([10, 20, 100], dtype=np.int64)
    tp, fp, fn, matches = match_events(pred, true, tolerance=2)
    assert (tp, fp, fn) == (2, 1, 0)
    assert {m[0] for m in matches} == {10, 20}


def test_counts_invariant_on_random_inputs() -> None:
    """tp+fp==n_pred and tp+fn==n_true, and no index matched twice (fuzz)."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        n_pred = int(rng.integers(0, 12))
        n_true = int(rng.integers(0, 12))
        pred = rng.integers(0, 200, size=n_pred).astype(np.int64)
        true = rng.integers(0, 200, size=n_true).astype(np.int64)
        tol = int(rng.integers(0, 6))
        tp, fp, fn, matches = match_events(pred, true, tolerance=tol)
        assert tp + fp == n_pred
        assert tp + fn == n_true
        assert tp == len(matches)
        # No predicted bar value position and no true bar reused: check the
        # matched true bars are distinct, and every accepted delta respects tol.
        matched_true_bars = [m[1] for m in matches]
        assert len(matched_true_bars) == len(set(matched_true_bars))
        assert all(0 <= m[2] <= tol for m in matches)


def test_empty_sides() -> None:
    """Empty predictions or truths yield only FN or only FP respectively."""
    some = np.array([1, 2, 3], dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    assert match_events(empty, some, tolerance=2) == (0, 0, 3, [])
    assert match_events(some, empty, tolerance=2) == (0, 3, 0, [])
    assert match_events(empty, empty, tolerance=2) == (0, 0, 0, [])


def test_negative_tolerance_raises() -> None:
    """A negative tolerance is rejected with ``ValueError``."""
    idx = np.array([1, 2], dtype=np.int64)
    try:
        match_events(idx, idx, tolerance=-1)
    except ValueError:
        return
    raise AssertionError("negative tolerance should have raised ValueError")
