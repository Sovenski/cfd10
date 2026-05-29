"""Tolerance-aware one-to-one matching of predicted to true event bars.

The detector emits *event* bar-indices (e.g. the bars it flags as turns); the
oracle supplies the *true* event bar-indices. Scoring those flags is a bipartite
matching problem: a predicted bar within ``tolerance`` bars of a true bar should
count once as a true positive, and no predicted bar may claim more than one true
bar (nor vice versa).

Algorithm
---------
We build a rectangular cost matrix ``C`` over predictions x truths with
``C[i, j] = |pred[i] - true[j]|`` and solve the minimum-cost assignment with
:func:`scipy.optimize.linear_sum_assignment` (the Hungarian / Jonker-Volgenant
algorithm). To stop the optimiser from spending a match on an out-of-tolerance
pair while a within-tolerance pair is still available, entries with
``|delta| > tolerance`` are inflated to a sentinel cost strictly larger than any
total achievable from feasible (within-tolerance) matches. Each returned
assignment is then accepted as a true positive **only** when its raw distance is
``<= tolerance``; assignments that fall back onto the sentinel are discarded.

The result is the canonical confusion triple for event detection:

* ``tp`` — predictions matched to a distinct true bar within tolerance,
* ``fp`` — predictions left unmatched (or matched only via the sentinel),
* ``fn`` — true bars left unmatched,

together with the explicit list of accepted ``(pred_idx, true_idx, |delta|)``
matches (sorted by true bar) for inspection and downstream reporting.

By construction ``tp + fp == len(pred_idx)`` and ``tp + fn == len(true_idx)`` and
no index appears in more than one match (no double-counting).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "EventMatch",
    "match_events",
]

# An accepted within-tolerance match: (predicted bar, true bar, |delta| bars).
EventMatch = tuple[int, int, int]


def _as_int_1d(arr: NDArray[np.int64], name: str) -> NDArray[np.int64]:
    """Return ``arr`` as a contiguous 1-D ``int64`` array.

    Args:
        arr: Candidate index array (bar positions).
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


def match_events(
    pred_idx: NDArray[np.int64],
    true_idx: NDArray[np.int64],
    tolerance: int,
) -> tuple[int, int, int, list[EventMatch]]:
    """Greedily-optimal one-to-one match of predicted to true event bars.

    A predicted bar matches a true bar when they lie within ``tolerance`` bars of
    one another; the global assignment minimising total absolute bar-distance is
    found with :func:`scipy.optimize.linear_sum_assignment`, and each assignment
    is kept only if its distance is ``<= tolerance``. No bar is matched twice.

    Args:
        pred_idx: 1-D array of predicted event bar-indices. Need not be sorted or
            unique, though duplicates compete for distinct true bars.
        true_idx: 1-D array of true event bar-indices.
        tolerance: Maximum absolute bar-distance ``|pred - true|`` (in bars) for a
            match to count as a true positive. Must be ``>= 0``.

    Returns:
        ``(tp, fp, fn, matches)`` where ``tp``/``fp``/``fn`` are the true-positive,
        false-positive and false-negative counts and ``matches`` is the list of
        accepted ``(pred_bar, true_bar, abs_delta)`` triples sorted by ``true_bar``.
        Invariants: ``tp + fp == len(pred_idx)`` and ``tp + fn == len(true_idx)``.

    Raises:
        ValueError: If ``tolerance`` is negative or an input is not 1-D.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")

    pred = _as_int_1d(pred_idx, "pred_idx")
    true = _as_int_1d(true_idx, "true_idx")
    n_pred = pred.shape[0]
    n_true = true.shape[0]

    # Degenerate sides: nothing can match, everything is FP / FN.
    if n_pred == 0 or n_true == 0:
        logger.debug(
            "match_events: empty side (n_pred=%d, n_true=%d); no matches",
            n_pred,
            n_true,
        )
        return 0, n_pred, n_true, []

    # Absolute bar-distance cost matrix (predictions x truths).
    cost = np.abs(pred[:, None] - true[None, :]).astype(np.float64)

    # Sentinel for out-of-tolerance pairs: strictly larger than any total a
    # feasible assignment could accrue, so the optimiser always prefers a real
    # within-tolerance match when one is still free. The largest possible sum of
    # feasible distances is < tolerance * min(n_pred, n_true) + 1.
    k = min(n_pred, n_true)
    sentinel = float(tolerance) * float(k) + 1.0
    gated = np.where(cost <= tolerance, cost, sentinel)

    row_ind, col_ind = linear_sum_assignment(gated)

    matches: list[EventMatch] = []
    matched_pred = np.zeros(n_pred, dtype=bool)
    matched_true = np.zeros(n_true, dtype=bool)
    for r, c in zip(row_ind, col_ind, strict=True):
        delta = int(cost[r, c])
        if delta <= tolerance:
            matched_pred[r] = True
            matched_true[c] = True
            matches.append((int(pred[r]), int(true[c]), delta))

    matches.sort(key=lambda m: (m[1], m[0]))
    tp = len(matches)
    fp = n_pred - tp
    fn = n_true - tp

    logger.debug(
        "match_events: tp=%d fp=%d fn=%d (tol=%d, n_pred=%d, n_true=%d)",
        tp,
        fp,
        fn,
        tolerance,
        n_pred,
        n_true,
    )
    return tp, fp, fn, matches
