"""Event-detection precision / recall / F1 built on tolerance-aware matching.

These are the headline detector scores. They reduce the confusion triple from
:func:`cfd10.eval_module.events.match_events` to the standard rates:

* ``precision = tp / (tp + fp)`` — share of flagged bars that hit a true event,
* ``recall    = tp / (tp + fn)`` — share of true events that were flagged,
* ``f1        = 2 * precision * recall / (precision + recall)`` — their harmonic
  mean.

Empty-denominator cases follow the usual scikit-learn ``zero_division=0``
convention: with no predictions precision is ``0``, with no true events recall is
``0``, and an all-zero confusion triple yields ``(0, 0, 0)``. ``event_prf`` is the
function the baseline calls, so its signature
``event_prf(pred_idx, true_idx, tolerance) -> (precision, recall, f1)`` is stable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cfd10.eval_module.events import match_events
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "prf_from_counts",
    "event_prf",
]


def prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Convert a confusion triple to ``(precision, recall, f1)``.

    Uses the ``zero_division=0`` convention: any rate with a zero denominator is
    reported as ``0.0`` (rather than ``NaN``), and F1 is ``0.0`` whenever either
    precision or recall is ``0.0``.

    Args:
        tp: True-positive count (``>= 0``).
        fp: False-positive count (``>= 0``).
        fn: False-negative count (``>= 0``).

    Returns:
        ``(precision, recall, f1)`` as plain floats in ``[0, 1]``.

    Raises:
        ValueError: If any count is negative.
    """
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError(f"counts must be non-negative, got tp={tp}, fp={fp}, fn={fn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    return float(precision), float(recall), float(f1)


def event_prf(
    pred_idx: NDArray[np.int64],
    true_idx: NDArray[np.int64],
    tolerance: int,
) -> tuple[float, float, float]:
    """Tolerance-aware event precision, recall and F1.

    Matches predicted event bars to true event bars with
    :func:`cfd10.eval_module.events.match_events` (one-to-one, gated by
    ``tolerance``) and returns the resulting rates.

    Args:
        pred_idx: 1-D array of predicted event bar-indices.
        true_idx: 1-D array of true event bar-indices.
        tolerance: Maximum absolute bar-distance for a match (``>= 0``).

    Returns:
        ``(precision, recall, f1)`` as floats in ``[0, 1]``.

    Raises:
        ValueError: If ``tolerance`` is negative or an input is not 1-D.
    """
    tp, fp, fn, _matches = match_events(pred_idx, true_idx, tolerance)
    precision, recall, f1 = prf_from_counts(tp, fp, fn)
    logger.debug(
        "event_prf: precision=%.4f recall=%.4f f1=%.4f (tol=%d)",
        precision,
        recall,
        f1,
        tolerance,
    )
    return precision, recall, f1
