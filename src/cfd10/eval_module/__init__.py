"""cfd10 evaluation layer: event matching, PRF metrics, deflation, reporting.

This package scores the turn detector's *event* predictions against the oracle's
true events and guards the results against multiple-testing selection bias.

Public API
----------
Event matching (:mod:`cfd10.eval_module.events`)
    :func:`match_events` — tolerance-gated one-to-one assignment of predicted to
    true event bars (Hungarian algorithm), returning ``(tp, fp, fn, matches)``;
    :data:`EventMatch` is the accepted-match triple type.
Metrics (:mod:`cfd10.eval_module.metrics`)
    :func:`event_prf` — event ``(precision, recall, f1)`` over a tolerance; the
    function the baseline calls. :func:`prf_from_counts` converts a confusion
    triple directly.
Ranking (:mod:`cfd10.eval_module.ranking`)
    Threshold-free, n-weighted scores that survive an F1 collapse:
    :func:`average_precision` (the headline PR-AUC, with an optional oracle-score
    ``sample_weight``), :func:`precision_at_k` / :func:`recall_at_k` /
    :func:`lift_at_k`, and their tolerance-windowed siblings
    :func:`precision_at_k_event` / :func:`recall_at_k_event`.
Deflation (:mod:`cfd10.eval_module.deflation`)
    :func:`deflated_metric` (Bailey / Lopez de Prado haircut or DSR probability),
    :func:`pbo_cscv` (CSCV probability of backtest overfitting in ``[0, 1]``),
    plus the building blocks :func:`expected_max_sharpe` and
    :func:`probabilistic_sharpe_ratio`.
Reporting (:mod:`cfd10.eval_module.report`)
    :class:`ScoreRow` / :class:`Scorecard` — per-``(side, asset, era)`` rows with
    a :meth:`Scorecard.to_markdown` renderer.
"""

from __future__ import annotations

from cfd10.eval_module.deflation import (
    deflated_metric,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
)
from cfd10.eval_module.events import EventMatch, match_events
from cfd10.eval_module.metrics import event_prf, prf_from_counts
from cfd10.eval_module.ranking import (
    average_precision,
    lift_at_k,
    precision_at_k,
    precision_at_k_event,
    recall_at_k,
    recall_at_k_event,
)
from cfd10.eval_module.report import Scorecard, ScoreRow

__all__ = [
    # Event matching.
    "match_events",
    "EventMatch",
    # Metrics.
    "event_prf",
    "prf_from_counts",
    # Ranking.
    "average_precision",
    "precision_at_k",
    "recall_at_k",
    "lift_at_k",
    "precision_at_k_event",
    "recall_at_k_event",
    # Deflation.
    "deflated_metric",
    "pbo_cscv",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    # Reporting.
    "Scorecard",
    "ScoreRow",
]
