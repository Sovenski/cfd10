"""cfd10 parity layer: prove the Pine export faithfully mirrors the student tree.

Two artefacts are produced upstream by :mod:`cfd10.student_module` /
``pipeline/distill_students.py``: a shallow ``DecisionTreeClassifier`` per side
(pickled) and its export-friendly node JSON. The Pine generator
(:mod:`cfd10.parity_module.export_pine`) transcribes those JSON trees into a
single TradingView ``@version=6`` indicator.

This package closes the loop: :func:`verify_student_export` re-implements the
emitted decision path in *pure Python* straight off the same tree JSON and
asserts, bar for bar, that it reproduces the fitted sklearn tree's ``predict()``
on the pooled feature matrix. A green check proves the emitted ruleset (and
therefore the Pine ``if/else`` it mirrors) is an exact translation of the
student. The remaining, *un-provable-in-Python* gap is whether TradingView
computes the same feature *values* as the Python feature bank; that is pinned by
the Data-Window export contract in ``pine_export_contract.md``.

Public API
----------
:func:`verify_student_export` — re-sim the JSON tree and assert exact agreement
with the sklearn ``predict()``.
:func:`simulate_tree` — the pure-Python decision-path evaluator (the parity
reference for the Pine ``if/else``).
:class:`ParityReport` (frozen) — the per-side verification result.
"""

from __future__ import annotations

from cfd10.parity_module.verify import (
    ParityReport,
    simulate_gboost,
    simulate_tree,
    verify_student_export,
)

__all__ = [
    "ParityReport",
    "simulate_tree",
    "simulate_gboost",
    "verify_student_export",
]
