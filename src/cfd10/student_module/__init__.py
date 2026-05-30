"""cfd10 student layer: shallow, Pine-exportable trees distilled from the teacher.

The GBDT teacher is accurate but un-exportable. This package produces the
*deployable* ruleset: one shallow :class:`~sklearn.tree.DecisionTreeClassifier`
per side whose ``feature <= threshold`` splits transcribe directly into a
TradingView Pine script.

Public API
----------
:class:`StudentConfig` (frozen) — small-student capacity knobs (``kind`` is
``"tree"`` or ``"gboost"``; tree depth up to 8, or a small gradient boost).
:func:`fit_student` — fit the deployable student (tree or gboost) to the oracle's
hard labels.
:func:`distill_from_teacher` — fit a shallow tree to the teacher's soft calls
(fidelity reporting).
:func:`tree_to_rules` / :func:`student_features` — render readable rules and list
the features a single tree actually uses.
:func:`ensemble_to_rules` / :func:`ensemble_features` — the gradient-boosting
counterparts (the latter is the union of features used across the ensemble).
"""

from __future__ import annotations

from cfd10.student_module.distill import (
    StudentConfig,
    distill_from_teacher,
    ensemble_features,
    ensemble_to_rules,
    fit_student,
    student_features,
    tree_to_rules,
)
from cfd10.student_module.export_pine import (
    FeaturePine,
    emit_indicator,
    feature_pine_expr,
    load_student_json,
    write_indicator,
)

__all__ = [
    "StudentConfig",
    "fit_student",
    "distill_from_teacher",
    "tree_to_rules",
    "student_features",
    "ensemble_to_rules",
    "ensemble_features",
    "FeaturePine",
    "feature_pine_expr",
    "load_student_json",
    "emit_indicator",
    "write_indicator",
]
