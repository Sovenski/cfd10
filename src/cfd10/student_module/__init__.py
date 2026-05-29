"""cfd10 student layer: shallow, Pine-exportable trees distilled from the teacher.

The GBDT teacher is accurate but un-exportable. This package produces the
*deployable* ruleset: one shallow :class:`~sklearn.tree.DecisionTreeClassifier`
per side whose ``feature <= threshold`` splits transcribe directly into a
TradingView Pine script.

Public API
----------
:class:`StudentConfig` (frozen) — shallow-tree capacity knobs.
:func:`fit_student` — fit the deployable student to the oracle's hard labels.
:func:`distill_from_teacher` — fit a shallow tree to the teacher's soft calls
(fidelity reporting).
:func:`tree_to_rules` / :func:`student_features` — render readable rules and list
the features a tree actually uses.
"""

from __future__ import annotations

from cfd10.student_module.distill import (
    StudentConfig,
    distill_from_teacher,
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
    "FeaturePine",
    "feature_pine_expr",
    "load_student_json",
    "emit_indicator",
    "write_indicator",
]
