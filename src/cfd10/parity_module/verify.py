"""Internal parity: re-sim the emitted rules and pin them to the sklearn tree.

The Pine export (:mod:`cfd10.parity_module.export_pine`) transcribes a student
tree's JSON into nested ``if/else`` rules. This module proves that transcription
is faithful *in pure Python*: :func:`simulate_tree` walks the very same JSON node
array — ``feature <= threshold`` routes LEFT, else RIGHT, leaf emits its stored
``predict`` — and :func:`verify_student_export` asserts, bar for bar, that this
re-simulation equals the fitted ``DecisionTreeClassifier.predict()`` on the
pooled feature matrix.

The JSON-vs-Pine boundary is textual (same thresholds, same branch order, same
leaf labels — checked by the export tests), and the JSON-vs-sklearn boundary is
this re-simulation. Chaining the two means: emitted Pine rules == JSON tree ==
sklearn ``predict()``. The only remaining gap — whether TradingView's *feature
values* equal Python's — is not provable here; it is pinned by the Data-Window
export contract in ``pine_export_contract.md``.

Why the JSON ``predict`` field is exact
---------------------------------------
``distill_students._tree_to_json`` stores ``predict = int(pos_proba >= 0.5)`` per
leaf. For a two-class tree this is identical to sklearn's ``predict()`` (argmax
of the leaf's class-weighted value), because ``pos_mass / total >= 0.5`` iff the
positive class has at least half the mass iff it is the argmax. The pooled matrix
is warm-up-free (no NaN), so routing is unambiguous and the agreement is exact,
not approximate.

Public API
----------
:func:`simulate_tree` — pure-Python evaluation of a JSON tree over a matrix.
:func:`verify_student_export` — assert the re-sim equals the sklearn ``predict()``.
:class:`ParityReport` (frozen) — the verification result (counts + agreement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "ParityReport",
    "TreeJson",
    "simulate_tree",
    "verify_student_export",
]

# A student tree as parsed from JSON: a heterogeneous mapping whose ``nodes`` key
# holds the node array. Typed as ``Any``-valued because the values mix str / int
# / float / bool / None / list per the export schema.
TreeJson = dict[str, Any]


@dataclass(frozen=True)
class ParityReport:
    """Result of pinning the re-simulated JSON tree to the sklearn ``predict()``.

    Attributes:
        n_rows: Number of evaluated rows.
        n_agree: Rows where the re-sim label equals the sklearn label.
        n_disagree: Rows where they differ (``0`` on a pass).
        exact: ``True`` iff every row agreed (``n_disagree == 0``).
        n_pred_pos_sim: Positive calls from the re-simulation.
        n_pred_pos_sklearn: Positive calls from the sklearn tree.
        features: The tree's used feature names (export targets).
    """

    n_rows: int
    n_agree: int
    n_disagree: int
    exact: bool
    n_pred_pos_sim: int
    n_pred_pos_sklearn: int
    features: list[str]


# --------------------------------------------------------------------------- #
# Pure-Python decision-path evaluator (the parity reference for the Pine).     #
# --------------------------------------------------------------------------- #


def _index_nodes(tree_json: TreeJson) -> dict[int, dict[str, Any]]:
    """Index the JSON node list by ``node_id``.

    Args:
        tree_json: A student tree dict (must carry a ``nodes`` list).

    Returns:
        Mapping ``node_id -> node`` for O(1) child resolution.

    Raises:
        ValueError: If ``nodes`` is missing or not a list.
    """
    nodes = tree_json.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("simulate_tree: tree_json has no 'nodes' list")
    return {int(n["node_id"]): n for n in nodes}


def _feature_columns(
    tree_json: TreeJson,
    feature_names: list[str],
) -> dict[str, int]:
    """Map each split feature name to its column index in ``feature_names``.

    Only features the tree actually splits on are resolved (leaves carry
    ``feature: null``). The JSON also stores a ``feature_index``, but resolving by
    *name* against the matrix's own column order is what makes the check robust to
    a differently-ordered matrix — and is exactly what the Pine export does.

    Args:
        tree_json: The student tree dict.
        feature_names: Column names aligned to the evaluation matrix.

    Returns:
        Mapping ``feature_name -> column index``.

    Raises:
        KeyError: If a split feature is absent from ``feature_names``.
    """
    name_to_col = {name: i for i, name in enumerate(feature_names)}
    out: dict[str, int] = {}
    for node in tree_json["nodes"]:
        if bool(node["is_leaf"]):
            continue
        fname = str(node["feature"])
        if fname not in name_to_col:
            raise KeyError(
                f"simulate_tree: split feature {fname!r} not in feature_names"
            )
        out[fname] = name_to_col[fname]
    return out


def simulate_tree(
    tree_json: TreeJson,
    X: NDArray[np.float64],
    feature_names: list[str],
) -> NDArray[np.int64]:
    """Re-simulate the emitted decision path from the JSON tree (pure Python).

    Walks the JSON node array per row: at an internal node, route to the LEFT
    child when ``X[feature] <= threshold`` (scikit-learn's rule) else RIGHT; at a
    leaf, emit its stored ``predict``. This is the exact computation the Pine
    ``if/else`` performs, so a match against the sklearn tree proves the export is
    faithful.

    NaN handling mirrors scikit-learn's default (a non-NaN-trained tree sends NaN
    down the *left* branch), but the pooled matrix is warm-up-free so this never
    triggers in the parity check.

    Args:
        tree_json: A student tree dict (``nodes`` array with ``feature`` names,
            ``threshold``, ``left`` / ``right`` child ids, leaf ``predict``).
        X: Dense feature matrix, shape ``(n_rows, n_features)``.
        feature_names: Column names aligned to ``X`` (resolves split features by
            name, matching the Pine export).

    Returns:
        ``int64`` array of per-row predictions (``0`` / ``1``), length ``n_rows``.

    Raises:
        ValueError: If ``X`` is not 2-D or its column count disagrees with
            ``feature_names``.
        KeyError: If a split feature is absent from ``feature_names``.
    """
    matrix = np.ascontiguousarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"simulate_tree: X must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"simulate_tree: X has {matrix.shape[1]} columns but "
            f"{len(feature_names)} feature names"
        )

    nodes = _index_nodes(tree_json)
    cols = _feature_columns(tree_json, feature_names)
    n_rows = matrix.shape[0]
    out = np.empty(n_rows, dtype=np.int64)

    for row in range(n_rows):
        node = nodes[0]
        while not bool(node["is_leaf"]):
            col = cols[str(node["feature"])]
            value = matrix[row, col]
            threshold = float(node["threshold"])
            # scikit-learn: ``<= threshold`` -> left; NaN -> left (default policy).
            go_left = not (value > threshold)
            nxt = int(node["left"]) if go_left else int(node["right"])
            node = nodes[nxt]
        out[row] = int(node["predict"])
    return out


# --------------------------------------------------------------------------- #
# Verification against the fitted sklearn tree.                                #
# --------------------------------------------------------------------------- #


def verify_student_export(
    tree_json: TreeJson,
    X,  # type: ignore[no-untyped-def]  # pandas.DataFrame | ndarray; kept loose.
    sklearn_tree=None,  # type: ignore[no-untyped-def]  # fitted DecisionTreeClassifier.
    feature_names: list[str] | None = None,
) -> ParityReport:
    """Assert the re-simulated JSON tree equals the sklearn tree's ``predict()``.

    Re-simulates ``tree_json`` over ``X`` via :func:`simulate_tree` and compares,
    row by row, against the fitted tree's ``predict()``. A pass (``exact=True``)
    proves the emitted ruleset — and therefore the Pine ``if/else`` it mirrors —
    is a faithful translation of the student.

    When ``sklearn_tree`` is ``None`` the JSON's own per-leaf ``predict`` is the
    reference (self-consistency of the export); the strong proof passes the actual
    fitted tree so the two independent code paths must agree.

    Args:
        tree_json: The student tree dict to re-simulate.
        X: Feature matrix — a ``pandas.DataFrame`` (its columns supply
            ``feature_names``) or a 2-D array (then ``feature_names`` is required).
        sklearn_tree: The fitted ``DecisionTreeClassifier`` to pin against; if
            ``None``, the re-sim is compared to itself (always exact) and only the
            counts are reported.
        feature_names: Column names when ``X`` is a bare array; ignored (and taken
            from the frame) when ``X`` is a ``DataFrame``.

    Returns:
        The :class:`ParityReport`.

    Raises:
        ValueError: If ``X`` is an array and ``feature_names`` is ``None``, or the
            re-simulation disagrees with a provided ``sklearn_tree``.
    """
    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        names = list(X.columns)
        matrix = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    else:
        if feature_names is None:
            raise ValueError(
                "verify_student_export: feature_names required when X is an array"
            )
        names = list(feature_names)
        matrix = np.ascontiguousarray(X, dtype=np.float64)

    sim = simulate_tree(tree_json, matrix, names)

    if sklearn_tree is not None:
        reference = np.asarray(sklearn_tree.predict(matrix), dtype=np.int64)
    else:
        reference = sim  # self-consistency: counts only.

    agree = sim == reference
    n_agree = int(agree.sum())
    n_disagree = int(agree.size - n_agree)
    exact = n_disagree == 0

    features = [str(f) for f in tree_json.get("features", [])]
    report = ParityReport(
        n_rows=int(sim.size),
        n_agree=n_agree,
        n_disagree=n_disagree,
        exact=exact,
        n_pred_pos_sim=int(sim.sum()),
        n_pred_pos_sklearn=int(reference.sum()),
        features=features,
    )

    logger.info(
        "verify_student_export: rows=%d agree=%d disagree=%d exact=%s "
        "(sim_pos=%d, sklearn_pos=%d)",
        report.n_rows,
        report.n_agree,
        report.n_disagree,
        report.exact,
        report.n_pred_pos_sim,
        report.n_pred_pos_sklearn,
    )

    if sklearn_tree is not None and not exact:
        # Surface a few offending rows to make a regression debuggable.
        bad = np.flatnonzero(~agree)[:5].tolist()
        raise ValueError(
            "verify_student_export: emitted-rules re-sim disagrees with sklearn "
            f"predict() on {n_disagree}/{report.n_rows} rows (e.g. rows {bad}); "
            "the Pine export is NOT a faithful translation of the student"
        )
    return report
