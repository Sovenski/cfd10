"""Internal parity: re-sim the emitted rules and pin them to the sklearn model.

The Pine export (:mod:`cfd10.student_module.export_pine`) transcribes a student
model's JSON into Pine. This module proves that transcription is faithful *in
pure Python* and pins it, bar for bar, to the fitted sklearn model on the pooled
feature matrix. Two model kinds are handled:

* **tree** — :func:`simulate_tree` walks the JSON node array (``feature <=
  threshold`` routes LEFT, else RIGHT, leaf emits its stored ``predict``) and the
  re-sim is compared to ``DecisionTreeClassifier.predict()``.
* **gboost** — :func:`simulate_gboost` re-implements the *emitted* additive
  margin ``init + learning_rate * sum_over_stages(stage leaf value)`` from the
  JSON, applies the same baked decision rule ``margin >= logit(threshold)`` the
  Pine emits, and the resulting BINARY SIGNAL is compared to
  ``GradientBoostingClassifier.predict()`` (at ``threshold = 0.5``). The continuous
  margin is also pinned to the booster's own ``decision_function`` so the
  max-abs margin error is reported (~0 / 1e-9).

The JSON-vs-Pine boundary is textual (same thresholds, branch order, leaf
values / labels — checked by the export tests), and the JSON-vs-sklearn boundary
is this re-simulation. Chaining the two means: emitted Pine rules == JSON model
== sklearn ``predict()``. The only remaining gap — whether TradingView's *feature
values* equal Python's — is not provable here; it is pinned by the Data-Window
export contract in ``pine_export_contract.md``.

Why the JSON labels are exact
-----------------------------
For a tree, ``_tree_nodes`` stores ``predict = int(pos_proba >= 0.5)`` per leaf,
identical to sklearn's ``predict()`` (argmax of the leaf's class-weighted value)
for a two-class tree. For a gboost, the binary log-loss booster's ``predict()``
is ``sigmoid(decision_function) >= 0.5`` <=> ``decision_function >= 0``; since the
sigmoid is monotone, comparing the raw margin to ``logit(threshold)`` reproduces
``proba >= threshold`` exactly (``logit(0.5) == 0`` recovers ``predict()``). The
pooled matrix is warm-up-free (no NaN), so routing is unambiguous and the
agreement is exact, not approximate.

Public API
----------
:func:`simulate_tree` — pure-Python evaluation of a JSON tree over a matrix.
:func:`simulate_gboost` — pure-Python additive margin of a JSON gboost over a
matrix.
:func:`verify_student_export` — re-sim the JSON model (tree or gboost) and assert
the binary signal equals the sklearn ``predict()`` exactly.
:class:`ParityReport` (frozen) — the verification result (counts + agreement, plus
the gboost margin max-abs error).
"""

from __future__ import annotations

import math
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
    "simulate_gboost",
    "verify_student_export",
]

# A student tree as parsed from JSON: a heterogeneous mapping whose ``nodes`` key
# holds the node array. Typed as ``Any``-valued because the values mix str / int
# / float / bool / None / list per the export schema.
TreeJson = dict[str, Any]


@dataclass(frozen=True)
class ParityReport:
    """Result of pinning the re-simulated JSON model to the sklearn ``predict()``.

    Attributes:
        n_rows: Number of evaluated rows.
        n_agree: Rows where the re-sim label equals the sklearn label.
        n_disagree: Rows where they differ (``0`` on a pass).
        exact: ``True`` iff every row agreed (``n_disagree == 0``).
        n_pred_pos_sim: Positive calls from the re-simulation.
        n_pred_pos_sklearn: Positive calls from the sklearn model.
        features: The model's used feature names (export targets).
        model_kind: ``"tree"`` or ``"gboost"``.
        margin_max_abs_err: For a gboost, the max absolute difference between the
            re-simulated additive margin and the booster's own
            ``decision_function`` (``0.0`` for a tree, or when no booster is
            supplied to pin the margin against). Should be ~0 / 1e-9 on a pass.
    """

    n_rows: int
    n_agree: int
    n_disagree: int
    exact: bool
    n_pred_pos_sim: int
    n_pred_pos_sklearn: int
    features: list[str]
    model_kind: str = "tree"
    margin_max_abs_err: float = 0.0


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
# Pure-Python additive-margin evaluator (the gboost parity reference).         #
# --------------------------------------------------------------------------- #


def _logit(p: float) -> float:
    """Return ``log(p / (1 - p))`` with ``p`` clipped to keep the log finite.

    The Pine bakes ``logit(threshold)`` and compares the raw margin against it;
    this mirrors that constant so the re-sim applies the identical decision rule.
    """
    clipped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(math.log(clipped / (1.0 - clipped)))


def _stage_columns(
    stages: list[list[dict[str, Any]]],
    feature_names: list[str],
) -> dict[str, int]:
    """Map every split feature across all gboost stages to its matrix column.

    Args:
        stages: The gboost ``stages`` list (each a node array — a list of node
            dicts — as written by ``redistill_students._model_to_json``).
        feature_names: Column names aligned to the evaluation matrix.

    Returns:
        Mapping ``feature_name -> column index`` for each feature any stage splits
        on.

    Raises:
        KeyError: If a split feature is absent from ``feature_names``.
    """
    name_to_col = {name: i for i, name in enumerate(feature_names)}
    out: dict[str, int] = {}
    for stage_nodes in stages:
        for node in stage_nodes:
            if bool(node["is_leaf"]):
                continue
            fname = str(node["feature"])
            if fname not in name_to_col:
                raise KeyError(
                    f"simulate_gboost: split feature {fname!r} not in feature_names"
                )
            out[fname] = name_to_col[fname]
    return out


def _stage_leaf_value(
    nodes: dict[int, dict[str, Any]],
    cols: dict[str, int],
    row: NDArray[np.float64],
) -> float:
    """Route ``row`` through one stage regressor tree and return its leaf value.

    ``feature <= threshold`` routes LEFT (scikit-learn's rule), matching both the
    classifier path and the emitted Pine ``if/else``.
    """
    node = nodes[0]
    while not bool(node["is_leaf"]):
        value = row[cols[str(node["feature"])]]
        threshold = float(node["threshold"])
        go_left = not (value > threshold)
        node = nodes[int(node["left"]) if go_left else int(node["right"])]
    return float(node["value"])


def simulate_gboost(
    model_json: TreeJson,
    X: NDArray[np.float64],
    feature_names: list[str],
) -> NDArray[np.float64]:
    """Re-simulate the emitted gboost additive margin from the JSON (pure Python).

    Computes, per row, the exact quantity the Pine export emits::

        margin = init + learning_rate * sum_over_stages(stage leaf value)

    by walking each stage's regressor node array (``feature <= threshold`` routes
    LEFT) and summing the selected leaf values. This is the continuous score the
    Pine compares against ``logit(threshold)``; a match against the booster's
    ``decision_function`` proves the additive transcription is faithful.

    Args:
        model_json: A gboost model dict (``init``, ``learning_rate``, ``stages``).
        X: Dense feature matrix, shape ``(n_rows, n_features)``.
        feature_names: Column names aligned to ``X`` (split features resolved by
            name, matching the Pine export).

    Returns:
        ``float64`` array of per-row additive margins, length ``n_rows``.

    Raises:
        ValueError: If ``model_json`` is not a gboost export, ``X`` is not 2-D, or
            its column count disagrees with ``feature_names``.
        KeyError: If a split feature is absent from ``feature_names``.
    """
    if str(model_json.get("model_kind", "tree")) != "gboost":
        raise ValueError("simulate_gboost: model_json is not a 'gboost' export")
    matrix = np.ascontiguousarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"simulate_gboost: X must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"simulate_gboost: X has {matrix.shape[1]} columns but "
            f"{len(feature_names)} feature names"
        )

    init = float(model_json["init"])
    lr = float(model_json["learning_rate"])
    stages = model_json["stages"]
    stage_nodes = [{int(n["node_id"]): n for n in stage} for stage in stages]
    cols = _stage_columns(stages, feature_names)

    n_rows = matrix.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    for row_i in range(n_rows):
        row = matrix[row_i]
        total = 0.0
        for nodes in stage_nodes:
            total += _stage_leaf_value(nodes, cols, row)
        out[row_i] = init + lr * total
    return out


# --------------------------------------------------------------------------- #
# Verification against the fitted sklearn model.                               #
# --------------------------------------------------------------------------- #


def _resolve_matrix(
    X,  # type: ignore[no-untyped-def]
    feature_names: list[str] | None,
) -> tuple[list[str], NDArray[np.float64]]:
    """Resolve ``(names, matrix)`` from a DataFrame or a bare array + names."""
    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        return list(X.columns), np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    if feature_names is None:
        raise ValueError(
            "verify_student_export: feature_names required when X is an array"
        )
    return list(feature_names), np.ascontiguousarray(X, dtype=np.float64)


def _gboost_margin_error(
    model,  # type: ignore[no-untyped-def]  # fitted GradientBoostingClassifier.
    matrix: NDArray[np.float64],
    sim_margin: NDArray[np.float64],
) -> float:
    """Max-abs error of the re-sim margin vs the booster's ``decision_function``.

    The booster's ``decision_function`` is its raw additive margin (``init_ +
    learning_rate * sum(stage leaf)``), exactly what :func:`simulate_gboost`
    reconstructs from the JSON. Returns ``0.0`` if the model cannot supply a
    decision function (then only the binary signal is pinned).
    """
    if not hasattr(model, "decision_function"):
        return 0.0
    raw = np.asarray(model.decision_function(matrix), dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(raw - sim_margin))) if raw.size else 0.0


def verify_student_export(
    tree_json: TreeJson,
    X,  # type: ignore[no-untyped-def]  # pandas.DataFrame | ndarray; kept loose.
    sklearn_tree=None,  # type: ignore[no-untyped-def]  # fitted tree OR gboost.
    feature_names: list[str] | None = None,
    threshold: float = 0.5,
) -> ParityReport:
    """Assert the re-simulated JSON model equals the sklearn ``predict()``.

    Dispatches on ``tree_json['model_kind']``:

    * **tree** — re-simulates the node array via :func:`simulate_tree` and compares
      row by row against the fitted tree's ``predict()``.
    * **gboost** — reconstructs the additive margin via :func:`simulate_gboost`,
      applies the *same* baked rule the Pine emits (``margin >= logit(threshold)``)
      to get a BINARY SIGNAL, and compares that signal against the booster's
      ``predict()`` (at ``threshold = 0.5`` this is exactly ``margin >= 0``). The
      continuous margin is additionally pinned to the booster's
      ``decision_function`` and the max-abs error reported (~0 / 1e-9).

    A pass (``exact=True``) proves the emitted rules — and therefore the Pine they
    mirror — are a faithful translation of the student. When ``sklearn_tree`` is
    ``None`` the re-sim is compared to itself (self-consistency; counts only).

    Args:
        tree_json: The student model dict (tree or gboost) to re-simulate.
        X: Feature matrix — a ``pandas.DataFrame`` (its columns supply
            ``feature_names``) or a 2-D array (then ``feature_names`` is required).
        sklearn_tree: The fitted ``DecisionTreeClassifier`` /
            ``GradientBoostingClassifier`` to pin against; if ``None``, only the
            re-sim counts are reported.
        feature_names: Column names when ``X`` is a bare array; ignored (and taken
            from the frame) when ``X`` is a ``DataFrame``.
        threshold: Probability cut for a gboost (the same value baked into the Pine
            via ``logit(threshold)``); ``0.5`` reproduces ``predict()``.

    Returns:
        The :class:`ParityReport`.

    Raises:
        ValueError: If ``X`` is an array and ``feature_names`` is ``None``, or the
            re-simulation disagrees with a provided ``sklearn_tree``.
        KeyError: If a split feature is absent from ``feature_names``.
    """
    names, matrix = _resolve_matrix(X, feature_names)
    kind = str(tree_json.get("model_kind", "tree"))

    margin_err = 0.0
    if kind == "gboost":
        sim_margin = simulate_gboost(tree_json, matrix, names)
        sim = (sim_margin >= _logit(threshold)).astype(np.int64)
        if sklearn_tree is not None:
            reference = np.asarray(sklearn_tree.predict(matrix), dtype=np.int64)
            margin_err = _gboost_margin_error(sklearn_tree, matrix, sim_margin)
        else:
            reference = sim
    else:
        sim = simulate_tree(tree_json, matrix, names)
        reference = (
            np.asarray(sklearn_tree.predict(matrix), dtype=np.int64)
            if sklearn_tree is not None
            else sim
        )

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
        model_kind=kind,
        margin_max_abs_err=margin_err,
    )

    logger.info(
        "verify_student_export[%s]: rows=%d agree=%d disagree=%d exact=%s "
        "(sim_pos=%d, sklearn_pos=%d, margin_max_abs_err=%.3e)",
        report.model_kind,
        report.n_rows,
        report.n_agree,
        report.n_disagree,
        report.exact,
        report.n_pred_pos_sim,
        report.n_pred_pos_sklearn,
        report.margin_max_abs_err,
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
