"""Fast CPU tests for the TCN teacher — wiring, shapes, and the group guarantee.

These run on tiny random / synthetic fixtures (never the full corpus) so the
suite is fast on CPU. They assert the *contract*, not a trained score:

* **overfit-a-batch** — a single fixed batch of 32 random windows is driven to a
  near-zero training loss within < 300 optimiser steps, proving the forward /
  backward path, the dual-head output, and the optimiser are correctly wired;
* **forward shape** — the model maps ``(N, W, F)`` to ``(N, 2)`` and the sigmoid
  of its logits lies in ``[0, 1]``;
* **group-boundary safety** — on a two-asset fixture *every* window the
  :class:`WindowDataset` emits has all its bars in a single group id (no window
  ever straddles the asset boundary), and no window ends on the forbidden
  boundary rows.

Real training happens on a Colab GPU via ``pipeline/fit_teacher.py``; these tests
deliberately do not exercise the full fit / CV path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cfd10.teacher_module.datasets import (
    WindowDataset,
    _valid_window_ends,
    compute_feature_stats,
)
from cfd10.teacher_module.models import ModelFactory, TCNConfig, TCNTurnModel

_SEED = 0


# --------------------------------------------------------------------------- #
# Fixtures.                                                                    #
# --------------------------------------------------------------------------- #


def _two_asset_frame(
    n_a: int = 120, n_b: int = 90, n_features: int = 6, seed: int = _SEED
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build a tiny two-asset pooled fixture (features, oracle labels, groups).

    Group 0 occupies the first ``n_a`` rows, group 1 the next ``n_b`` — exactly the
    contiguous-block layout :func:`build_pooled_dataset` produces. A few bars carry
    a positive oracle tier so the label/weight path is exercised.

    Returns:
        ``(X, labels, groups)``.
    """
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    groups = np.array([0] * n_a + [1] * n_b, dtype=np.int64)

    top_tier = np.full(n, "none", dtype=object)
    bottom_tier = np.full(n, "none", dtype=object)
    top_weight = np.zeros(n, dtype=np.float64)
    bottom_weight = np.zeros(n, dtype=np.float64)
    # Sprinkle a few positives in each asset (well inside the blocks).
    for r in (40, 70, 100, 160, 185):
        top_tier[r] = "strong"
        top_weight[r] = 0.7
    for r in (55, 95, 150, 175):
        bottom_tier[r] = "regular"
        bottom_weight[r] = 0.4

    labels = pd.DataFrame(
        {
            "top_score": top_weight,
            "bottom_score": bottom_weight,
            "top_tier": top_tier,
            "bottom_tier": bottom_tier,
            "top_weight": top_weight,
            "bottom_weight": bottom_weight,
        }
    )
    return X, labels, groups


# --------------------------------------------------------------------------- #
# Overfit a batch (gradients / wiring).                                        #
# --------------------------------------------------------------------------- #


def test_overfit_single_batch() -> None:
    """The model drives 32 random windows to ~0 loss within < 300 steps.

    A capacity-rich TCN must be able to memorise a small fixed batch; if it
    cannot, the forward/backward wiring or the dual-head output is broken. We use
    a plain (unweighted) BCE here — this is a pure optimisation sanity check, not
    the production weighted objective.
    """
    torch.manual_seed(_SEED)
    n, window, n_features = 32, 16, 6
    x = torch.randn(n, window, n_features)
    # Random but fixed two-logit targets.
    y = (torch.rand(n, 2) > 0.5).float()

    model = TCNTurnModel(TCNConfig(in_features=n_features, channels=32, dropout=0.0))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.BCEWithLogitsLoss()

    max_steps = 300
    final_loss = float("inf")
    for _step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if final_loss < 1e-3:
            break

    assert final_loss < 1e-2, (
        f"failed to overfit 32 windows in < {max_steps} steps "
        f"(final loss {final_loss:.4f}) — gradient/wiring problem"
    )


# --------------------------------------------------------------------------- #
# Forward shape + probability range.                                           #
# --------------------------------------------------------------------------- #


def test_forward_output_shape_and_sigmoid_range() -> None:
    """Forward maps ``(N, W, F)`` to ``(N, 2)`` and sigmoid(logits) is in [0, 1]."""
    torch.manual_seed(_SEED)
    n, window, n_features = 8, 24, 11
    model = TCNTurnModel(TCNConfig(in_features=n_features, channels=16))
    model.eval()
    x = torch.randn(n, window, n_features)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (n, 2)

    probs = torch.sigmoid(logits)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    assert torch.isfinite(probs).all()


def test_factory_resolves_tcn() -> None:
    """The model registry exposes the TCN builder under the name ``'tcn'``."""
    builder = ModelFactory("tcn")
    model = builder(TCNConfig(in_features=5))
    assert isinstance(model, TCNTurnModel)
    with pytest.raises(KeyError):
        ModelFactory("does-not-exist")


def test_param_count_under_budget() -> None:
    """The default-width TCN stays comfortably under the 100k-parameter budget."""
    model = TCNTurnModel(TCNConfig(in_features=29))  # ~feature-bank width
    n_params = model.count_parameters()
    assert n_params < 100_000, f"TCN has {n_params} params, over the 100k budget"
    assert n_params > 0


# --------------------------------------------------------------------------- #
# Group-boundary guarantee.                                                    #
# --------------------------------------------------------------------------- #


def test_window_never_spans_two_groups() -> None:
    """Every emitted window's W bars share a single group id (no asset crossing)."""
    X, labels, groups = _two_asset_frame()
    window = 8
    stats = compute_feature_stats(X, np.arange(len(X), dtype=np.int64))
    dataset = WindowDataset(X, labels, groups, window=window, stats=stats)

    assert len(dataset) > 0, "fixture should yield some valid windows"

    for item in range(len(dataset)):
        window_groups = dataset.window_groups(item)
        assert window_groups.shape[0] == window
        assert np.unique(window_groups).size == 1, (
            f"item {item} spans groups {np.unique(window_groups).tolist()} — "
            "a window crossed the asset boundary"
        )


def test_window_ends_avoid_boundary_rows() -> None:
    """No window ends on a row whose look-back reaches into the previous asset.

    For a window of length ``W`` the first valid end in group 1 is ``n_a + W - 1``;
    every end in ``[n_a, n_a + W - 2]`` would reach back into group 0 and must be
    excluded. The first group's valid ends start at ``W - 1``.
    """
    n_a, n_b = 120, 90
    X, labels, groups = _two_asset_frame(n_a=n_a, n_b=n_b)
    window = 8
    ends = _valid_window_ends(groups, window)

    # No end falls in the forbidden boundary band [n_a, n_a + window - 2].
    forbidden = set(range(n_a, n_a + window - 1))
    assert not (set(ends.tolist()) & forbidden), (
        "a window end fell inside the cross-asset boundary band"
    )
    # The two blocks contribute exactly their interior counts.
    expected = (n_a - (window - 1)) + (n_b - (window - 1))
    assert ends.shape[0] == expected
    # Every window's bars are single-group (cross-checks the end set directly).
    for end in ends.tolist():
        seg = groups[end - window + 1 : end + 1]
        assert np.unique(seg).size == 1


def test_dataset_item_target_is_last_bar_label() -> None:
    """The item target equals the per-side label at the window's LAST bar."""
    X, labels, groups = _two_asset_frame()
    window = 8
    stats = compute_feature_stats(X, np.arange(len(X), dtype=np.int64))
    dataset = WindowDataset(X, labels, groups, window=window, stats=stats)

    # Row 40 is a strong top turn; find the item whose window ends there.
    ends = dataset.end_indices
    pos = int(np.flatnonzero(ends == 40)[0])
    _window, target, weight = dataset[pos]
    # top (index 0) positive, bottom (index 1) negative at row 40.
    assert float(target[0]) == 1.0
    assert float(target[1]) == 0.0
    # Sample-weight convention: 1 + score on the turn, 1 on the non-turn side.
    assert float(weight[0]) == pytest.approx(1.0 + 0.7)
    assert float(weight[1]) == pytest.approx(1.0)


def test_standardization_imputes_nan_to_zero() -> None:
    """Residual NaNs are imputed to 0 (the post-standardization mean)."""
    X, labels, groups = _two_asset_frame(n_features=4)
    X.iloc[10, 0] = np.nan  # inject a missing value inside group 0.
    window = 5
    stats = compute_feature_stats(X, np.arange(len(X), dtype=np.int64))
    dataset = WindowDataset(X, labels, groups, window=window, stats=stats)
    # Every emitted window must be finite (no NaN/inf leaks into a tensor).
    for item in range(len(dataset)):
        win, _t, _w = dataset[item]
        assert torch.isfinite(win).all(), f"item {item} window has non-finite values"
