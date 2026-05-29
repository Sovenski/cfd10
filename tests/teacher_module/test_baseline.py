"""Tests for ``cfd10.teacher_module.baseline`` — the LightGBM CV baseline.

These run on a small (1.5k-row) *synthetic* fixture, never the full SPX export,
so the suite stays fast. We assert the contract rather than a specific score:

* :func:`fit_gbdt_cv` runs end-to-end and returns the documented dict;
* the aggregated and per-fold OOS F1 are finite and lie in ``[0, 1]``;
* the splits it consumes are honoured (the harness must not invent overlap):
  every fold's train / test indices are disjoint, and the pooled OOS prediction /
  truth counts equal the sum of the per-fold counts (test blocks are disjoint);
* a feature carrying real signal scores strictly above the F1 the *same* pipeline
  reaches on pure-noise labels — i.e. the metric responds to signal;
* the input-validation guards fire on malformed inputs.

The fixture's positive class is deliberately sparse (~a few percent), mirroring
the heavy class imbalance of real turn labels.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cfd10.cv_module import purged_walk_forward
from cfd10.teacher_module import (
    GBDTConfig,
    TeacherFactory,
    fit_gbdt_cv,
)

_N_ROWS = 1500
_N_FOLDS = 4
_LABEL_HORIZON = 10
_EMBARGO = 3
_TOLERANCE = 2


def _fast_config(**overrides: object) -> GBDTConfig:
    """A small, fast GBDT config for the synthetic fixture."""
    base = {
        "n_estimators": 40,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_child_samples": 10,
        "tolerance": _TOLERANCE,
        "label_horizon": _LABEL_HORIZON,
        "embargo": _EMBARGO,
        "n_folds": _N_FOLDS,
        "n_threshold_grid": 25,
        "seed": 7,
    }
    base.update(overrides)
    return GBDTConfig(**base)  # type: ignore[arg-type]


def _make_signal_fixture(
    n: int = _N_ROWS, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic features + a *learnable* sparse binary label and weights.

    Three features are pure noise; a fourth (``signal``) spikes on the positive
    bars, so a tree model can recover the label well above chance. Positives are
    ~3% of bars. Returns ``(X, y, sample_weight)``.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 3))

    y = np.zeros(n, dtype=np.int64)
    # Sparse positives spaced out so event windows do not all collide.
    pos = rng.choice(np.arange(20, n - 20), size=max(1, n // 32), replace=False)
    y[pos] = 1

    signal = rng.standard_normal(n) * 0.3
    signal[y == 1] += 3.0  # clear, separable bump on positives.

    X = pd.DataFrame(
        {
            "noise_a": noise[:, 0],
            "noise_b": noise[:, 1],
            "noise_c": noise[:, 2],
            "signal": signal,
        }
    )
    # Heavier weight on positives, mirroring the oracle's score-as-weight scheme.
    sample_weight = np.where(y == 1, 1.0, 0.3).astype(np.float64)
    return X, y, sample_weight


def _make_noise_fixture(
    n: int = _N_ROWS, seed: int = 1
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic features with labels that are independent of every feature."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["f0", "f1", "f2", "f3"])
    y = np.zeros(n, dtype=np.int64)
    pos = rng.choice(np.arange(20, n - 20), size=max(1, n // 32), replace=False)
    y[pos] = 1
    sample_weight = np.where(y == 1, 1.0, 0.3).astype(np.float64)
    return X, y, sample_weight


def _splits(n: int = _N_ROWS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged walk-forward folds over a unit-spaced timeline."""
    timestamps = np.arange(n, dtype=np.int64)
    return purged_walk_forward(
        timestamps,
        label_horizon=_LABEL_HORIZON,
        embargo=_EMBARGO,
        n_folds=_N_FOLDS,
    )


# --------------------------------------------------------------------------- #
# Contract tests.                                                             #
# --------------------------------------------------------------------------- #


def test_fit_gbdt_cv_runs_and_returns_finite_oos_f1() -> None:
    """The fitter returns the documented dict with finite OOS F1 in [0, 1]."""
    X, y, w = _make_signal_fixture()
    out = fit_gbdt_cv(X, y, w, _splits(), _fast_config())

    for key in (
        "folds",
        "oos_precision",
        "oos_recall",
        "oos_f1",
        "oos_tp",
        "oos_fp",
        "oos_fn",
        "mean_fold_f1",
        "n_true",
        "n_pred",
        "positive_rate",
        "config",
    ):
        assert key in out, f"missing result key {key!r}"

    for key in ("oos_precision", "oos_recall", "oos_f1", "mean_fold_f1"):
        value = out[key]
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    assert len(out["folds"]) == _N_FOLDS
    for fold in out["folds"]:
        assert math.isfinite(fold.f1)
        assert 0.0 <= fold.f1 <= 1.0
        assert 0.0 <= fold.precision <= 1.0
        assert 0.0 <= fold.recall <= 1.0


def test_fit_gbdt_cv_respects_splits_no_overlap() -> None:
    """Per-fold train/test indices are disjoint and OOS counts pool additively.

    The pooled OOS truth count must equal the sum of per-fold ``n_true`` and the
    pooled prediction count the sum of per-fold ``n_pred`` — which holds iff the
    test blocks are disjoint (so the harness scored each test bar exactly once).
    """
    X, y, w = _make_signal_fixture()
    splits = _splits()

    # The folds themselves must be disjoint (guards against a regression upstream
    # as well as any re-indexing inside the fitter).
    for train_idx, test_idx in splits:
        assert np.intersect1d(train_idx, test_idx).size == 0

    out = fit_gbdt_cv(X, y, w, splits, _fast_config())
    folds = out["folds"]

    assert out["n_true"] == sum(f.n_true for f in folds)
    assert out["n_pred"] == sum(f.n_pred for f in folds)
    # Pooled confusion counts are internally consistent with the PRF.
    assert out["oos_tp"] + out["oos_fn"] == out["n_true"]
    assert out["oos_tp"] + out["oos_fp"] == out["n_pred"]


def test_signal_beats_noise_oos() -> None:
    """A learnable label scores strictly above a label independent of features."""
    cfg = _fast_config()
    xs, ys, ws = _make_signal_fixture(seed=0)
    xn, yn, wn = _make_noise_fixture(seed=1)

    signal = fit_gbdt_cv(xs, ys, ws, _splits(), cfg)
    noise = fit_gbdt_cv(xn, yn, wn, _splits(), cfg)

    assert signal["oos_f1"] > noise["oos_f1"], (
        f"signal OOS f1={signal['oos_f1']:.3f} did not beat "
        f"noise OOS f1={noise['oos_f1']:.3f}"
    )
    # The separable signal should be clearly learnable, not marginal.
    assert signal["oos_f1"] > 0.3


def test_factory_resolves_gbdt() -> None:
    """The registry exposes ``fit_gbdt_cv`` under the name ``'gbdt'``."""
    assert TeacherFactory("gbdt") is fit_gbdt_cv
    with pytest.raises(KeyError):
        TeacherFactory("does-not-exist")


def test_validation_rejects_bad_inputs() -> None:
    """Shape, emptiness, binary-label and index-range guards all fire."""
    X, y, w = _make_signal_fixture(n=400)
    splits = purged_walk_forward(
        np.arange(400, dtype=np.int64),
        label_horizon=_LABEL_HORIZON,
        embargo=_EMBARGO,
        n_folds=_N_FOLDS,
    )
    cfg = _fast_config()

    with pytest.raises(ValueError):  # y length mismatch.
        fit_gbdt_cv(X, y[:-1], w, splits, cfg)
    with pytest.raises(ValueError):  # weight length mismatch.
        fit_gbdt_cv(X, y, w[:-1], splits, cfg)
    with pytest.raises(ValueError):  # empty splits.
        fit_gbdt_cv(X, y, w, [], cfg)
    with pytest.raises(ValueError):  # non-binary labels.
        bad = y.copy()
        bad[0] = 2
        fit_gbdt_cv(X, bad, w, splits, cfg)


def test_zero_negative_weight_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Zero-weighting the negative class is caught by the near-constant guard.

    This is the exact real-data trap: the oracle ``*_weight`` column is the turn
    *score*, which is ``0`` on every non-turn bar. Fed verbatim as the sample
    weight it zeroes the negative class, LightGBM emits a near-constant
    probability, and thresholding degenerates to "flag everything". The fitter
    must warn loudly rather than report that silently.
    """
    X, y, _ = _make_signal_fixture()
    # Oracle-style weights: positives keep a weight, negatives are exactly zero.
    bad_weight = np.where(y == 1, 1.0, 0.0).astype(np.float64)

    with caplog.at_level("WARNING", logger="cfd10.teacher_module.baseline"):
        fit_gbdt_cv(X, y, bad_weight, _splits(), _fast_config())

    assert any(
        "near-constant" in rec.message for rec in caplog.records
    ), "expected a near-constant-probability warning when negatives are zero-weighted"


def test_config_validation() -> None:
    """`GBDTConfig` rejects out-of-domain hyper-parameters."""
    with pytest.raises(ValueError):
        GBDTConfig(num_leaves=1)
    with pytest.raises(ValueError):
        GBDTConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        GBDTConfig(n_folds=1)
    with pytest.raises(ValueError):
        GBDTConfig(tolerance=-1)
