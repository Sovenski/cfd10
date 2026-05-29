"""Tests for ``cfd10.feature_module.overextension`` — top "tell" features.

These tests pin the *dimensionless, bounded* contract of the overextension /
vol-regime block (the optional top-side tells layered onto the structural
oracle) and the way :func:`cfd10.feature_module.bank.build_feature_matrix` wires
that block in behind ``FeatureConfig.include_overextension``.

Contract pinned here
--------------------
* Analytic ranges hold on real and synthetic data: ``dist_above_sma_z`` in
  ``(-1, 1)``; ``drawdown_from_high`` ``<= 0``; ``realized_vol_pct`` /
  ``up_streak_norm`` / ``vol_of_vol`` in ``[0, 1]``.
* Warm-up bars (no full defining window) are ``NaN``, and the first defined bar
  matches each function's documented index.
* A monotonically (super-linearly) overextended series scores ``dist_above_sma_z``
  near ``+1`` and ``drawdown_from_high`` at ~0 (it keeps printing fresh highs).
* Each window statistic's scalar reference agrees with the vectorized public
  function to ``1e-9``.
* ``build_feature_matrix(FeatureConfig(include_overextension=True))`` appends the
  documented columns deterministically; the default config does not.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from cfd10.data_module import load_csv
from cfd10.feature_module import (
    FeatureConfig,
    build_feature_matrix,
    dist_above_sma_z,
    drawdown_from_high,
    get_overextension_fn,
    realized_vol_pct,
    up_streak_norm,
    vol_of_vol,
)
from cfd10.feature_module.overextension import (
    OVEREXTENSION_FACTORY,
    _dist_above_sma_z_scalar,
    _drawdown_from_high_scalar,
    _up_streak_norm_scalar,
)

_SPX_1D = Path(
    r"C:\Users\kuben\Desktop\Projekte\cfd10\data\raw_v16\SP_SPX, 1D_a20e0.csv"
)


def _random_close(n: int = 400, seed: int = 11) -> np.ndarray:
    """Return a strictly positive random close series (geometric random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.012, size=n)
    return (100.0 * np.exp(np.cumsum(steps))).astype(np.float64)


def _overextended_close(n: int = 300) -> np.ndarray:
    """Return a strictly increasing, super-linearly accelerating up-trend.

    The convex (accelerating) shape pulls the price progressively further above
    its trailing SMA — exactly the overextension the top tells must detect — while
    every bar prints a fresh high.
    """
    k = np.arange(n, dtype=np.float64)
    close = (100.0 * np.exp(0.0003 * k**2)).astype(np.float64)
    return close


# --------------------------------------------------------------------------- #
# Registry contract.                                                          #
# --------------------------------------------------------------------------- #


def test_registry_contains_all_features() -> None:
    """Every documented feature is registered and resolvable through the factory."""
    expected = {
        "dist_above_sma_z": dist_above_sma_z,
        "drawdown_from_high": drawdown_from_high,
        "realized_vol_pct": realized_vol_pct,
        "up_streak_norm": up_streak_norm,
        "vol_of_vol": vol_of_vol,
    }
    assert set(OVEREXTENSION_FACTORY) == set(expected)
    for name, fn in expected.items():
        assert get_overextension_fn(name) is fn


# --------------------------------------------------------------------------- #
# Bounded ranges (analytic contract).                                         #
# --------------------------------------------------------------------------- #


def test_dist_above_sma_z_within_open_unit_interval() -> None:
    """``dist_above_sma_z`` is a tanh, hence strictly inside ``(-1, 1)``."""
    close = _random_close()
    out = dist_above_sma_z(close, length=50, z_win=30)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert finite.min() > -1.0 and finite.max() < 1.0


def test_drawdown_from_high_is_non_positive() -> None:
    """``drawdown_from_high`` is ``<= 0`` everywhere it is defined."""
    close = _random_close()
    out = drawdown_from_high(close, lookback=50)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert finite.max() <= 1e-12  # numerically <= 0 (0 at fresh highs).


def test_unit_interval_features_within_zero_one() -> None:
    """``realized_vol_pct`` / ``up_streak_norm`` / ``vol_of_vol`` stay in ``[0, 1]``."""
    close = _random_close()
    rv = realized_vol_pct(close, win=20, range_len=100)
    streak = up_streak_norm(close, cap=5)
    vov = vol_of_vol(close, win=20)
    for name, arr in (("realized_vol_pct", rv), ("up_streak_norm", streak), ("vol_of_vol", vov)):
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0, f"{name} produced no finite values"
        assert finite.min() >= -1e-12 and finite.max() <= 1.0 + 1e-12, (
            f"{name} escaped [0, 1]: [{finite.min()}, {finite.max()}]"
        )


# --------------------------------------------------------------------------- #
# Warm-up NaN convention.                                                     #
# --------------------------------------------------------------------------- #


def test_dist_above_sma_z_warmup_nan() -> None:
    """Bars before ``max(length, z_win) - 1`` are NaN; the first defined bar is finite."""
    close = _random_close(n=80)
    length, z_win = 20, 30
    out = dist_above_sma_z(close, length=length, z_win=z_win)
    first = max(length, z_win) - 1
    assert np.all(np.isnan(out[:first]))
    assert np.isfinite(out[first])


def test_drawdown_from_high_warmup_nan() -> None:
    """Bars before ``lookback - 1`` are NaN; the first defined bar is finite."""
    close = _random_close(n=60)
    lookback = 25
    out = drawdown_from_high(close, lookback=lookback)
    assert np.all(np.isnan(out[: lookback - 1]))
    assert np.isfinite(out[lookback - 1])


def test_realized_vol_pct_warmup_nan() -> None:
    """The realized-vol warm-up (``i < win``) propagates as NaN through the PIR."""
    close = _random_close(n=80)
    win = 20
    out = realized_vol_pct(close, win=win, range_len=100)
    # First defined realized-vol bar is at index ``win`` (returns 1..win).
    assert np.all(np.isnan(out[:win]))
    assert np.isfinite(out[win])


def test_up_streak_norm_no_multibar_warmup() -> None:
    """``up_streak_norm`` has no multi-bar warm-up; bar 0 is exactly 0.0."""
    close = _random_close(n=40)
    out = up_streak_norm(close, cap=5)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0


# --------------------------------------------------------------------------- #
# Overextended toy series behaviour.                                          #
# --------------------------------------------------------------------------- #


def test_overextended_series_high_dist_and_zero_drawdown() -> None:
    """A super-linear up-trend scores high ``dist_above_sma_z`` and ~0 drawdown.

    On a strictly increasing, accelerating series the close sits many trailing
    standard deviations above its long SMA, so the squashed z-distance saturates
    toward ``+1``; and because every bar is a fresh high the drawdown-from-high is
    essentially ``0`` throughout the defined region.
    """
    close = _overextended_close(n=300)
    assert np.all(np.diff(close) > 0)  # strictly increasing.

    dist = dist_above_sma_z(close, length=100, z_win=50)
    assert dist[-1] > 0.9, f"expected strong overextension, got {dist[-1]}"

    dd = drawdown_from_high(close, lookback=50)
    finite_dd = dd[np.isfinite(dd)]
    # Fresh high on every bar -> drawdown pinned at ~0.
    np.testing.assert_allclose(finite_dd, 0.0, atol=1e-12)


def test_up_streak_norm_saturates_on_monotone_series() -> None:
    """A long monotone up-run drives ``up_streak_norm`` to its 1.0 saturation."""
    close = _overextended_close(n=50)
    cap = 5
    out = up_streak_norm(close, cap=cap)
    # After ``cap`` consecutive up bars the normalised streak is exactly 1.0.
    assert out[-1] == 1.0
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_drawdown_from_high_reads_negative_after_selloff() -> None:
    """After a sharp drop from a high, ``drawdown_from_high`` is strongly negative."""
    up = np.linspace(100.0, 200.0, 60, dtype=np.float64)
    down = np.linspace(200.0, 150.0, 20, dtype=np.float64)[1:]
    close = np.concatenate([up, down]).astype(np.float64)
    out = drawdown_from_high(close, lookback=40)
    # Last bar: high is 200 (still in window), close 150 -> drawdown = -0.25.
    assert math.isclose(out[-1], (150.0 - 200.0) / 200.0, abs_tol=1e-12)
    assert out[-1] < -0.1


# --------------------------------------------------------------------------- #
# Scalar / vectorized parity (pinned to 1e-9).                                #
# --------------------------------------------------------------------------- #


def test_dist_above_sma_z_scalar_parity() -> None:
    """Scalar and vectorized ``dist_above_sma_z`` agree to 1e-9."""
    close = _random_close(n=300, seed=3)
    length, z_win = 50, 30
    vec = dist_above_sma_z(close, length=length, z_win=z_win)
    scalar = np.array(
        [_dist_above_sma_z_scalar(close, length, z_win, i) for i in range(len(close))],
        dtype=np.float64,
    )
    np.testing.assert_allclose(vec, scalar, atol=1e-9, rtol=0.0, equal_nan=True)


def test_drawdown_from_high_scalar_parity() -> None:
    """Scalar and vectorized ``drawdown_from_high`` agree to 1e-9."""
    close = _random_close(n=300, seed=4)
    lookback = 40
    vec = drawdown_from_high(close, lookback=lookback)
    scalar = np.array(
        [_drawdown_from_high_scalar(close, lookback, i) for i in range(len(close))],
        dtype=np.float64,
    )
    np.testing.assert_allclose(vec, scalar, atol=1e-9, rtol=0.0, equal_nan=True)


def test_up_streak_norm_scalar_parity() -> None:
    """Scalar and vectorized ``up_streak_norm`` agree to 1e-9."""
    close = _random_close(n=200, seed=5)
    cap = 7
    vec = up_streak_norm(close, cap=cap)
    scalar = np.array(
        [_up_streak_norm_scalar(close, cap, i) for i in range(len(close))],
        dtype=np.float64,
    )
    np.testing.assert_allclose(vec, scalar, atol=1e-9, rtol=0.0)


# --------------------------------------------------------------------------- #
# Degenerate-input guards.                                                    #
# --------------------------------------------------------------------------- #


def test_flat_window_dist_above_sma_z_is_zero() -> None:
    """A perfectly flat ``z_win`` window (zero std) maps to 0.0, not NaN/inf."""
    close = np.full(40, 5.0, dtype=np.float64)
    out = dist_above_sma_z(close, length=10, z_win=10)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 0.0, atol=1e-12)


def test_flat_series_unit_interval_features_are_half() -> None:
    """On a flat series the PIR-based vol regimes degenerate to 0.5 (flat-window)."""
    close = np.full(120, 42.0, dtype=np.float64)
    rv = realized_vol_pct(close, win=20, range_len=50)
    vov = vol_of_vol(close, win=20)
    for arr in (rv, vov):
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0
        np.testing.assert_allclose(finite, 0.5, atol=1e-12)


# --------------------------------------------------------------------------- #
# build_feature_matrix wiring (optional block).                               #
# --------------------------------------------------------------------------- #


def _expected_oe_names(cfg: FeatureConfig) -> list[str]:
    """Reconstruct the overextension block's column names from the config grid."""
    names: list[str] = []
    for length in cfg.oe_sma_lengths:
        names.append(f"dist_above_sma_z_l{length}")
    for lb in cfg.oe_dd_lookbacks:
        names.append(f"drawdown_from_high_l{lb}")
    names.append("realized_vol_pct")
    names.append("up_streak_norm")
    names.append("vol_of_vol")
    return names


def _load_spx() -> pd.DataFrame:
    """Load the canonical SPX 1D frame used across the data-layer tests."""
    assert _SPX_1D.is_file(), f"missing test fixture: {_SPX_1D}"
    return load_csv(_SPX_1D)


def test_default_config_excludes_overextension() -> None:
    """With the default config (flag off), no overextension column is emitted."""
    df = _load_spx()
    _, names = build_feature_matrix(df, FeatureConfig())
    assert not any(n in names for n in _expected_oe_names(FeatureConfig()))


def test_include_overextension_appends_columns() -> None:
    """``include_overextension=True`` appends exactly the documented columns, in order.

    The block is *appended* after the default blocks, so the baseline columns are
    a strict prefix of the extended ones and the extra suffix equals the expected
    overextension grid expansion.
    """
    df = _load_spx()
    _, base_names = build_feature_matrix(df, FeatureConfig())
    cfg = FeatureConfig(include_overextension=True)
    X, names = build_feature_matrix(df, cfg)

    expected_oe = _expected_oe_names(cfg)
    assert names == base_names + expected_oe
    assert list(X.columns) == names
    assert len(names) == len(set(names)), "feature names must be unique"
    assert (X.dtypes == np.float64).all()
    assert len(X) == len(df)
    assert X.index.equals(df.index)


def test_include_overextension_no_inf_and_warmup_only_nan() -> None:
    """Overextension columns add no inf and confine NaN to the leading warm-up."""
    df = _load_spx()
    cfg = FeatureConfig(include_overextension=True)
    X, names = build_feature_matrix(df, cfg)
    oe = X[_expected_oe_names(cfg)]

    values = oe.to_numpy(dtype=np.float64)
    assert not np.isinf(values).any()

    row_has_nan = oe.isna().any(axis=1).to_numpy()
    assert (~row_has_nan).any(), "expected at least one fully-warmed overextension row"
    first_clean = int(np.argmax(~row_has_nan))
    assert not row_has_nan[first_clean:].any(), (
        "overextension NaN must be confined to the leading warm-up region"
    )


def test_include_overextension_deterministic() -> None:
    """Two independent calls with the flag on produce bit-identical matrices."""
    df = _load_spx()
    cfg = FeatureConfig(include_overextension=True)
    X1, names1 = build_feature_matrix(df, cfg)
    X2, names2 = build_feature_matrix(df, cfg)
    assert names1 == names2
    pd.testing.assert_frame_equal(X1, X2)
