"""Tests for ``cfd10.feature_module.bank`` — the assembled dense feature bank.

These tests exercise :func:`cfd10.feature_module.bank.build_feature_matrix` end
to end on a *real* loaded SPX daily series (the same ``raw_v16`` export the rest
of the data layer is pinned against), plus the registry/factory contract.

Contract pinned here
--------------------
* ``build_feature_matrix`` returns ``(X, feature_names)`` where ``X`` is a
  :class:`pandas.DataFrame` whose columns are exactly ``feature_names`` (same
  order), aligned to the input frame's index.
* The configured grids of scales / periods / methods / lookbacks expand to the
  expected, deterministic set of column names.
* No value is ``+/-inf``.
* ``NaN`` appears **only** in the leading warm-up region: once the matrix has a
  first all-finite row, every subsequent row is all-finite.
* Two independent calls on the same input are bit-identical (determinism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfd10.data_module import load_csv
from cfd10.feature_module import (
    FeatureBankFactory,
    FeatureConfig,
    build_feature_matrix,
    register_feature,
)
from cfd10.feature_module.bank import FEATURE_REGISTRY

_SPX_1D = Path(r"C:\Users\kuben\Desktop\Projekte\cfd10\data\raw_v16\SP_SPX, 1D_a20e0.csv")


def _expected_feature_names(cfg: FeatureConfig) -> list[str]:
    """Independently reconstruct the expected column names from a config.

    Mirrors the documented grid expansion in
    :func:`cfd10.feature_module.bank.build_feature_matrix` so the test fails if
    the naming scheme or grid wiring drifts.
    """
    names: list[str] = []
    for s in cfg.pir_scales:
        names.append(f"pir_s{s}")
    names.append("agree_high")
    names.append("agree_low")
    for p in cfg.er_periods:
        names.append(f"er_dir_p{p}")
        names.append(f"er_abs_p{p}")
    for method in cfg.vola_methods:
        for length in cfg.vola_lengths:
            names.append(f"vola_pos_{method}_l{length}")
    for s in cfg.trend_scales:
        names.append(f"sma_slope_s{s}")
        names.append(f"linreg_slope_s{s}")
    for lb in cfg.mom_lookbacks:
        names.append(f"price_return_L{lb}")
        names.append(f"mom_divergence_L{lb}")
        names.append(f"mom_velocity_L{lb}")
    names.append("gjr_asym")
    names.append("har_vol")
    return names


def _load_spx() -> pd.DataFrame:
    """Load the canonical SPX 1D frame used across the data-layer tests."""
    assert _SPX_1D.is_file(), f"missing test fixture: {_SPX_1D}"
    return load_csv(_SPX_1D)


# --------------------------------------------------------------------------- #
# Registry / factory contract.                                                #
# --------------------------------------------------------------------------- #


def test_registry_factory_roundtrip() -> None:
    """A feature registered via the decorator is resolvable through the factory."""

    @register_feature("unit_test_probe_feature")
    def _probe(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, np.ndarray]:
        return {"unit_test_probe_feature": np.zeros(len(df), dtype=np.float64)}

    try:
        assert "unit_test_probe_feature" in FEATURE_REGISTRY
        assert FeatureBankFactory("unit_test_probe_feature") is _probe
    finally:
        FEATURE_REGISTRY.pop("unit_test_probe_feature", None)


# --------------------------------------------------------------------------- #
# build_feature_matrix on real market data.                                   #
# --------------------------------------------------------------------------- #


def test_build_feature_matrix_columns_match_names() -> None:
    """Returned DataFrame columns equal ``feature_names`` and the expected grid."""
    df = _load_spx()
    cfg = FeatureConfig()
    X, names = build_feature_matrix(df, cfg)

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == names
    assert names == _expected_feature_names(cfg)
    assert len(names) == len(set(names)), "feature names must be unique"
    assert len(X) == len(df)
    assert X.index.equals(df.index)
    assert (X.dtypes == np.float64).all()


def test_build_feature_matrix_no_inf() -> None:
    """No feature value is +/-inf on real data."""
    df = _load_spx()
    X, _ = build_feature_matrix(df, FeatureConfig())
    values = X.to_numpy(dtype=np.float64)
    assert not np.isinf(values).any(), "feature matrix must not contain infinities"


def test_build_feature_matrix_nan_only_in_warmup() -> None:
    """NaN appears only in the leading warm-up band, never after it clears."""
    df = _load_spx()
    X, _ = build_feature_matrix(df, FeatureConfig())
    row_has_nan = X.isna().any(axis=1).to_numpy()

    # There must be at least one fully-finite row on a multi-decade SPX series.
    assert (~row_has_nan).any(), "expected at least one fully-warmed row"
    first_clean = int(np.argmax(~row_has_nan))
    # From the first clean row onward, no row may reintroduce a NaN.
    assert not row_has_nan[first_clean:].any(), (
        "NaN must be confined to the leading warm-up region; found a NaN at or "
        f"after the first clean row (index {first_clean})"
    )
    # Sanity: the warm-up band is bounded well within the series length.
    assert first_clean < len(df), "series never fully warms up"


def test_build_feature_matrix_deterministic() -> None:
    """Two independent calls produce bit-identical matrices and names."""
    df = _load_spx()
    cfg = FeatureConfig()
    X1, names1 = build_feature_matrix(df, cfg)
    X2, names2 = build_feature_matrix(df, cfg)

    assert names1 == names2
    pd.testing.assert_frame_equal(X1, X2)


def test_build_feature_matrix_bounded_features_in_range() -> None:
    """Bounded/dimensionless features stay inside their analytic ranges.

    PIR, agreement fractions and volatility-position live in ``[0, 1]``; the
    GJR/HAR asymmetries and directional ER live in ``[-1, 1]``; absolute ER in
    ``[0, 1]``. This guards against a mis-wired feature leaking out of range.
    """
    df = _load_spx()
    cfg = FeatureConfig()
    X, _ = build_feature_matrix(df, cfg)

    unit_interval_prefixes = ("pir_s", "agree_", "vola_pos_", "er_abs_")
    signed_unit = ("gjr_asym", "har_vol")

    for col in X.columns:
        finite = X[col].to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        if col.startswith(unit_interval_prefixes):
            assert finite.min() >= -1e-9 and finite.max() <= 1.0 + 1e-9, (
                f"{col} escaped [0, 1]: [{finite.min()}, {finite.max()}]"
            )
        elif col in signed_unit or col.startswith("er_dir_"):
            assert finite.min() >= -1.0 - 1e-9 and finite.max() <= 1.0 + 1e-9, (
                f"{col} escaped [-1, 1]: [{finite.min()}, {finite.max()}]"
            )
