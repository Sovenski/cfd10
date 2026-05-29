"""Tests for ``cfd10.data_module.pooling`` (multi-asset stacking).

These run on the real ``raw_v16`` corpus but only touch 2-3 small daily
instruments, so they stay fast and are skipped cleanly when the data is absent.
The contract under test is the one the pooled baseline depends on:

* group ids are dense ``0 .. n_assets-1`` and align to ``asset_names``;
* each asset's row count in the pool equals its own standalone warm-up-free
  length (group boundaries align with asset lengths);
* **no cross-asset feature contamination** — an asset's rows in the pool are
  byte-identical to building that asset alone (so a rolling window never crossed
  the concat boundary), and the first pooled row of every asset is ``NaN``-free;
* a single-asset (SPX) subset is recoverable via the row mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfd10.data_module import (
    build_pooled_dataset,
    load_csv,
)
from cfd10.data_module.loader import parse_v16_name
from cfd10.feature_module import FeatureConfig, build_feature_matrix
from cfd10.label_module import LABEL_COLUMNS, OracleConfig, label_turns

DATA_ROOT = Path(r"C:\Users\kuben\Desktop\Projekte\cfd10\data\raw_v16")

# A small, fast 3-asset pool: the structural anchor plus a mega-cap and gold.
_TICKERS = ["SPX", "AAPL", "GC1!"]

# The famous-lows-validated STRUCTURAL oracle (kept identical to the pipeline).
_ORACLE_CFG = OracleConfig(
    scale_nest=(20, 50, 100, 200),
    weight_curve="linear",
    drawdown_pct=0.10,
    horizon=60,
    tau_strong=0.60,
    tau_regular=0.30,
)

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="real raw_v16 data not present"
)


def _ticker_to_path() -> dict[str, Path]:
    """Map the three test tickers to their daily CSV paths under the corpus."""
    paths: dict[str, Path] = {}
    for csv_path in DATA_ROOT.glob("*.csv"):
        try:
            _, ticker, tf, _ = parse_v16_name(csv_path.name)
        except ValueError:
            continue
        if tf == "1D" and ticker in _TICKERS:
            paths[ticker] = csv_path
    return paths


def _standalone_block(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Reproduce one asset's warm-up-free (X, labels, timestamps) in isolation."""
    df = load_csv(path)
    X, _ = build_feature_matrix(df, FeatureConfig())
    labels = label_turns(df, _ORACLE_CFG)
    valid = ~X.isna().any(axis=1).to_numpy()
    X_valid = X.loc[valid].reset_index(drop=True)
    labels_valid = labels.loc[valid].reset_index(drop=True)
    ts = df.loc[valid, "time"].to_numpy(dtype=np.int64)
    return X_valid, labels_valid, ts


@pytest.fixture(scope="module")
def pooled():
    """Build the 3-asset pool once for the module."""
    paths = _ticker_to_path()
    present = [t for t in _TICKERS if t in paths]
    if len(present) < 2:
        pytest.skip(f"need >=2 of {_TICKERS}; found {present}")
    dataset, issues = build_pooled_dataset(
        DATA_ROOT,
        feature_cfg=FeatureConfig(),
        oracle_cfg=_ORACLE_CFG,
        timeframe="1D",
        tickers=present,
    )
    return dataset, issues, present, paths


def test_group_ids_dense_and_aligned(pooled) -> None:
    """Group ids are a dense 0..n-1 range aligned to ``asset_names``."""
    dataset, _issues, present, _paths = pooled
    assert dataset.asset_names == present  # order preserved from the request
    assert dataset.n_assets == len(present)
    uniq = np.unique(dataset.groups)
    assert uniq.tolist() == list(range(len(present)))
    # group_of round-trips against asset_names.
    for g, name in enumerate(dataset.asset_names):
        assert dataset.group_of(name) == g


def test_group_blocks_are_contiguous_and_sized_by_asset(pooled) -> None:
    """Each group occupies one contiguous block sized to its standalone length."""
    dataset, _issues, present, paths = pooled
    # Groups appear as contiguous runs in request order (concat preserves it).
    assert np.all(np.diff(dataset.groups) >= 0), "group ids are not non-decreasing"

    counts = dataset.counts_per_group()
    offset = 0
    for g, ticker in enumerate(present):
        X_solo, _labels_solo, _ts_solo = _standalone_block(paths[ticker])
        n_solo = len(X_solo)
        # Row count in the pool == the asset's own warm-up-free length.
        assert counts[ticker] == n_solo
        block = dataset.groups[offset : offset + n_solo]
        assert np.all(block == g), f"{ticker} block not contiguous at {offset}"
        offset += n_solo
    assert offset == dataset.n_rows  # blocks tile the pool exactly


def test_no_cross_asset_feature_contamination(pooled) -> None:
    """An asset's pooled rows equal its standalone build (no boundary leakage)."""
    dataset, _issues, present, paths = pooled
    feat_cols = list(dataset.X.columns)
    for ticker in present:
        mask = dataset.subset_mask(ticker)
        X_pool = dataset.X.loc[mask].reset_index(drop=True)
        ts_pool = dataset.timestamps[mask]

        X_solo, labels_solo, ts_solo = _standalone_block(paths[ticker])

        # Features are byte-identical: a rolling window that had reached across
        # the concat boundary would perturb the head/tail rows here.
        pd.testing.assert_frame_equal(X_pool[feat_cols], X_solo[feat_cols])
        np.testing.assert_array_equal(ts_pool, ts_solo)

        # First pooled row of the asset is fully NaN-free (warm-up dropped).
        assert not X_pool.iloc[0].isna().any()

        # Labels carry through unchanged with the documented schema.
        labels_pool = dataset.labels.loc[mask].reset_index(drop=True)
        assert tuple(labels_pool.columns) == LABEL_COLUMNS
        pd.testing.assert_frame_equal(labels_pool, labels_solo)


def test_no_nan_anywhere_in_pool(pooled) -> None:
    """The stacked feature matrix has no warm-up NaNs left (every asset cleaned)."""
    dataset, _issues, _present, _paths = pooled
    assert not dataset.X.isna().to_numpy().any()
    assert dataset.timestamps.shape[0] == dataset.n_rows
    assert dataset.groups.shape[0] == dataset.n_rows


def test_spx_subset_extractable(pooled) -> None:
    """The SPX rows are recoverable by mask and match a standalone SPX build."""
    dataset, _issues, present, paths = pooled
    if "SPX" not in present:
        pytest.skip("SPX not in the available test pool")
    mask = dataset.subset_mask("SPX")
    assert mask.sum() > 0
    # Every selected row really belongs to SPX's group.
    assert np.all(dataset.groups[mask] == dataset.group_of("SPX"))

    X_spx, _labels, _ts = _standalone_block(paths["SPX"])
    assert int(mask.sum()) == len(X_spx)


def test_explicit_tickers_filter_two_assets() -> None:
    """Requesting exactly two tickers yields a clean two-group pool."""
    paths = _ticker_to_path()
    pair = [t for t in ["SPX", "GC1!"] if t in paths]
    if len(pair) < 2:
        pytest.skip(f"need SPX and GC1!; found {pair}")
    dataset, issues = build_pooled_dataset(
        DATA_ROOT,
        feature_cfg=FeatureConfig(),
        oracle_cfg=_ORACLE_CFG,
        timeframe="1D",
        tickers=pair,
    )
    assert dataset.n_assets == 2
    assert set(dataset.asset_names) == set(pair)
    assert issues == []  # both present, nothing capped
