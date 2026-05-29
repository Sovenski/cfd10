"""Tests for ``cfd10.data_module.schema``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfd10.data_module import CANONICAL_COLS, normalize


def test_volume_renamed_and_lowercased() -> None:
    """A capital ``Volume`` column is renamed to canonical ``volume``."""
    df = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "Volume": [10, 20, 30],
        }
    )
    out = normalize(df)
    assert tuple(out.columns) == CANONICAL_COLS
    assert "Volume" not in out.columns
    assert "volume" in out.columns


def test_sorted_and_deduped_keep_last() -> None:
    """Rows are sorted ascending by time; duplicate timestamps keep the last."""
    df = pd.DataFrame(
        {
            "time": [3, 1, 2, 2],
            "open": [30.0, 10.0, 20.0, 22.0],
            "high": [30.0, 10.0, 20.0, 22.0],
            "low": [30.0, 10.0, 20.0, 22.0],
            "close": [30.0, 10.0, 20.0, 22.0],
            "Volume": [3, 1, 2, 99],
        }
    )
    out = normalize(df)
    assert list(out["time"]) == [1, 2, 3]
    # Duplicate time==2 keeps the LAST occurrence (close/volume from the 22.0 row).
    row2 = out.loc[out["time"] == 2].iloc[0]
    assert row2["close"] == 22.0
    assert row2["volume"] == 99
    # Strictly monotonic after dedup.
    assert out["time"].is_monotonic_increasing
    assert out["time"].is_unique


def test_negative_time_preserved_as_int64() -> None:
    """Negative epoch seconds (e.g. the 1871 S&P epoch) survive as int64."""
    df = pd.DataFrame(
        {
            "time": [-3121407238, -3118988038, -3113717638],
            "open": [4.5, 4.61, 4.86],
            "high": [4.5, 4.61, 4.86],
            "low": [4.5, 4.61, 4.86],
            "close": [4.5, 4.61, 4.86],
            "Volume": [0, 0, 0],
        }
    )
    out = normalize(df)
    assert out["time"].dtype == np.int64
    assert out["time"].iloc[0] == -3121407238
    # OHLCV coerced to float64.
    for col in ("open", "high", "low", "close", "volume"):
        assert out[col].dtype == np.float64


def test_does_not_mutate_input() -> None:
    """``normalize`` does not mutate the caller's DataFrame."""
    df = pd.DataFrame(
        {
            "time": [2, 1],
            "open": [2.0, 1.0],
            "high": [2.0, 1.0],
            "low": [2.0, 1.0],
            "close": [2.0, 1.0],
            "Volume": [2, 1],
        }
    )
    before = df.copy(deep=True)
    _ = normalize(df)
    pd.testing.assert_frame_equal(df, before)
