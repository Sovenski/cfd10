"""Tests for ``cfd10.data_module.resample``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfd10.data_module import resample_ohlcv


def _hourly_bars() -> pd.DataFrame:
    """Four consecutive 1-hour bars starting at 2024-01-01 00:00:00 UTC."""
    base = 1704067200  # 2024-01-01 00:00:00 UTC, in epoch seconds.
    hour = 3600
    return pd.DataFrame(
        {
            "time": [base + i * hour for i in range(4)],
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.5, 11.8, 12.2, 14.0],
            "low": [9.5, 10.2, 11.0, 12.7],
            "close": [11.0, 12.0, 13.0, 13.5],
            "volume": [100.0, 200.0, 300.0, 400.0],
        }
    )


def test_four_1h_bars_into_one_4h_bar() -> None:
    """OHLCV aggregation: open=first, high=max, low=min, close=last, vol=sum."""
    df = _hourly_bars()
    out = resample_ohlcv(df, "4h")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 10.0  # first
    assert row["high"] == 14.0  # max
    assert row["low"] == 9.5  # min
    assert row["close"] == 13.5  # last
    assert row["volume"] == 1000.0  # sum
    # The aggregated bar is stamped at the window open (2024-01-01 00:00 UTC).
    assert out["time"].iloc[0] == 1704067200
    assert tuple(out.columns) == ("time", "open", "high", "low", "close", "volume")


def test_resample_preserves_dtypes_and_does_not_mutate() -> None:
    """Output dtypes are canonical and the input frame is untouched."""
    df = _hourly_bars()
    before = df.copy(deep=True)
    out = resample_ohlcv(df, "2h")
    # Two 2h windows from four 1h bars.
    assert len(out) == 2
    assert out["time"].dtype == np.int64
    for col in ("open", "high", "low", "close", "volume"):
        assert out[col].dtype == np.float64
    pd.testing.assert_frame_equal(df, before)
