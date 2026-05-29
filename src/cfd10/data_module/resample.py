"""Time-based OHLCV resampling onto a UTC ``DatetimeIndex``.

Higher-timeframe bars are built from a lower-timeframe canonical frame with the
standard aggregation: ``open`` = first, ``high`` = max, ``low`` = min,
``close`` = last, ``volume`` = sum. The ``time`` column (epoch seconds) is
interpreted as UTC for the duration of the resample and stamped back at each
window's left edge (its open), matching TradingView's bar-open convention.

Resampling assumes **non-negative** intraday timestamps: pandas datetime types
cannot represent the negative pre-1970 epochs that daily history may carry, and
resampling is only ever applied to non-negative intraday data here.
"""

from __future__ import annotations

import pandas as pd

from cfd10.data_module.schema import CANONICAL_COLS
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

# Per-column reduction applied within each resample window.
_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

__all__ = ["resample_ohlcv"]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate a canonical OHLCV frame to a coarser timeframe.

    Args:
        df: Canonical OHLCV frame (columns :data:`CANONICAL_COLS`) with
            non-negative epoch-second ``time`` values.
        rule: A pandas offset alias for the target window, e.g. ``"4h"`` or
            ``"1D"``.

    Returns:
        A new canonical frame with one row per non-empty window, ``time`` stamped
        at each window's open (epoch seconds, ``int64``) and OHLCV in
        ``float64``. The input ``df`` is not mutated.

    Raises:
        ValueError: If any ``time`` value is negative (unsupported for datetime
            resampling).
    """
    if (df["time"] < 0).any():
        raise ValueError("resample_ohlcv: negative timestamps are unsupported")

    index = pd.to_datetime(df["time"].to_numpy(), unit="s", utc=True)
    indexed = df.loc[:, list(_AGG.keys())].copy()
    indexed.index = pd.DatetimeIndex(index, name="time")

    resampled = indexed.resample(rule, label="left", closed="left").agg(_AGG)
    # Drop windows that contained no source bars (gaps/weekends).
    resampled = resampled.dropna(how="all")

    out = resampled.reset_index()
    # Convert the window-open datetime back to epoch SECONDS as int64,
    # independent of the datetime resolution (pandas may use s, ms, or ns).
    epoch_ns = out["time"].astype("datetime64[ns, UTC]").astype("int64")
    out["time"] = (epoch_ns // 1_000_000_000).astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col].astype("float64")

    out = out.loc[:, list(CANONICAL_COLS)]
    logger.debug("resample_ohlcv: rule=%s -> %d rows", rule, len(out))
    return out
