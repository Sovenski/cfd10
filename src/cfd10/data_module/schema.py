"""Canonical OHLCV schema and normalization for cfd10 market data.

The data layer speaks a single canonical column contract so that downstream
feature/oracle code never has to second-guess column names, dtypes, or row
ordering. ``time`` is kept as raw epoch *seconds* in ``int64`` — crucially this
preserves negative timestamps (e.g. ``-3121407238``, the 1871 S&P epoch). We do
**not** parse ``time`` into a datetime here, because pandas/NumPy datetime types
cannot round-trip such early epochs cleanly and Pine parity works off raw bar
ordering, not wall-clock semantics.
"""

from __future__ import annotations

import pandas as pd

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

CANONICAL_COLS: tuple[str, str, str, str, str, str] = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# Source-column aliases mapped onto the canonical names. TradingView exports the
# volume column capitalized ("Volume"); everything else is already lowercase.
_RENAME_MAP: dict[str, str] = {"Volume": "volume"}

_FLOAT_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

__all__ = ["CANONICAL_COLS", "normalize"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical OHLCV frame from a raw market DataFrame.

    The transformation is pure (the input ``df`` is never mutated) and performs,
    in order:

    1. Rename source aliases (``Volume`` -> ``volume``).
    2. Select and order columns to :data:`CANONICAL_COLS`.
    3. Coerce ``time`` to ``int64`` (negative epochs preserved) and OHLCV to
       ``float64``.
    4. Sort ascending by ``time``.
    5. Drop duplicate timestamps, keeping the last occurrence.
    6. Assert strictly monotonic ``time`` after dedup.

    Args:
        df: Raw DataFrame containing at least the OHLCV columns (with ``time``
            and a ``volume`` or ``Volume`` column).

    Returns:
        A new DataFrame with exactly :data:`CANONICAL_COLS`, a fresh
        ``RangeIndex``, and the dtype/ordering guarantees above.

    Raises:
        KeyError: If a required canonical column is missing after renaming.
        AssertionError: If ``time`` is not strictly increasing after dedup.
    """
    work = df.rename(columns=_RENAME_MAP)

    missing = [col for col in CANONICAL_COLS if col not in work.columns]
    if missing:
        raise KeyError(f"normalize: missing required columns {missing}")

    work = work.loc[:, list(CANONICAL_COLS)].copy()

    # int64 for time keeps negative epochs intact; float64 for OHLCV.
    work["time"] = work["time"].astype("int64")
    for col in _FLOAT_COLS:
        work[col] = work[col].astype("float64")

    work = work.sort_values("time", kind="stable")
    work = work.drop_duplicates(subset="time", keep="last")
    work = work.reset_index(drop=True)

    if not work["time"].is_monotonic_increasing or not work["time"].is_unique:
        raise AssertionError("normalize: time is not strictly monotonic after dedup")

    logger.debug("normalize: %d canonical rows", len(work))
    return work
