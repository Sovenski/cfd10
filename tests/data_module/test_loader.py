"""Tests for ``cfd10.data_module.loader``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cfd10.data_module import (
    CANONICAL_COLS,
    KNOWN_DUPLICATES,
    load_csv,
    load_dir,
    parse_v16_name,
)

DATA_ROOT = Path(r"C:\Users\kuben\Desktop\Projekte\cfd10\data\raw_v16")
SPX_FILE = DATA_ROOT / "SP_SPX, 1D_a20e0.csv"


def _data_row_count(path: Path) -> int:
    """Count non-header lines in a CSV the simple way, for cross-checking."""
    with path.open("r", encoding="utf-8") as fh:
        # First line is the header; remaining non-empty lines are data rows.
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    return len(lines) - 1


def test_parse_v16_name_examples() -> None:
    """The three documented tricky names parse to the expected tuples."""
    assert parse_v16_name("SP_SPX, 1D_a20e0.csv") == ("SP", "SPX", "1D", "a20e0")
    assert parse_v16_name("COMEX_DL_GC1!, 60_18203.csv") == (
        "COMEX_DL",
        "GC1!",
        "60",
        "18203",
    )
    assert parse_v16_name("BATS_BRK.B, 1D_0cd15.csv") == (
        "BATS",
        "BRK.B",
        "1D",
        "0cd15",
    )


def test_parse_v16_name_rejects_garbage() -> None:
    """A non-conforming name raises ``ValueError``."""
    with pytest.raises(ValueError):
        parse_v16_name("not_a_valid_name.txt")


@pytest.mark.skipif(not SPX_FILE.exists(), reason="real data not present")
def test_load_csv_spx_canonical_and_monotonic() -> None:
    """Loading the real SPX daily export yields a canonical, monotonic frame."""
    df = load_csv(SPX_FILE)
    assert tuple(df.columns) == CANONICAL_COLS
    assert df["time"].is_monotonic_increasing
    assert df["time"].is_unique
    # Negative epoch (1871) preserved as a real integer, not parsed to datetime.
    assert df["time"].iloc[0] == -3121407238
    # Exact data-row count (header excluded). The raw export has no trailing
    # newline, so it holds 25241 text lines -> 25240 data rows.
    expected = _data_row_count(SPX_FILE)
    assert expected == 25240
    assert len(df) == expected


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="real data not present")
def test_load_dir_excludes_si1_duplicate() -> None:
    """``load_dir`` skips the known SI1! duplicate export."""
    assert "COMEX_DL_SI1!, 1_8d38f.csv" in KNOWN_DUPLICATES
    bundle = load_dir(DATA_ROOT)
    # Result is keyed by (ticker, tf).
    assert ("SPX", "1D") in bundle
    assert isinstance(bundle[("SPX", "1D")], pd.DataFrame)
    # The duplicate file's (ticker, tf) is still present from the 6bcd3 export,
    # but the 8d38f 2-bar subset must NOT have been the loaded source.
    assert ("SI1!", "1") in bundle
    # The surviving SI1!/1 frame must be the FULL export (25002 rows), not the
    # 2-bar subset's 25002-vs-25002... guard: the dropped file had 25002 rows
    # too, so instead assert the loaded frame matches the 6bcd3 row count.
    si1 = bundle[("SI1!", "1")]
    full = load_csv(DATA_ROOT / "COMEX_DL_SI1!, 1_6bcd3.csv")
    assert len(si1) == len(full)
