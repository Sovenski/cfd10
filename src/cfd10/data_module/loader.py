"""Loading TradingView-native ``raw_v16`` CSV exports into canonical frames.

TradingView exports carry their provenance in the filename, e.g.
``"SP_SPX, 1D_a20e0.csv"``. :func:`parse_v16_name` recovers
``(exchange, ticker, timeframe, hash)`` from that convention; :func:`load_csv`
reads one file through :func:`cfd10.data_module.schema.normalize`; and
:func:`load_dir` bulk-loads a directory keyed by ``(ticker, timeframe)`` while
skipping a known duplicate export.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cfd10.data_module.schema import normalize
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

# Filename grammar: ``<exch>_<ticker>, <tf>_<hash>.csv``.
#   - ``exch`` greedily consumes everything up to the LAST underscore before the
#     comma, so multi-segment exchanges ("COMEX_DL", "XETR_DLY") stay intact.
#   - ``ticker`` may contain '!' or '.' (e.g. "GC1!", "BRK.B") but never '_'.
#   - ``tf`` is an intraday minute count (e.g. "60", "240") or the literal "1D".
#   - ``hash`` is TradingView's lowercase hex export suffix.
_V16_NAME_RE: re.Pattern[str] = re.compile(
    r"^(?P<exch>.+)_(?P<ticker>[^_,]+), (?P<tf>\d+|1D)_(?P<hash>[0-9a-f]+)\.csv$"
)

# Files that duplicate another export and must be skipped by ``load_dir``.
# "COMEX_DL_SI1!, 1_8d38f.csv" is a 2-bar-shorter near-duplicate of the full
# "COMEX_DL_SI1!, 1_6bcd3.csv" 1-minute silver export (same start timestamp,
# overlapping bars); keeping both would collide on the ("SI1!", "1") key.
KNOWN_DUPLICATES: frozenset[str] = frozenset({"COMEX_DL_SI1!, 1_8d38f.csv"})

__all__ = ["KNOWN_DUPLICATES", "parse_v16_name", "load_csv", "load_dir"]


def parse_v16_name(fname: str) -> tuple[str, str, str, str]:
    """Parse a ``raw_v16`` filename into its provenance tuple.

    Args:
        fname: A bare filename such as ``"COMEX_DL_GC1!, 60_18203.csv"``. A full
            path is accepted; only its final component is parsed.

    Returns:
        ``(exchange, ticker, timeframe, hash)`` as strings, e.g.
        ``("COMEX_DL", "GC1!", "60", "18203")``.

    Raises:
        ValueError: If ``fname`` does not match the ``raw_v16`` convention.
    """
    name = Path(fname).name
    match = _V16_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"parse_v16_name: unrecognized raw_v16 filename {name!r}")
    return (
        match.group("exch"),
        match.group("ticker"),
        match.group("tf"),
        match.group("hash"),
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a single ``raw_v16`` CSV into a canonical OHLCV frame.

    Args:
        path: Filesystem path to the CSV export.

    Returns:
        A canonical DataFrame (see :func:`cfd10.data_module.schema.normalize`).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"load_csv: no such file {csv_path}")
    raw = pd.read_csv(csv_path)
    return normalize(raw)


def load_dir(root: str | Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Bulk-load every ``raw_v16`` CSV under ``root``, keyed by ``(ticker, tf)``.

    Files listed in :data:`KNOWN_DUPLICATES` are skipped. With the current
    ``raw_v16`` corpus every surviving ``(ticker, timeframe)`` maps to exactly
    one file; a residual collision would raise to surface an unexpected
    duplicate.

    Args:
        root: Directory containing the CSV exports.

    Returns:
        A dict mapping ``(ticker, timeframe)`` to its canonical DataFrame.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If two non-skipped files collide on the same ``(ticker, tf)``.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"load_dir: not a directory {root_path}")

    bundle: dict[tuple[str, str], pd.DataFrame] = {}
    for csv_path in sorted(root_path.glob("*.csv")):
        if csv_path.name in KNOWN_DUPLICATES:
            logger.info("load_dir: skipping known duplicate %s", csv_path.name)
            continue
        _, ticker, tf, _ = parse_v16_name(csv_path.name)
        key = (ticker, tf)
        if key in bundle:
            raise ValueError(
                f"load_dir: unexpected duplicate key {key} from {csv_path.name}; "
                "add it to KNOWN_DUPLICATES if intentional"
            )
        bundle[key] = load_csv(csv_path)

    logger.info("load_dir: loaded %d (ticker, tf) frames from %s", len(bundle), root_path)
    return bundle
