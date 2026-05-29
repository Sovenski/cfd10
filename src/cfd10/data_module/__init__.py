"""cfd10 data layer: canonical OHLCV schema, CSV loading, and resampling.

Public API:
    - :data:`CANONICAL_COLS` / :func:`normalize` — the canonical column contract.
    - :func:`parse_v16_name` / :func:`load_csv` / :func:`load_dir` /
      :data:`KNOWN_DUPLICATES` — TradingView ``raw_v16`` ingestion.
    - :func:`resample_ohlcv` — higher-timeframe aggregation.
    - :class:`PooledDataset` / :func:`build_pooled_dataset` — multi-asset pooling.
"""

from __future__ import annotations

from cfd10.data_module.loader import (
    KNOWN_DUPLICATES,
    load_csv,
    load_dir,
    parse_v16_name,
)
from cfd10.data_module.pooling import (
    DEFAULT_TIMEFRAME,
    REPRESENTATIVE_DAILY,
    PooledDataset,
    build_pooled_dataset,
)
from cfd10.data_module.resample import resample_ohlcv
from cfd10.data_module.schema import CANONICAL_COLS, normalize

__all__ = [
    "CANONICAL_COLS",
    "normalize",
    "parse_v16_name",
    "load_csv",
    "load_dir",
    "KNOWN_DUPLICATES",
    "resample_ohlcv",
    "PooledDataset",
    "build_pooled_dataset",
    "REPRESENTATIVE_DAILY",
    "DEFAULT_TIMEFRAME",
]
