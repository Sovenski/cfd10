"""Multi-asset *pooling*: stack per-asset feature/label tables into one corpus.

The single-asset turn oracle is structurally sparse — SPX alone yields only a
few dozen strong turns, far too few to train a discriminative detector. The
core design bet of cfd10 is that **pooling many daily instruments** multiplies
the positive count while the features stay dimensionless/bounded (so a turn on
gold and a turn on SPX live in the same feature space).

This module realises that bet. :func:`build_pooled_dataset` loads every daily
``raw_v16`` instrument, builds its dense feature bank and oracle labels
**independently** (never across the concat boundary, so no feature leaks from
one asset's tail into another's head), drops warm-up ``NaN`` rows per asset, and
stacks the survivors into one :class:`PooledDataset`. Each row carries an integer
*group id* (its asset), and :attr:`PooledDataset.asset_names` maps a group id back
to its ticker, so a single-asset (e.g. SPX) subset is recoverable by a boolean
mask.

The pooled time axis is what :func:`cfd10.cv_module.purged_walk_forward` splits
on (with ``pooled_groups`` so purge/embargo respect each asset's own bar clock).

Public API
----------
:class:`PooledDataset` — the stacked corpus (features, labels, timestamps, group
ids, asset names) with :meth:`PooledDataset.subset_mask` /
:meth:`PooledDataset.group_of` helpers. :func:`build_pooled_dataset` — assemble
it from a ``raw_v16`` directory. :data:`REPRESENTATIVE_DAILY` — the curated
liquid subset used when a caller caps the corpus size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfd10.data_module.loader import load_csv, parse_v16_name
from cfd10.feature_module import FeatureConfig, build_feature_matrix
from cfd10.label_module import OracleConfig, label_turns
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "PooledDataset",
    "build_pooled_dataset",
    "REPRESENTATIVE_DAILY",
    "DEFAULT_TIMEFRAME",
]

DEFAULT_TIMEFRAME: str = "1D"

# A curated, liquid spread of daily instruments (index futures, metals, FX, and
# a range of mega-caps incl. the structural anchor SPX). Used only when a caller
# caps the corpus via ``max_assets`` / the >3-minute efficiency guard; the full
# corpus build is fast enough that this is rarely needed. SPX, NDX, DAX and GC1!
# are listed first so a head-slice always keeps the anchors.
REPRESENTATIVE_DAILY: tuple[str, ...] = (
    "SPX",
    "NDX",
    "DAX",
    "GC1!",  # gold future
    "SI1!",  # silver future
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",  # not always present; tolerated if missing
    "TSLA",
    "JPM",
    "JNJ",
    "XOM",
    "KO",
    "PG",
    "HD",
    "EURUSD",
    "GBPUSD",
)

# A full-corpus build above this wall-clock budget triggers the documented
# fall-back to ``REPRESENTATIVE_DAILY`` (see the task brief). Measured per-asset
# cost is ~0.1-0.4 s, so 45 assets sit comfortably under this; the guard only
# fires on pathological slowdowns.
_SLOW_BUILD_SECONDS: float = 180.0


@dataclass(frozen=True)
class PooledDataset:
    """A multi-asset corpus: per-asset feature/label tables stacked row-wise.

    Every row belongs to exactly one asset, identified by its integer *group id*
    ``groups[i]``; :attr:`asset_names` maps that id to a ticker
    (``asset_names[groups[i]]``). Features and labels were computed *per asset*
    before stacking, so no row's features depend on another asset's bars.

    Attributes:
        X: Stacked dense feature matrix with a contiguous ``RangeIndex``; warm-up
            ``NaN`` rows have already been dropped per asset. Split indices from
            the CV layer address its rows directly.
        labels: Oracle label frame (:data:`cfd10.label_module.LABEL_COLUMNS`)
            row-aligned to ``X``.
        timestamps: Per-row epoch ``int64`` timestamps aligned to ``X``.
        groups: Per-row integer asset id aligned to ``X`` (``0 .. n_assets-1``),
            i.e. the per-row asset index and the CV ``pooled_groups`` argument.
        asset_names: Ticker per group id; ``asset_names[g]`` is group ``g``'s
            symbol. Length equals the number of pooled assets.
    """

    X: pd.DataFrame
    labels: pd.DataFrame
    timestamps: NDArray[np.int64]
    groups: NDArray[np.int64]
    asset_names: list[str] = field(default_factory=list)

    @property
    def n_assets(self) -> int:
        """Number of pooled assets (== ``len(asset_names)``)."""
        return len(self.asset_names)

    @property
    def n_rows(self) -> int:
        """Total pooled rows after warm-up removal."""
        return int(self.X.shape[0])

    def group_of(self, ticker: str) -> int:
        """Return the integer group id for ``ticker``.

        Args:
            ticker: An asset symbol present in :attr:`asset_names`.

        Returns:
            The group id ``g`` such that ``asset_names[g] == ticker``.

        Raises:
            KeyError: If ``ticker`` is not in the pool.
        """
        try:
            return self.asset_names.index(ticker)
        except ValueError as exc:
            known = ", ".join(self.asset_names) or "<empty>"
            raise KeyError(
                f"PooledDataset.group_of: {ticker!r} not pooled; known: {known}"
            ) from exc

    def subset_mask(self, ticker: str) -> NDArray[np.bool_]:
        """Boolean row mask selecting only ``ticker``'s rows in the pool.

        Args:
            ticker: An asset symbol present in :attr:`asset_names`.

        Returns:
            A boolean array over pooled rows, ``True`` where the row is
            ``ticker``'s. Use it to extract a single-asset OOS slice, e.g.
            ``X.loc[mask]``.

        Raises:
            KeyError: If ``ticker`` is not in the pool.
        """
        return self.groups == self.group_of(ticker)

    def counts_per_group(self) -> dict[str, int]:
        """Return ``{ticker: n_rows}`` for every pooled asset (in group order)."""
        return {
            name: int((self.groups == g).sum())
            for g, name in enumerate(self.asset_names)
        }


# --------------------------------------------------------------------------- #
# Per-asset block + file discovery.                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _AssetBlock:
    """One asset's warm-up-free feature/label/timestamp block (pre-stack)."""

    ticker: str
    X: pd.DataFrame
    labels: pd.DataFrame
    timestamps: NDArray[np.int64]
    n_raw: int
    n_dropped: int


def _discover_daily_files(root: Path, timeframe: str) -> dict[str, Path]:
    """Map ``ticker -> path`` for every ``raw_v16`` file on ``timeframe``.

    On the rare duplicate ticker (e.g. two share classes resolving to the same
    symbol) the first path in sorted order wins; this is deterministic and the
    skipped file is logged.

    Args:
        root: Directory of ``raw_v16`` CSV exports.
        timeframe: Timeframe token to keep (e.g. ``"1D"``).

    Returns:
        A dict mapping each ticker to its CSV path on ``timeframe``.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"build_pooled_dataset: not a directory {root}")

    found: dict[str, Path] = {}
    for csv_path in sorted(root.glob("*.csv")):
        try:
            _, ticker, tf, _ = parse_v16_name(csv_path.name)
        except ValueError:
            logger.warning("pooling: skipping unparseable filename %s", csv_path.name)
            continue
        if tf != timeframe:
            continue
        if ticker in found:
            logger.warning(
                "pooling: duplicate ticker %r (%s); keeping %s",
                ticker,
                csv_path.name,
                found[ticker].name,
            )
            continue
        found[ticker] = csv_path
    return found


def _select_tickers(
    available: dict[str, Path],
    tickers: list[str] | None,
    max_assets: int | None,
    issues: list[str],
) -> list[str]:
    """Resolve which tickers to pool, honouring an explicit list / size cap.

    Args:
        available: Discovered ``ticker -> path`` map on the timeframe.
        tickers: Explicit ticker whitelist, or ``None`` for "all available".
        max_assets: Optional hard cap on the asset count; when the available set
            exceeds it, the curated :data:`REPRESENTATIVE_DAILY` head is used
            (always keeping the SPX/NDX/DAX/GC1! anchors) and a note is appended
            to ``issues``.
        issues: Mutable list collecting human-readable subset notes.

    Returns:
        The selected tickers in a deterministic order.
    """
    if tickers is not None:
        missing = [t for t in tickers if t not in available]
        if missing:
            issues.append(f"requested tickers absent on timeframe: {missing}")
        chosen = [t for t in tickers if t in available]
        return chosen

    all_tickers = sorted(available)
    if max_assets is None or len(all_tickers) <= max_assets:
        return all_tickers

    # Cap exceeded: fall back to the curated liquid subset (anchors first), then
    # backfill with any other available tickers up to the cap.
    rep = [t for t in REPRESENTATIVE_DAILY if t in available]
    extra = [t for t in all_tickers if t not in rep]
    chosen = (rep + extra)[:max_assets]
    issues.append(
        f"corpus capped to {max_assets} representative assets "
        f"(of {len(all_tickers)} available); using {chosen}"
    )
    return chosen


def _build_asset_block(
    ticker: str,
    path: Path,
    feature_cfg: FeatureConfig,
    oracle_cfg: OracleConfig,
) -> _AssetBlock | None:
    """Load one asset and build its warm-up-free feature/label block.

    Features and labels are computed on this asset's frame *alone* — the rolling
    windows never see another instrument — then every row carrying a feature
    ``NaN`` (the warm-up prefix) is dropped and the survivors re-indexed.

    Args:
        ticker: The asset symbol (for logging / tagging).
        path: CSV path for the asset on the chosen timeframe.
        feature_cfg: Dense feature-bank configuration.
        oracle_cfg: Turn-oracle configuration.

    Returns:
        The asset's :class:`_AssetBlock`, or ``None`` if no rows survive warm-up
        removal (too-short series).
    """
    df = load_csv(path)
    X, _names = build_feature_matrix(df, feature_cfg)
    labels = label_turns(df, oracle_cfg)

    valid = ~X.isna().any(axis=1).to_numpy()
    n_dropped = int((~valid).sum())
    if not valid.any():
        logger.warning("pooling: %s has no warm-up-free rows; skipping", ticker)
        return None

    X_valid = X.loc[valid].reset_index(drop=True)
    labels_valid = labels.loc[valid].reset_index(drop=True)
    timestamps = df.loc[valid, "time"].to_numpy(dtype=np.int64)

    return _AssetBlock(
        ticker=ticker,
        X=X_valid,
        labels=labels_valid,
        timestamps=timestamps,
        n_raw=int(len(df)),
        n_dropped=n_dropped,
    )


# --------------------------------------------------------------------------- #
# Public API.                                                                  #
# --------------------------------------------------------------------------- #


def build_pooled_dataset(
    root: str | Path,
    feature_cfg: FeatureConfig,
    oracle_cfg: OracleConfig,
    timeframe: str = DEFAULT_TIMEFRAME,
    tickers: list[str] | None = None,
    max_assets: int | None = None,
) -> tuple[PooledDataset, list[str]]:
    """Stack per-asset feature/label tables into one pooled training corpus.

    For each daily instrument under ``root`` (filtered to ``timeframe``): load
    the canonical frame, build the dense feature bank, label oracle turns, drop
    warm-up ``NaN`` rows, and tag the survivors with an integer group id (one per
    asset). The per-asset blocks are then concatenated onto a single contiguous
    row axis. **Features and labels are always computed within one asset** — the
    concatenation happens only after each block is finished, so no rolling window
    or forward oracle peek crosses an asset boundary.

    Args:
        root: Directory of ``raw_v16`` CSV exports (read-only).
        feature_cfg: Dense feature-bank configuration.
        oracle_cfg: Turn-oracle configuration (the supervision signal).
        timeframe: Timeframe token to pool (default ``"1D"`` — daily).
        tickers: Optional explicit ticker whitelist; ``None`` pools every
            available instrument on ``timeframe``. Missing requests are noted in
            the returned issues.
        max_assets: Optional cap on the number of pooled assets. When exceeded
            (and ``tickers`` is ``None``) the curated :data:`REPRESENTATIVE_DAILY`
            subset is used and noted in the issues.

    Returns:
        ``(dataset, issues)`` where ``dataset`` is the :class:`PooledDataset` and
        ``issues`` is a list of human-readable notes (empty when the full
        requested corpus was built cleanly).

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If no instrument survives (e.g. an empty/over-filtered root).
    """
    root_path = Path(root)
    issues: list[str] = []

    available = _discover_daily_files(root_path, timeframe)
    if not available:
        raise ValueError(
            f"build_pooled_dataset: no {timeframe!r} files under {root_path}"
        )

    selected = _select_tickers(available, tickers, max_assets, issues)
    if not selected:
        raise ValueError(
            f"build_pooled_dataset: no tickers selected on {timeframe!r} "
            f"(requested={tickers})"
        )

    logger.info(
        "build_pooled_dataset: pooling %d %s assets from %s",
        len(selected),
        timeframe,
        root_path,
    )

    blocks: list[_AssetBlock] = []
    asset_names: list[str] = []
    start = time.perf_counter()
    slow_noted = False
    for ticker in selected:
        block = _build_asset_block(ticker, available[ticker], feature_cfg, oracle_cfg)
        if block is None:
            issues.append(f"{ticker}: dropped (no warm-up-free rows)")
            continue
        blocks.append(block)
        asset_names.append(ticker)

        elapsed = time.perf_counter() - start
        if elapsed > _SLOW_BUILD_SECONDS and not slow_noted:
            slow_noted = True
            issues.append(
                f"build exceeded {_SLOW_BUILD_SECONDS:.0f}s after "
                f"{len(asset_names)} assets; consider passing max_assets"
            )
            logger.warning(
                "build_pooled_dataset: slow build (%.0fs, %d assets so far)",
                elapsed,
                len(asset_names),
            )

    if not blocks:
        raise ValueError("build_pooled_dataset: every selected asset was dropped")

    # Concatenate the finished per-asset blocks. ``ignore_index`` rebuilds one
    # contiguous RangeIndex so CV split indices address pooled rows directly.
    X = pd.concat([b.X for b in blocks], ignore_index=True)
    labels = pd.concat([b.labels for b in blocks], ignore_index=True)
    timestamps = np.concatenate([b.timestamps for b in blocks]).astype(
        np.int64, copy=False
    )
    groups = np.concatenate(
        [np.full(b.X.shape[0], g, dtype=np.int64) for g, b in enumerate(blocks)]
    )

    logger.info(
        "build_pooled_dataset: pooled %d assets -> %d rows (%d feature cols); "
        "%d total warm-up rows dropped",
        len(asset_names),
        X.shape[0],
        X.shape[1],
        sum(b.n_dropped for b in blocks),
    )

    dataset = PooledDataset(
        X=X,
        labels=labels,
        timestamps=timestamps,
        groups=groups,
        asset_names=asset_names,
    )
    return dataset, issues
