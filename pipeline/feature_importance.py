"""Per-side GBDT GAIN feature importances on the multi-asset pool ("what succeeded").

Run with::

    uv run python pipeline/feature_importance.py

The pooled GBDT (see :mod:`pipeline.fit_pooled`) is the current best turn detector.
Out-of-sample event PRF tells us *how well* it ranks pivots, but not *which* tells
carry the signal. This script answers that: for **each side** (HIGH = top turns,
LOW = bottom turns) it fits **one** class-balanced LightGBM on **all** pooled rows
(no CV, no threshold — a single full-data fit, which is the standard, honest way to
read global feature importances) and dumps the LightGBM **GAIN** importances,
sorted descending, to ``outputs/feature_importance_{low,high}.md``. The top tells
per side are also printed.

This is a descriptive diagnostic, not an OOS score: importances are read on the
whole corpus on purpose, so they summarise what the detector leans on across every
asset and era rather than the idiosyncrasies of one fold.

The label and sample-weight conventions match the pooled baseline exactly:
tier in ``{strong, regular}`` -> ``y = 1``; ``w = where(y == 1, 1 + oracle_score, 1)``
with the oracle ``*_weight`` (n-)score; and ``scale_pos_weight = n_neg / n_pos`` per
side so the structurally rare turns drive the tree splits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np

from cfd10.data_module import PooledDataset, build_pooled_dataset
from cfd10.feature_module import FeatureConfig
from cfd10.label_module import OracleConfig
from cfd10.teacher_module import GBDTConfig
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DATA_ROOT: Path = _REPO_ROOT / "data" / "raw_v16"
_OUT_DIR: Path = _REPO_ROOT / "outputs"

_SEED: int = 42
_TIMEFRAME: str = "1D"
_TOP_N: int = 12

# The famous-lows-validated STRUCTURAL oracle (matches pipeline.fit_pooled): the
# user's weighted multi-scale pivot oracle, big-scale turns weighted heavily.
_ORACLE_CFG = OracleConfig(
    scale_nest=(20, 50, 100, 200),
    weight_curve="linear",
    drawdown_pct=0.10,
    horizon=60,
    tau_strong=0.60,
    tau_regular=0.30,
)

# Base GBDT config; ``scale_pos_weight`` is overridden per side (n_neg/n_pos).
_GBDT_CFG = GBDTConfig(
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=400,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    label_horizon=_ORACLE_CFG.horizon,
    seed=_SEED,
)

_POSITIVE_TIERS: frozenset[str] = frozenset({"strong", "regular"})


@dataclass(frozen=True)
class ImportanceReport:
    """One side's class-balanced GBDT GAIN importances over the whole pool.

    Attributes:
        side: ``"high"`` (top turns) or ``"low"`` (bottom turns).
        feature_names: Feature column names in descending-gain order.
        gains: GAIN importance per feature, aligned to ``feature_names`` (same
            descending order).
        n_pos: Number of positive (strong/regular) labels in the pool.
        n_rows: Total pooled rows the model was fitted on.
        scale_pos_weight: The ``n_neg / n_pos`` class-balance multiplier used.
    """

    side: str
    feature_names: list[str]
    gains: list[float]
    n_pos: int
    n_rows: int
    scale_pos_weight: float

    def top(self, n: int) -> list[tuple[str, float]]:
        """Return the top-``n`` ``(feature, gain)`` pairs in descending order."""
        return list(zip(self.feature_names[:n], self.gains[:n], strict=False))


def _side_columns(side: str) -> tuple[str, str]:
    """Return the ``(tier_col, weight_col)`` oracle columns for ``side``.

    Args:
        side: ``"high"`` (top) or ``"low"`` (bottom).

    Returns:
        The tier and weight column names in the oracle label frame.

    Raises:
        ValueError: If ``side`` is not ``"high"`` or ``"low"``.
    """
    if side == "high":
        return "top_tier", "top_weight"
    if side == "low":
        return "bottom_tier", "bottom_weight"
    raise ValueError(f"feature_importance: unknown side {side!r} (expected 'high'/'low')")


def fit_side_importance(
    data: PooledDataset,
    side: str,
    cfg: GBDTConfig = _GBDT_CFG,
) -> ImportanceReport:
    """Fit one class-balanced GBDT on all pooled rows and read GAIN importances.

    No CV and no threshold: a single full-data fit is the standard way to read a
    model's global feature importances. The label / sample-weight / class-balance
    conventions are identical to :func:`pipeline.fit_pooled.fit_side` so the tells
    reflect the same detector that is scored out-of-sample.

    Args:
        data: The pooled corpus.
        side: ``"high"`` or ``"low"``.
        cfg: Base GBDT configuration (``scale_pos_weight`` is overridden here).

    Returns:
        The side's :class:`ImportanceReport` with features sorted by GAIN desc.
    """
    tier_col, weight_col = _side_columns(side)
    tiers = data.labels[tier_col].to_numpy()
    y = np.isin(tiers, list(_POSITIVE_TIERS)).astype(np.int64)

    # SAMPLE-WEIGHT CONVENTION (matches the pooled baseline): the oracle weight
    # column IS the turn n-score (0 on non-turns). Floor negatives at 1 so they are
    # not zeroed, and let the score add emphasis on positives: w = 1 + score, else 1.
    oracle_score = data.labels[weight_col].to_numpy(dtype=np.float64)
    sample_weight = np.where(y == 1, 1.0 + oracle_score, 1.0).astype(np.float64)

    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    spw = float(n_neg / max(n_pos, 1))
    cfg = replace(cfg, scale_pos_weight=spw)

    logger.info(
        "feature_importance[%s]: %d positives / %d rows (%.3f%%); "
        "scale_pos_weight=%.1f",
        side,
        n_pos,
        y.shape[0],
        100.0 * n_pos / max(y.shape[0], 1),
        spw,
    )

    set_seed(cfg.seed)
    params = cfg.lgb_params()
    # Read GAIN (total split-gain) importances rather than the default split-count.
    params["importance_type"] = "gain"
    model = lgb.LGBMClassifier(**params)
    # Pass the named DataFrame so ``booster_`` carries the real feature names.
    model.fit(data.X, y, sample_weight=sample_weight)

    names = list(data.X.columns)
    gains = np.asarray(model.feature_importances_, dtype=np.float64)
    order = np.argsort(-gains, kind="stable")
    sorted_names = [names[i] for i in order]
    sorted_gains = [float(gains[i]) for i in order]

    logger.info(
        "feature_importance[%s]: top feature %r (gain=%.1f)",
        side,
        sorted_names[0] if sorted_names else "<none>",
        sorted_gains[0] if sorted_gains else 0.0,
    )
    return ImportanceReport(
        side=side,
        feature_names=sorted_names,
        gains=sorted_gains,
        n_pos=n_pos,
        n_rows=int(y.shape[0]),
        scale_pos_weight=spw,
    )


def _render_report(data: PooledDataset, rep: ImportanceReport) -> str:
    """Render one side's GAIN importances as a Markdown document (full ranking).

    Args:
        data: The pooled corpus (for the asset / setup header).
        rep: The side's importance report.

    Returns:
        The Markdown document as a single string (ASCII only).
    """
    name = "top" if rep.side == "high" else "bottom"
    total_gain = sum(rep.gains) or 1.0

    lines: list[str] = []
    lines.append(
        f"# cfd10 GBDT GAIN feature importances - {rep.side.upper()} ({name} turns)"
    )
    lines.append("")
    lines.append(
        "Single class-balanced LightGBM fitted on **all** pooled rows (no CV, no "
        "threshold) to read the detector's global GAIN importances - i.e. *which "
        "tells the best turn detector leans on*. Descriptive only; not an "
        "out-of-sample score."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- Pool: {data.n_assets} `{_TIMEFRAME}` assets, {data.n_rows} rows after "
        "per-asset warm-up removal."
    )
    lines.append(f"- Features: {data.X.shape[1]} columns.")
    lines.append(
        f"- STRUCTURAL oracle: nest={_ORACLE_CFG.scale_nest}, "
        f"weight_curve={_ORACLE_CFG.weight_curve!r}, horizon={_ORACLE_CFG.horizon}, "
        f"drawdown_pct={_ORACLE_CFG.drawdown_pct}, tau_strong={_ORACLE_CFG.tau_strong}, "
        f"tau_regular={_ORACLE_CFG.tau_regular}."
    )
    lines.append(
        f"- Positives: {rep.n_pos} of {rep.n_rows} rows "
        f"({100.0 * rep.n_pos / max(rep.n_rows, 1):.3f}%); "
        f"scale_pos_weight={rep.scale_pos_weight:.1f}; "
        f"GBDT num_leaves={_GBDT_CFG.num_leaves}, "
        f"n_estimators={_GBDT_CFG.n_estimators}, lr={_GBDT_CFG.learning_rate}."
    )
    lines.append("- Importance type: GAIN (total split gain).")
    lines.append("")
    lines.append(f"## GAIN importances ({len(rep.feature_names)} features, sorted)")
    lines.append("")
    lines.append("| rank | feature | gain | gain % |")
    lines.append("| --: | :-- | --: | --: |")
    for rank, (feat, gain) in enumerate(
        zip(rep.feature_names, rep.gains, strict=False), start=1
    ):
        pct = 100.0 * gain / total_gain
        lines.append(f"| {rank} | {feat} | {gain:.1f} | {pct:.2f} |")
    lines.append("")
    return "\n".join(lines)


def run(
    seed: int = _SEED, max_assets: int | None = None
) -> tuple[PooledDataset, dict[str, ImportanceReport]]:
    """Build the pool, fit a class-balanced GBDT per side, save GAIN importances.

    Args:
        seed: Global RNG seed for reproducibility.
        max_assets: Optional cap on the pooled asset count (forwarded to
            :func:`build_pooled_dataset`); ``None`` pools the full daily corpus.

    Returns:
        ``(dataset, reports)`` where ``reports`` maps each side to its
        :class:`ImportanceReport`.

    Raises:
        FileNotFoundError: If the data root is missing.
    """
    set_seed(seed)
    if not _DATA_ROOT.is_dir():
        raise FileNotFoundError(f"feature_importance: missing data root {_DATA_ROOT}")

    dataset, issues = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=FeatureConfig(),
        oracle_cfg=_ORACLE_CFG,
        timeframe=_TIMEFRAME,
        max_assets=max_assets,
    )
    for msg in issues:
        logger.info("feature_importance: build note - %s", msg)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports: dict[str, ImportanceReport] = {}
    for side in ("low", "high"):
        rep = fit_side_importance(dataset, side)
        reports[side] = rep
        out_path = _OUT_DIR / f"feature_importance_{side}.md"
        out_path.write_text(_render_report(dataset, rep), encoding="utf-8")
        logger.info("feature_importance: wrote %s", out_path)

    return dataset, reports


def main() -> None:
    """Entry point: fit both sides and print the TOP-N tells per side."""
    dataset, reports = run()
    print()
    print(f"  pool: {dataset.n_assets} assets, {dataset.n_rows} rows, "
          f"{dataset.X.shape[1]} features")
    for side in ("low", "high"):
        rep = reports[side]
        name = "top" if side == "high" else "bottom"
        print("-" * 72)
        print(f"  {side.upper()} ({name} turns) - TOP {_TOP_N} GAIN features "
              f"(n_pos={rep.n_pos}):")
        for rank, (feat, gain) in enumerate(rep.top(_TOP_N), start=1):
            print(f"    {rank:2d}. {feat:24s}  gain={gain:.1f}")
        print(f"  -> {_OUT_DIR / f'feature_importance_{side}.md'}")


if __name__ == "__main__":
    main()
