"""Fit the GBDT baseline on the **multi-asset pool** with the STRUCTURAL oracle.

Run with::

    uv run python pipeline/fit_pooled.py

This tests the core design bet of cfd10: that pooling many daily instruments
solves the label-scarcity problem of the single-asset turn oracle. SPX alone
yields only ~82 strong/regular turns — too few to train a discriminative
detector. By stacking ~45 daily instruments (each labelled *independently*; see
:func:`cfd10.data_module.build_pooled_dataset`) the positive count grows by an
order of magnitude while the bounded/dimensionless feature bank keeps a turn on
gold and a turn on SPX in the same space.

The chain mirrors :mod:`pipeline.fit_baseline` but on the pool:

1. build the pooled corpus with the famous-lows-validated STRUCTURAL oracle
   (nest ``(20, 50, 100, 200)``, 10% drawdown, horizon 60, tau 0.60/0.30);
2. build purged + embargoed walk-forward folds on the pooled time axis, with
   ``pooled_groups`` so purge/embargo respect each asset's own bar clock;
3. fit the LightGBM baseline per side (HIGH/LOW), tuning each fold's decision
   threshold on the train fold only;
4. score out-of-sample event PRF **both** pooled (all assets) **and** restricted
   to the SPX-subset rows, using the same fitted models/thresholds;
5. write ``outputs/baseline_pooled_scorecard.md`` with the asset list, per-side
   structural label counts, both OOS slices, and an explicit comparison to the
   single-asset *loose*-oracle baseline.

This is an oracle-vs-detector OOS evaluation only — no Pine-signal comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cfd10.cv_module import Fold, purged_walk_forward
from cfd10.data_module import PooledDataset, build_pooled_dataset
from cfd10.eval_module import Scorecard, ScoreRow, event_prf, match_events
from cfd10.feature_module import FeatureConfig
from cfd10.label_module import OracleConfig
from cfd10.teacher_module import GBDTConfig
from cfd10.teacher_module.baseline import _fit_one_fold
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DATA_ROOT: Path = _REPO_ROOT / "data" / "raw_v16"
_OUT_DIR: Path = _REPO_ROOT / "outputs"
_SCORECARD_PATH: Path = _OUT_DIR / "baseline_pooled_scorecard.md"

_SPX: str = "SPX"
_SEED: int = 42
_TIMEFRAME: str = "1D"

# The famous-lows-validated STRUCTURAL oracle: big-scale turns weighted heavily.
# A confirmed turn at the dominant scales scores high; the candidate gate (local
# extreme over the +/-200-bar window) keeps positives structurally sparse.
_ORACLE_CFG = OracleConfig(
    scale_nest=(20, 50, 100, 200),
    weight_curve="linear",
    drawdown_pct=0.10,
    horizon=60,
    tau_strong=0.60,
    tau_regular=0.30,
)

# Baseline GBDT config. ``label_horizon`` mirrors the oracle horizon so the purge
# removes exactly the bars whose forward label window overlaps a test block.
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
    tolerance=3,
    label_horizon=_ORACLE_CFG.horizon,
    embargo=5,
    n_folds=6,
    n_threshold_grid=50,
    seed=_SEED,
)

_POSITIVE_TIERS: frozenset[str] = frozenset({"strong", "regular"})

# The published single-asset *loose*-oracle SPX baseline (the number to beat).
_SINGLE_ASSET_LOW = {"f1": 0.249, "precision": 0.152, "recall": 0.696}
_SINGLE_ASSET_HIGH = {"f1": 0.106, "precision": float("nan"), "recall": float("nan")}


# --------------------------------------------------------------------------- #
# Result containers.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OosSlice:
    """Out-of-sample event PRF over one row slice (all-assets or SPX-only).

    Attributes:
        precision: Event precision in ``[0, 1]``.
        recall: Event recall in ``[0, 1]``.
        f1: Event F1 in ``[0, 1]``.
        tp / fp / fn: Confusion counts behind the metrics.
        n_true / n_pred: True and predicted OOS event counts in the slice.
    """

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_true: int
    n_pred: int


@dataclass(frozen=True)
class SideReport:
    """One side's pooled fit: label balance + pooled and SPX-subset OOS.

    Attributes:
        side: ``"high"`` (top turns) or ``"low"`` (bottom turns).
        n_pos: Number of positive (strong/regular) labels pooled.
        n_strong: Number of ``strong``-tier labels pooled.
        n_regular: Number of ``regular``-tier labels pooled.
        positive_rate: Positive label fraction over the whole pool.
        mean_fold_f1: Mean per-fold OOS F1 (all assets).
        fold_f1s: Per-fold OOS F1 (all assets).
        pooled: All-assets pooled OOS slice.
        spx: SPX-subset OOS slice (same fitted models / thresholds).
        spx_pos: Number of positive labels on the SPX subset.
    """

    side: str
    n_pos: int
    n_strong: int
    n_regular: int
    positive_rate: float
    mean_fold_f1: float
    fold_f1s: list[float]
    pooled: OosSlice
    spx: OosSlice
    spx_pos: int


# --------------------------------------------------------------------------- #
# Side fitting (retaining OOS indices for the SPX-subset slice).               #
# --------------------------------------------------------------------------- #


def _side_columns(side: str) -> tuple[str, str]:
    """Return the ``(tier_col, weight_col)`` oracle columns for ``side``."""
    if side == "high":
        return "top_tier", "top_weight"
    if side == "low":
        return "bottom_tier", "bottom_weight"
    raise ValueError(f"fit_pooled: unknown side {side!r} (expected 'high'/'low')")


def _slice_oos(
    pred_idx: NDArray[np.int64],
    true_idx: NDArray[np.int64],
    keep: NDArray[np.bool_] | None,
    tolerance: int,
) -> OosSlice:
    """Score event PRF over predicted/true indices, optionally row-filtered.

    Args:
        pred_idx: Pooled OOS predicted bar indices (into the pooled rows).
        true_idx: Pooled OOS true oracle bar indices.
        keep: Optional boolean mask over *all* pooled rows; when given, only
            predicted/true indices whose row is kept are scored (the SPX subset).
            ``None`` scores the full pool.
        tolerance: Event-match tolerance in bars.

    Returns:
        The :class:`OosSlice` for the (optionally filtered) indices.
    """
    if keep is not None:
        pred_idx = pred_idx[keep[pred_idx]]
        true_idx = true_idx[keep[true_idx]]
    tp, fp, fn, _ = match_events(pred_idx, true_idx, tolerance)
    p, r, f1 = event_prf(pred_idx, true_idx, tolerance)
    return OosSlice(
        precision=p,
        recall=r,
        f1=f1,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        n_true=int(true_idx.size),
        n_pred=int(pred_idx.size),
    )


def fit_side(
    data: PooledDataset,
    splits: list[Fold],
    side: str,
    spx_mask: NDArray[np.bool_] | None,
    cfg: GBDTConfig = _GBDT_CFG,
) -> SideReport:
    """Fit the per-side GBDT on the pool and score pooled + SPX-subset OOS.

    The fold loop reuses the teacher's :func:`_fit_one_fold` (identical model,
    train-only threshold tuning, and OOS scoring as :func:`fit_gbdt_cv`) but
    retains each fold's OOS predicted / true bar indices. The disjoint per-fold
    test blocks are concatenated and scored once for the all-assets number, then
    re-scored after restricting to SPX rows — both from the *same* fitted models,
    so the SPX slice is a faithful view of the pooled detector, not a refit.

    Args:
        data: The pooled corpus.
        splits: Purged walk-forward folds over the pooled time axis.
        side: ``"high"`` or ``"low"``.
        spx_mask: Boolean mask over pooled rows selecting SPX (``None`` if SPX is
            absent from the pool).
        cfg: Baseline configuration.

    Returns:
        The assembled :class:`SideReport`.
    """
    tier_col, weight_col = _side_columns(side)
    tiers = data.labels[tier_col].to_numpy()
    y = np.isin(tiers, list(_POSITIVE_TIERS)).astype(np.int64)

    # SAMPLE-WEIGHT CONVENTION: the oracle weight column IS the turn score (0 on
    # non-turns). Used raw it would zero the negative class. Floor negatives at 1
    # and let the score *add* emphasis on positives: w = 1 + score on turns, else 1.
    oracle_score = data.labels[weight_col].to_numpy(dtype=np.float64)
    sample_weight = np.where(y == 1, 1.0 + oracle_score, 1.0).astype(np.float64)

    n_strong = int((tiers == "strong").sum())
    n_regular = int((tiers == "regular").sum())
    n_pos = int(y.sum())

    # Class balancing. The STRUCTURAL oracle is ~0.2% positive, so an unweighted
    # GBDT (scale_pos_weight=1) lets the turns wash out: probabilities stay
    # near-flat, train event-F1 ~0, and threshold tuning then defaults to ~max ->
    # ZERO predicted events. Re-weight the positive class by the negative/positive
    # ratio so structural turns actually drive the tree splits. The train-only
    # threshold tuner downstream still controls the precision/recall trade-off.
    from dataclasses import replace

    n_neg = int(y.shape[0] - n_pos)
    spw = float(n_neg / max(n_pos, 1))
    cfg = replace(cfg, scale_pos_weight=spw)
    logger.info(
        "fit_pooled[%s]: %d positives (%d strong, %d regular) / %d rows (%.3f%%); "
        "scale_pos_weight=%.1f",
        side,
        n_pos,
        n_strong,
        n_regular,
        y.shape[0],
        100.0 * n_pos / max(y.shape[0], 1),
        spw,
    )

    set_seed(cfg.seed)
    X_values = np.ascontiguousarray(data.X.to_numpy(dtype=np.float64))

    pred_blocks: list[NDArray[np.int64]] = []
    true_blocks: list[NDArray[np.int64]] = []
    fold_f1s: list[float] = []
    for fold_id, fold in enumerate(splits):
        result, test_pred_idx, test_true_idx = _fit_one_fold(
            X_values, y, sample_weight, fold_id, fold, cfg
        )
        pred_blocks.append(test_pred_idx)
        true_blocks.append(test_true_idx)
        fold_f1s.append(result.f1)

    all_pred = (
        np.concatenate(pred_blocks) if pred_blocks else np.empty(0, dtype=np.int64)
    )
    all_true = (
        np.concatenate(true_blocks) if true_blocks else np.empty(0, dtype=np.int64)
    )

    pooled_slice = _slice_oos(all_pred, all_true, None, cfg.tolerance)
    spx_slice = _slice_oos(all_pred, all_true, spx_mask, cfg.tolerance)
    spx_pos = int(y[spx_mask].sum()) if spx_mask is not None else 0

    logger.info(
        "fit_pooled[%s]: pooled OOS f1=%.4f (p=%.4f r=%.4f) | SPX OOS f1=%.4f "
        "(p=%.4f r=%.4f, n_true=%d)",
        side,
        pooled_slice.f1,
        pooled_slice.precision,
        pooled_slice.recall,
        spx_slice.f1,
        spx_slice.precision,
        spx_slice.recall,
        spx_slice.n_true,
    )

    return SideReport(
        side=side,
        n_pos=n_pos,
        n_strong=n_strong,
        n_regular=n_regular,
        positive_rate=float(y.mean()) if y.size else 0.0,
        mean_fold_f1=float(np.mean(fold_f1s)) if fold_f1s else 0.0,
        fold_f1s=[round(f, 4) for f in fold_f1s],
        pooled=pooled_slice,
        spx=spx_slice,
        spx_pos=spx_pos,
    )


# --------------------------------------------------------------------------- #
# Scorecard + report.                                                          #
# --------------------------------------------------------------------------- #


def build_scorecard(reports: list[SideReport]) -> Scorecard:
    """Assemble a scorecard with pooled + SPX OOS rows per side.

    Args:
        reports: Per-side pooled fit results.

    Returns:
        A :class:`Scorecard` with ``(side, POOL, oos)`` and ``(side, SPX, oos)``
        rows for each side.
    """
    card = Scorecard()
    for rep in reports:
        for asset, sl in (("POOL", rep.pooled), ("SPX", rep.spx)):
            card = card.add(
                ScoreRow(
                    side=rep.side,
                    asset=asset,
                    era="oos",
                    tp=sl.tp,
                    fp=sl.fp,
                    fn=sl.fn,
                    precision=sl.precision,
                    recall=sl.recall,
                    f1=sl.f1,
                )
            )
    return card


def _fmt_delta(new: float, base: float) -> str:
    """Format a signed F1 delta vs the single-asset baseline (e.g. ``+0.081``)."""
    if base != base:  # NaN base (no published value)
        return "n/a"
    d = new - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _render_report(
    data: PooledDataset,
    reports: list[SideReport],
    issues: list[str],
    spx_present: bool,
) -> str:
    """Render the full pooled Markdown scorecard (header, setup, tables, notes)."""
    lines: list[str] = []
    lines.append("# cfd10 GBDT baseline — multi-asset POOLED, STRUCTURAL oracle")
    lines.append("")
    lines.append(
        "Tests the core design bet: that **pooling daily instruments** solves the "
        "label scarcity of the single-asset turn oracle. Per-side LightGBM turn "
        "detectors are trained on the stacked corpus and scored out-of-sample with "
        "purged + embargoed walk-forward CV (purge/embargo per asset's own bar "
        f"clock). Metric: event precision/recall/F1 at a {_GBDT_CFG.tolerance}-bar "
        "tolerance."
    )
    lines.append("")
    lines.append(
        "**Scope:** oracle-vs-detector only (no Pine-signal comparison). Features "
        "and labels are computed *per asset* before stacking — no rolling window "
        "or forward oracle peek crosses an asset boundary."
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- Pool: {data.n_assets} `{_TIMEFRAME}` assets, {data.n_rows} rows after "
        "per-asset warm-up removal."
    )
    counts = data.counts_per_group()
    asset_list = ", ".join(f"{a} ({counts[a]})" for a in data.asset_names)
    lines.append(f"- Assets (rows): {asset_list}.")
    lines.append(f"- Features: {data.X.shape[1]} columns (default bank).")
    lines.append(
        f"- STRUCTURAL oracle: nest={_ORACLE_CFG.scale_nest}, "
        f"weight_curve={_ORACLE_CFG.weight_curve!r}, "
        f"horizon={_ORACLE_CFG.horizon}, drawdown_pct={_ORACLE_CFG.drawdown_pct}, "
        f"tau_strong={_ORACLE_CFG.tau_strong}, tau_regular={_ORACLE_CFG.tau_regular}."
    )
    lines.append(
        f"- CV: purged walk-forward, n_folds={_GBDT_CFG.n_folds}, "
        f"label_horizon={_GBDT_CFG.label_horizon}, embargo={_GBDT_CFG.embargo}, "
        "pooled_groups=per-asset."
    )
    lines.append(
        f"- GBDT: num_leaves={_GBDT_CFG.num_leaves}, "
        f"n_estimators={_GBDT_CFG.n_estimators}, lr={_GBDT_CFG.learning_rate}, "
        "scale_pos_weight=auto (n_neg/n_pos per side)."
    )
    if not spx_present:
        lines.append("- NOTE: SPX absent from the pool; SPX-subset rows are empty.")
    if issues:
        lines.append("")
        lines.append("**Build issues / subset notes:**")
        for msg in issues:
            lines.append(f"- {msg}")
    lines.append("")

    # Structural label counts — the answer to "is it now enough to train?".
    lines.append("## Structural label counts (pooled)")
    lines.append("")
    lines.append("| side | strong | regular | positives | SPX positives |")
    lines.append("| :-- | --: | --: | --: | --: |")
    for rep in reports:
        lines.append(
            f"| {rep.side} | {rep.n_strong} | {rep.n_regular} | {rep.n_pos} | "
            f"{rep.spx_pos} |"
        )
    lines.append("")

    lines.append("## Out-of-sample event metrics")
    lines.append("")
    lines.append(build_scorecard(reports).to_markdown())
    lines.append("")
    lines.append(
        "`POOL` = all assets pooled; `SPX` = the SPX-subset rows scored from the "
        "*same* pooled-trained models and thresholds."
    )
    lines.append("")

    # Explicit comparison to the single-asset loose-oracle baseline.
    lines.append("## Comparison vs single-asset loose-oracle SPX baseline")
    lines.append("")
    lines.append(
        "Baseline (SPX-only, *loose* oracle nest=(5,10,20,40), drawdown 5%, "
        "tau 0.5/0.25): "
        f"LOW F1={_SINGLE_ASSET_LOW['f1']:.3f} "
        f"(P={_SINGLE_ASSET_LOW['precision']:.3f}, "
        f"R={_SINGLE_ASSET_LOW['recall']:.3f}); "
        f"HIGH F1={_SINGLE_ASSET_HIGH['f1']:.3f}."
    )
    lines.append("")
    lines.append(
        "| side | baseline F1 | pooled-train SPX OOS F1 | delta F1 | pooled (all) F1 |"
    )
    lines.append("| :-- | --: | --: | --: | --: |")
    base_by_side = {"low": _SINGLE_ASSET_LOW, "high": _SINGLE_ASSET_HIGH}
    for rep in reports:
        base = base_by_side[rep.side]["f1"]
        base_str = f"{base:.3f}" if base == base else "n/a"
        lines.append(
            f"| {rep.side} | {base_str} | {rep.spx.f1:.4f} | "
            f"{_fmt_delta(rep.spx.f1, base)} | {rep.pooled.f1:.4f} |"
        )
    lines.append("")

    # Per-side detail.
    lines.append("## Per-side detail")
    lines.append("")
    for rep in reports:
        name = "top" if rep.side == "high" else "bottom"
        lines.append(f"### {rep.side.upper()} ({name} turns)")
        lines.append("")
        lines.append(
            f"- Pooled positives: {rep.n_pos} ({rep.n_strong} strong, "
            f"{rep.n_regular} regular); positive rate {rep.positive_rate * 100:.3f}%."
        )
        lines.append(
            f"- Pooled OOS: P={rep.pooled.precision:.4f}, R={rep.pooled.recall:.4f}, "
            f"F1={rep.pooled.f1:.4f} (tp={rep.pooled.tp}, fp={rep.pooled.fp}, "
            f"fn={rep.pooled.fn}, n_true={rep.pooled.n_true}, "
            f"n_pred={rep.pooled.n_pred})."
        )
        lines.append(
            f"- SPX-subset OOS: P={rep.spx.precision:.4f}, R={rep.spx.recall:.4f}, "
            f"F1={rep.spx.f1:.4f} (tp={rep.spx.tp}, fp={rep.spx.fp}, "
            f"fn={rep.spx.fn}, n_true={rep.spx.n_true}, n_pred={rep.spx.n_pred})."
        )
        lines.append(f"- Mean per-fold F1 (all assets): {rep.mean_fold_f1:.4f}.")
        lines.append(f"- Per-fold OOS F1 (all assets): {rep.fold_f1s}.")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #


def run(
    seed: int = _SEED, max_assets: int | None = None
) -> tuple[PooledDataset, list[SideReport], list[str]]:
    """Build the pool, fit both sides, score OOS, and save the scorecard.

    Args:
        seed: Global RNG seed for reproducibility.
        max_assets: Optional cap on the pooled asset count (forwarded to
            :func:`build_pooled_dataset`); ``None`` pools the full daily corpus.

    Returns:
        ``(dataset, reports, issues)``.

    Raises:
        FileNotFoundError: If the data root is missing.
    """
    set_seed(seed)
    if not _DATA_ROOT.is_dir():
        raise FileNotFoundError(f"fit_pooled: missing data root {_DATA_ROOT}")

    dataset, issues = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=FeatureConfig(),
        oracle_cfg=_ORACLE_CFG,
        timeframe=_TIMEFRAME,
        max_assets=max_assets,
    )

    spx_present = _SPX in dataset.asset_names
    spx_mask = dataset.subset_mask(_SPX) if spx_present else None

    splits = purged_walk_forward(
        dataset.timestamps,
        label_horizon=_GBDT_CFG.label_horizon,
        embargo=_GBDT_CFG.embargo,
        n_folds=_GBDT_CFG.n_folds,
        pooled_groups=dataset.groups,
    )

    reports = [fit_side(dataset, splits, side, spx_mask) for side in ("high", "low")]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    document = _render_report(dataset, reports, issues, spx_present)
    _SCORECARD_PATH.write_text(document, encoding="utf-8")
    logger.info("fit_pooled: wrote scorecard to %s", _SCORECARD_PATH)
    return dataset, reports, issues


def main() -> None:
    """Entry point: run the pooled pipeline and print the scorecard."""
    dataset, reports, _issues = run()
    print()
    print(_SCORECARD_PATH.read_text(encoding="utf-8"))
    print("-" * 72)
    print(f"  pool: {dataset.n_assets} assets, {dataset.n_rows} rows")
    for rep in reports:
        print(
            f"  {rep.side.upper():4s}  pos={rep.n_pos:4d} (S{rep.n_strong}/"
            f"R{rep.n_regular})  POOL F1={rep.pooled.f1:.4f}  "
            f"SPX F1={rep.spx.f1:.4f} (n_true={rep.spx.n_true})"
        )
    print(f"  scorecard -> {_SCORECARD_PATH}")


if __name__ == "__main__":
    main()
