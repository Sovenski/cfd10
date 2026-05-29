"""Fit the **deep TCN teacher** on the multi-asset pool (Colab GPU entrypoint).

Run on a Colab T4/L4 GPU (a few minutes with the small defaults)::

    !python pipeline/fit_teacher.py --epochs 30
    # window / batch / lr are also overridable:
    !python pipeline/fit_teacher.py --epochs 40 --window 48 --batch 256 --lr 1e-3

DO NOT run this to completion on a CPU — training a conv net over the full pool is
far too slow there. The CPU unit tests (``tests/teacher_module/test_tcn.py``)
cover the wiring; this script is for the GPU run.

What it does
------------
1. Build the pooled corpus with the famous-lows-validated **STRUCTURAL** oracle
   (nest ``(20, 50, 100, 200)``, linear weight curve, 10% drawdown, horizon 60,
   tau 0.60/0.30) — the *same* labels the GBDT baseline uses, so the comparison is
   apples-to-apples.
2. Build a :class:`WindowDataset` of ``W``-bar windows that never cross an asset
   boundary, standardized with per-fold **train-split** statistics.
3. Build purged + embargoed walk-forward folds on the pooled time axis
   (``label_horizon=60, embargo=5, n_folds=6``) with ``pooled_groups`` so
   purge/embargo respect each asset's own bar clock.
4. Per fold, train ONE dual-head TCN (it emits *both* the top and bottom turn
   logits — the architecture is intrinsically two-sided), early-stopping on a
   time-tail validation slice of the train fold, then isotonically calibrate each
   side on that same held slice. The per-side decision threshold is tuned to
   maximise event-F1 **on the train fold only** and frozen for the OOS test fold.
5. Pool the disjoint OOS test blocks and score per-side event precision / recall /
   F1 at a 3-bar tolerance, plus the SPX-subset slice.
6. **Compare to the GBDT baseline** (run on the identical pooled data / folds) —
   the beat-the-baseline gate — and write ``outputs/teacher_pooled_scorecard.md``.

It also prints ``torch.cuda.is_available()``, the resolved device, and the TCN
parameter count up front. This module is import-safe (nothing runs on import).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from cfd10.cv_module import Fold, purged_walk_forward
from cfd10.data_module import PooledDataset, build_pooled_dataset
from cfd10.eval_module import Scorecard, ScoreRow, event_prf, match_events
from cfd10.feature_module import FeatureConfig
from cfd10.label_module import OracleConfig
from cfd10.teacher_module import (
    GBDTConfig,
    TCNConfig,
    TCNTurnModel,
    TrainConfig,
    WindowDataset,
    apply_calibrators,
    compute_feature_stats,
    fit_calibrators,
    fit_gbdt_cv,
    predict_proba,
    resolve_device,
    train_teacher,
)
from cfd10.teacher_module.datasets import make_labels
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DATA_ROOT: Path = _REPO_ROOT / "data" / "raw_v16"
_OUT_DIR: Path = _REPO_ROOT / "outputs"
_SCORECARD_PATH: Path = _OUT_DIR / "teacher_pooled_scorecard.md"

_SPX: str = "SPX"
_SEED: int = 42
_TIMEFRAME: str = "1D"
_TOLERANCE: int = 3
_LABEL_HORIZON: int = 60
_EMBARGO: int = 5
_N_FOLDS: int = 6
_N_THRESHOLD_GRID: int = 50
# Fraction of each train fold (its time tail) held out for early-stop + isotonic
# calibration. Kept disjoint from the OOS test block by the purged splitter.
_VAL_TAIL_FRAC: float = 0.2

# Side index <-> name. Index 0 = top (high) turns, 1 = bottom (low) turns.
_SIDES: tuple[tuple[int, str], ...] = ((0, "high"), (1, "low"))

# The famous-lows-validated STRUCTURAL oracle (identical to the GBDT baseline).
_ORACLE_CFG = OracleConfig(
    scale_nest=(20, 50, 100, 200),
    weight_curve="linear",
    drawdown_pct=0.10,
    horizon=60,
    tau_strong=0.60,
    tau_regular=0.30,
)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Args:
    """Parsed command-line arguments (small defaults for a fast GPU run).

    Attributes:
        epochs: Max training epochs per fold.
        window: Window length ``W`` (bars per sample).
        batch: Mini-batch size.
        lr: AdamW learning rate.
        channels: TCN hidden channel width.
        max_assets: Optional cap on the pooled asset count (``None`` = all).
        seed: Global RNG seed.
    """

    epochs: int
    window: int
    batch: int
    lr: float
    channels: int
    max_assets: int | None
    seed: int


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse CLI arguments into an :class:`Args` (defaults tuned for a few-minute run).

    Args:
        argv: Argument list (``None`` reads ``sys.argv``).

    Returns:
        The populated :class:`Args`.
    """
    parser = argparse.ArgumentParser(description="Fit the TCN turn teacher on the pool.")
    parser.add_argument("--epochs", type=int, default=30, help="max epochs per fold")
    parser.add_argument("--window", type=int, default=32, help="window length W (bars)")
    parser.add_argument("--batch", type=int, default=256, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--channels", type=int, default=48, help="TCN hidden channels")
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="cap pooled asset count (default: all)",
    )
    parser.add_argument("--seed", type=int, default=_SEED, help="global RNG seed")
    ns = parser.parse_args(argv)
    return Args(
        epochs=ns.epochs,
        window=ns.window,
        batch=ns.batch,
        lr=ns.lr,
        channels=ns.channels,
        max_assets=ns.max_assets,
        seed=ns.seed,
    )


# --------------------------------------------------------------------------- #
# Result containers.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OosSlice:
    """Out-of-sample event PRF over one row slice.

    Attributes:
        precision / recall / f1: Event metrics in ``[0, 1]``.
        tp / fp / fn: Confusion counts.
        n_true / n_pred: True and predicted OOS event counts.
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
    """One side's pooled TCN fit: balance, OOS slices, and the baseline gate.

    Attributes:
        side: ``"high"`` or ``"low"``.
        n_pos: Pooled positive count.
        positive_rate: Pooled positive fraction.
        mean_threshold: Mean per-fold decision threshold (diagnostic).
        pooled: All-assets pooled OOS slice (TCN).
        spx: SPX-subset OOS slice (TCN, same fitted models / thresholds).
        baseline_f1: GBDT baseline pooled OOS F1 on the same data / folds.
        baseline_precision / baseline_recall: GBDT baseline pooled OOS P / R.
    """

    side: str
    n_pos: int
    positive_rate: float
    mean_threshold: float
    pooled: OosSlice
    spx: OosSlice
    baseline_f1: float
    baseline_precision: float
    baseline_recall: float


# --------------------------------------------------------------------------- #
# Row <-> item mapping and threshold tuning.                                   #
# --------------------------------------------------------------------------- #


def _row_to_item(dataset: WindowDataset, n_rows: int) -> NDArray[np.int64]:
    """Map a pooled row index to its dataset item index (or ``-1`` if no window).

    Args:
        dataset: The windowed dataset.
        n_rows: Number of pooled rows.

    Returns:
        An ``(n_rows,)`` array; entry ``r`` is the item whose window *ends* at row
        ``r``, or ``-1`` if no valid window ends there.
    """
    mapping = np.full(n_rows, -1, dtype=np.int64)
    ends = dataset.end_indices
    mapping[ends] = np.arange(ends.shape[0], dtype=np.int64)
    return mapping


def _items_for_rows(
    row_to_item: NDArray[np.int64], rows: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return the dataset items whose end-bar lies in ``rows`` (drops the rest).

    Args:
        row_to_item: Row -> item map from :func:`_row_to_item`.
        rows: Pooled row indices (a CV split).

    Returns:
        Sorted item indices for the windows ending on those rows.
    """
    items = row_to_item[np.ascontiguousarray(rows, dtype=np.int64)]
    items = items[items >= 0]
    return np.ascontiguousarray(np.sort(items), dtype=np.int64)


def _time_tail_split(
    items: NDArray[np.int64], tail_frac: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split items into (head=train, tail=val) by position (a time-ordered tail).

    Items are produced in pooled-row order, which is time-ordered within each
    asset, so the trailing slice is the most recent portion of the train fold — a
    sound early-stop / calibration hold-out that does not peek at the OOS block.

    Args:
        items: Sorted train-fold item indices.
        tail_frac: Fraction to hold out as the tail (``0 < tail_frac < 1``).

    Returns:
        ``(train_items, val_items)``. ``val_items`` is empty if too few items.
    """
    n = items.shape[0]
    n_val = int(round(n * tail_frac))
    if n_val < 1 or n_val >= n:
        return items, np.empty(0, dtype=np.int64)
    return items[: n - n_val], items[n - n_val :]


def _tune_threshold_side(
    proba: NDArray[np.float64],
    end_idx: NDArray[np.int64],
    true_rows: NDArray[np.int64],
    tolerance: int,
    n_grid: int,
) -> tuple[float, float]:
    """Pick the per-side threshold maximising event-F1 on the (train) probs.

    Mirrors the baseline's tuner: candidates are train-probability quantiles, swept
    with the event metric; ties break toward the higher threshold (fewer, more
    confident flags). A degenerate side falls back to ``0.5``.

    Args:
        proba: Calibrated probabilities for one side over the scored items.
        end_idx: Pooled end-bar index of each scored item (aligned to ``proba``).
        true_rows: Pooled bar indices of true oracle turns in this split.
        tolerance: Event-match tolerance in bars.
        n_grid: Number of quantile candidates.

    Returns:
        ``(threshold, train_f1)``.
    """
    if proba.size == 0 or true_rows.size == 0:
        return 0.5, 0.0
    quantiles = np.linspace(0.0, 1.0, n_grid)
    candidates = np.unique(np.quantile(proba, quantiles))
    top = float(np.nextafter(float(proba.max()), 0.0))
    candidates = np.unique(np.concatenate([candidates, np.array([top])]))

    max_pred = max(512, 6 * int(true_rows.size))
    best_threshold = float(candidates[-1])
    best_f1 = -1.0
    for thr in candidates[::-1]:
        pred_rows = np.sort(end_idx[proba >= float(thr)])
        if pred_rows.size > max_pred:
            break
        _p, _r, f1 = event_prf(pred_rows, true_rows, tolerance)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
    return best_threshold, float(max(best_f1, 0.0))


# --------------------------------------------------------------------------- #
# Per-fold TCN fit + OOS prediction.                                           #
# --------------------------------------------------------------------------- #


def _true_rows_for_side(
    y_side: NDArray[np.float64], rows: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Pooled bar indices of true turns for one side, restricted to ``rows``."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    mask = y_side[rows] == 1.0
    return np.ascontiguousarray(np.sort(rows[mask]), dtype=np.int64)


@dataclass(frozen=True)
class _FoldOos:
    """Per-fold OOS predicted/true bar indices for both sides."""

    pred_rows: dict[int, NDArray[np.int64]]
    true_rows: dict[int, NDArray[np.int64]]
    thresholds: dict[int, float]


def _fit_one_fold_tcn(
    data: PooledDataset,
    fold_id: int,
    fold: Fold,
    args: Args,
    device: torch.device,
) -> _FoldOos:
    """Train the dual-head TCN on one fold and emit per-side OOS event indices.

    Builds a fold-local :class:`WindowDataset` standardized on the **train rows**
    only, fits the TCN with a time-tail validation slice (early stop), calibrates
    both sides on that slice, tunes a per-side threshold on the train fold, and
    flags OOS test bars whose calibrated probability clears it.

    Args:
        data: The pooled corpus.
        fold_id: 0-based fold index.
        fold: ``(train_rows, test_rows)`` pooled indices.
        args: CLI arguments (window / epochs / batch / lr / channels / seed).
        device: Compute device.

    Returns:
        The fold's :class:`_FoldOos`.
    """
    train_rows, test_rows = fold

    stats = compute_feature_stats(data.X, train_rows)
    dataset = WindowDataset(
        data.X, data.labels, data.groups, window=args.window, stats=stats
    )
    row_to_item = _row_to_item(dataset, data.n_rows)

    train_items = _items_for_rows(row_to_item, train_rows)
    test_items = _items_for_rows(row_to_item, test_rows)
    fit_items, val_items = _time_tail_split(train_items, _VAL_TAIL_FRAC)

    model_cfg = TCNConfig(in_features=data.X.shape[1], channels=args.channels)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed + fold_id,
    )
    model = train_teacher(
        dataset,
        fit_items if val_items.size else train_items,
        val_items if val_items.size else None,
        model_cfg,
        train_cfg,
        device=device,
    )

    # Calibrate on the held val slice (fall back to the train fold if no tail).
    cal_items = val_items if val_items.size else train_items
    cal_res = predict_proba(model, dataset, cal_items, device=device)
    calibrators = fit_calibrators(cal_res)

    # Score the train fold (for threshold tuning) and the OOS test fold.
    train_res = predict_proba(model, dataset, train_items, device=device)
    test_res = predict_proba(model, dataset, test_items, device=device)
    train_cal = apply_calibrators(calibrators, train_res.proba)
    test_cal = apply_calibrators(calibrators, test_res.proba)

    y, _w = make_labels(data.labels)

    pred_rows: dict[int, NDArray[np.int64]] = {}
    true_rows_oos: dict[int, NDArray[np.int64]] = {}
    thresholds: dict[int, float] = {}
    for side, _name in _SIDES:
        train_true = _true_rows_for_side(y[:, side], train_rows)
        threshold, _train_f1 = _tune_threshold_side(
            train_cal[:, side],
            train_res.end_idx,
            train_true,
            _TOLERANCE,
            _N_THRESHOLD_GRID,
        )
        thresholds[side] = threshold
        flag = test_cal[:, side] >= threshold
        pred_rows[side] = np.ascontiguousarray(
            np.sort(test_res.end_idx[flag]), dtype=np.int64
        )
        true_rows_oos[side] = _true_rows_for_side(y[:, side], test_rows)
        logger.info(
            "fold %d [%s]: thr=%.4f -> OOS pred=%d true=%d",
            fold_id,
            _name,
            threshold,
            pred_rows[side].size,
            true_rows_oos[side].size,
        )

    return _FoldOos(pred_rows=pred_rows, true_rows=true_rows_oos, thresholds=thresholds)


# --------------------------------------------------------------------------- #
# OOS scoring + baseline comparison.                                           #
# --------------------------------------------------------------------------- #


def _slice_oos(
    pred_rows: NDArray[np.int64],
    true_rows: NDArray[np.int64],
    keep: NDArray[np.bool_] | None,
    tolerance: int,
) -> OosSlice:
    """Score event PRF over pooled predicted/true rows, optionally row-filtered."""
    if keep is not None:
        pred_rows = pred_rows[keep[pred_rows]]
        true_rows = true_rows[keep[true_rows]]
    tp, fp, fn, _ = match_events(pred_rows, true_rows, tolerance)
    p, r, f1 = event_prf(pred_rows, true_rows, tolerance)
    return OosSlice(
        precision=p,
        recall=r,
        f1=f1,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        n_true=int(true_rows.size),
        n_pred=int(pred_rows.size),
    )


def _baseline_side(
    data: PooledDataset, splits: list[Fold], side: int
) -> tuple[float, float, float]:
    """Run the GBDT baseline on one side over the same pooled data / folds.

    Args:
        data: The pooled corpus.
        splits: The shared purged walk-forward folds.
        side: ``0`` (top) or ``1`` (bottom).

    Returns:
        ``(precision, recall, f1)`` pooled OOS for the GBDT baseline.
    """
    y, w = make_labels(data.labels)
    cfg = GBDTConfig(
        n_estimators=400,
        tolerance=_TOLERANCE,
        label_horizon=_LABEL_HORIZON,
        embargo=_EMBARGO,
        n_folds=_N_FOLDS,
        seed=_SEED,
    )
    out = fit_gbdt_cv(
        data.X,
        y[:, side].astype(np.int64),
        w[:, side].astype(np.float64),
        splits,
        cfg,
    )
    return (
        cast(float, out["oos_precision"]),
        cast(float, out["oos_recall"]),
        cast(float, out["oos_f1"]),
    )


def _assemble_side_report(
    data: PooledDataset,
    side: int,
    name: str,
    fold_oos: list[_FoldOos],
    spx_mask: NDArray[np.bool_] | None,
    baseline_prf: tuple[float, float, float],
) -> SideReport:
    """Pool a side's per-fold OOS indices and assemble its :class:`SideReport`."""
    pred = np.concatenate([f.pred_rows[side] for f in fold_oos]) if fold_oos else np.empty(0, np.int64)
    true = np.concatenate([f.true_rows[side] for f in fold_oos]) if fold_oos else np.empty(0, np.int64)

    pooled = _slice_oos(pred, true, None, _TOLERANCE)
    spx = _slice_oos(pred, true, spx_mask, _TOLERANCE)

    y, _w = make_labels(data.labels)
    n_pos = int((y[:, side] == 1.0).sum())
    mean_thr = float(np.mean([f.thresholds[side] for f in fold_oos])) if fold_oos else 0.5
    b_p, b_r, b_f1 = baseline_prf

    logger.info(
        "teacher[%s]: pooled OOS f1=%.4f (p=%.4f r=%.4f) | baseline f1=%.4f | "
        "SPX f1=%.4f",
        name,
        pooled.f1,
        pooled.precision,
        pooled.recall,
        b_f1,
        spx.f1,
    )
    return SideReport(
        side=name,
        n_pos=n_pos,
        positive_rate=float((y[:, side] == 1.0).mean()) if y.size else 0.0,
        mean_threshold=mean_thr,
        pooled=pooled,
        spx=spx,
        baseline_f1=b_f1,
        baseline_precision=b_p,
        baseline_recall=b_r,
    )


# --------------------------------------------------------------------------- #
# Reporting.                                                                   #
# --------------------------------------------------------------------------- #


def _fmt_delta(new: float, base: float) -> str:
    """Format a signed F1 delta vs the baseline (e.g. ``+0.081``)."""
    if base != base:
        return "n/a"
    d = new - base
    return f"{'+' if d >= 0 else ''}{d:.3f}"


def _build_scorecard(reports: list[SideReport]) -> Scorecard:
    """Assemble a scorecard with pooled + SPX OOS rows per side."""
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


def _render_report(
    data: PooledDataset,
    reports: list[SideReport],
    args: Args,
    issues: list[str],
    param_count: int,
    device: torch.device,
    spx_present: bool,
) -> str:
    """Render the full TCN scorecard Markdown (setup, OOS, baseline gate, detail)."""
    lines: list[str] = []
    lines.append("# cfd10 TCN teacher — multi-asset POOLED, STRUCTURAL oracle")
    lines.append("")
    lines.append(
        "A dilated causal **TCN** turn detector trained on the stacked daily corpus "
        "and scored out-of-sample with purged + embargoed walk-forward CV "
        "(purge/embargo per asset's own bar clock). One dual-head network emits both "
        "the top and bottom turn logits per window; windows never cross an asset "
        f"boundary. Metric: event precision/recall/F1 at a {_TOLERANCE}-bar tolerance."
    )
    lines.append("")
    lines.append(
        f"**Runtime:** device=`{device}`, CUDA available={torch.cuda.is_available()}, "
        f"TCN trainable params={param_count}."
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- Pool: {data.n_assets} `{_TIMEFRAME}` assets, {data.n_rows} rows after "
        "per-asset warm-up removal."
    )
    lines.append(f"- Features: {data.X.shape[1]} columns (default bank).")
    lines.append(
        f"- TCN: window W={args.window}, channels={args.channels}, dilations=(1,2,4,8), "
        f"epochs<= {args.epochs}, batch={args.batch}, lr={args.lr}, AdamW."
    )
    lines.append(
        f"- STRUCTURAL oracle: nest={_ORACLE_CFG.scale_nest}, "
        f"weight_curve={_ORACLE_CFG.weight_curve!r}, horizon={_ORACLE_CFG.horizon}, "
        f"drawdown_pct={_ORACLE_CFG.drawdown_pct}, tau={_ORACLE_CFG.tau_strong}/"
        f"{_ORACLE_CFG.tau_regular}."
    )
    lines.append(
        f"- CV: purged walk-forward, n_folds={_N_FOLDS}, "
        f"label_horizon={_LABEL_HORIZON}, embargo={_EMBARGO}, pooled_groups=per-asset."
    )
    lines.append(
        "- Per fold: train-split feature standardization; time-tail val slice for "
        "early stop + isotonic calibration; per-side threshold tuned on the train "
        "fold only."
    )
    if not spx_present:
        lines.append("- NOTE: SPX absent from the pool; SPX-subset rows are empty.")
    if issues:
        lines.append("")
        lines.append("**Build issues / subset notes:**")
        for msg in issues:
            lines.append(f"- {msg}")
    lines.append("")

    lines.append("## Out-of-sample event metrics (TCN)")
    lines.append("")
    lines.append(_build_scorecard(reports).to_markdown())
    lines.append("")
    lines.append(
        "`POOL` = all assets pooled; `SPX` = the SPX-subset rows scored from the "
        "*same* pooled-trained models and thresholds."
    )
    lines.append("")

    # The beat-the-baseline gate.
    lines.append("## Beat-the-baseline gate (TCN vs GBDT, same data + folds)")
    lines.append("")
    lines.append("| side | GBDT F1 | TCN POOL F1 | delta F1 | TCN P | TCN R | verdict |")
    lines.append("| :-- | --: | --: | --: | --: | --: | :-- |")
    for rep in reports:
        verdict = "TCN wins" if rep.pooled.f1 > rep.baseline_f1 else "GBDT wins"
        lines.append(
            f"| {rep.side} | {rep.baseline_f1:.4f} | {rep.pooled.f1:.4f} | "
            f"{_fmt_delta(rep.pooled.f1, rep.baseline_f1)} | {rep.pooled.precision:.4f} "
            f"| {rep.pooled.recall:.4f} | {verdict} |"
        )
    lines.append("")
    n_win = sum(1 for r in reports if r.pooled.f1 > r.baseline_f1)
    lines.append(
        f"**Gate:** TCN beats the GBDT baseline on {n_win}/{len(reports)} sides "
        "(pooled OOS event-F1, identical labels / folds)."
    )
    lines.append("")

    lines.append("## Per-side detail")
    lines.append("")
    for rep in reports:
        turn = "top" if rep.side == "high" else "bottom"
        lines.append(f"### {rep.side.upper()} ({turn} turns)")
        lines.append("")
        lines.append(
            f"- Pooled positives: {rep.n_pos} (rate {rep.positive_rate * 100:.3f}%); "
            f"mean per-fold threshold {rep.mean_threshold:.4f}."
        )
        lines.append(
            f"- TCN pooled OOS: P={rep.pooled.precision:.4f}, R={rep.pooled.recall:.4f}, "
            f"F1={rep.pooled.f1:.4f} (tp={rep.pooled.tp}, fp={rep.pooled.fp}, "
            f"fn={rep.pooled.fn}, n_true={rep.pooled.n_true}, n_pred={rep.pooled.n_pred})."
        )
        lines.append(
            f"- TCN SPX-subset OOS: P={rep.spx.precision:.4f}, R={rep.spx.recall:.4f}, "
            f"F1={rep.spx.f1:.4f} (n_true={rep.spx.n_true})."
        )
        lines.append(
            f"- GBDT baseline pooled OOS: P={rep.baseline_precision:.4f}, "
            f"R={rep.baseline_recall:.4f}, F1={rep.baseline_f1:.4f}."
        )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #


def run(args: Args) -> tuple[PooledDataset, list[SideReport]]:
    """Build the pool, fit the TCN per fold, score OOS, compare to GBDT, save card.

    Args:
        args: Parsed CLI arguments.

    Returns:
        ``(dataset, reports)``.

    Raises:
        FileNotFoundError: If the data root is missing.
    """
    set_seed(args.seed)
    device = resolve_device(None)
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    if not _DATA_ROOT.is_dir():
        raise FileNotFoundError(f"fit_teacher: missing data root {_DATA_ROOT}")

    dataset, issues = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=FeatureConfig(),
        oracle_cfg=_ORACLE_CFG,
        timeframe=_TIMEFRAME,
        max_assets=args.max_assets,
    )

    # Report the TCN parameter count up front.
    probe = TCNConfig(in_features=dataset.X.shape[1], channels=args.channels)
    param_count = TCNTurnModel(probe).count_parameters()
    print(f"TCN trainable params = {param_count} (window W={args.window})")

    spx_present = _SPX in dataset.asset_names
    spx_mask = dataset.subset_mask(_SPX) if spx_present else None

    splits = purged_walk_forward(
        dataset.timestamps,
        label_horizon=_LABEL_HORIZON,
        embargo=_EMBARGO,
        n_folds=_N_FOLDS,
        pooled_groups=dataset.groups,
    )

    fold_oos = [
        _fit_one_fold_tcn(dataset, fold_id, fold, args, device)
        for fold_id, fold in enumerate(splits)
    ]

    reports: list[SideReport] = []
    for side, name in _SIDES:
        baseline_prf = _baseline_side(dataset, splits, side)
        reports.append(
            _assemble_side_report(
                dataset, side, name, fold_oos, spx_mask, baseline_prf
            )
        )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    document = _render_report(
        dataset, reports, args, issues, param_count, device, spx_present
    )
    _SCORECARD_PATH.write_text(document, encoding="utf-8")
    logger.info("fit_teacher: wrote scorecard to %s", _SCORECARD_PATH)
    return dataset, reports


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args, run the pooled TCN pipeline, print the scorecard."""
    args = parse_args(argv)
    dataset, reports = run(args)
    print()
    print(_SCORECARD_PATH.read_text(encoding="utf-8"))
    print("-" * 72)
    print(f"  pool: {dataset.n_assets} assets, {dataset.n_rows} rows")
    for rep in reports:
        verdict = "WIN " if rep.pooled.f1 > rep.baseline_f1 else "LOSS"
        print(
            f"  {rep.side.upper():4s}  TCN F1={rep.pooled.f1:.4f}  "
            f"GBDT F1={rep.baseline_f1:.4f}  [{verdict}]"
        )
    print(f"  scorecard -> {_SCORECARD_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
