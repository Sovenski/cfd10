"""Fit the GBDT baseline on SPX 1D and report the first out-of-sample result.

Run with::

    uv run python pipeline/fit_baseline.py

This wires the whole chain end-to-end on the long SPX daily ``raw_v16`` export:

1. load the canonical OHLCV frame (:func:`cfd10.data_module.load_csv`);
2. build the dense 36-feature bank (:func:`cfd10.feature_module.build_feature_matrix`);
3. label forward-looking turns with the oracle (:func:`cfd10.label_module.label_turns`);
4. drop warm-up ``NaN`` feature rows and re-index features / labels / timestamps to
   a common, contiguous bar axis;
5. build purged + embargoed walk-forward folds
   (:func:`cfd10.cv_module.purged_walk_forward`);
6. fit the LightGBM baseline **separately for the HIGH (top) and LOW (bottom)
   side** (:func:`cfd10.teacher_module.fit_gbdt_cv`), tuning each fold's decision
   threshold on the train fold only;
7. assemble a :class:`cfd10.eval_module.Scorecard`, print it, and save it to
   ``outputs/baseline_spx1d_scorecard.md``.

This is an **oracle-vs-detector** out-of-sample evaluation only. The Pine-signal
("Run 3") comparison is not possible here — it needs the user's exported Pine
signals — so this script reports OOS-vs-oracle and nothing else. The two sides are
reported independently and without tuning to manufacture a result on either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cfd10.cv_module import Fold, purged_walk_forward
from cfd10.data_module import load_csv
from cfd10.eval_module import Scorecard, ScoreRow
from cfd10.feature_module import FeatureConfig, build_feature_matrix
from cfd10.label_module import OracleConfig, label_turns
from cfd10.teacher_module import GBDTConfig, fit_gbdt_cv
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_SPX_1D: Path = _REPO_ROOT / "data" / "raw_v16" / "SP_SPX, 1D_a20e0.csv"
_OUT_DIR: Path = _REPO_ROOT / "outputs"
_SCORECARD_PATH: Path = _OUT_DIR / "baseline_spx1d_scorecard.md"

_ASSET: str = "SPX"
_SEED: int = 42

# Oracle tuned so positives are present but sparse (~0.7%): a 5%+ reversal at the
# 5-40 bar scales. This is the "sane default tau" the baseline trains against.
_ORACLE_CFG = OracleConfig(
    scale_nest=(5, 10, 20, 40),
    horizon=40,
    drawdown_pct=0.05,
    tau_strong=0.5,
    tau_regular=0.25,
)

# Baseline GBDT config. ``label_horizon`` mirrors the oracle horizon so the purge
# removes exactly the bars whose label window overlaps a test block.
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


@dataclass(frozen=True)
class PreparedData:
    """The aligned, warm-up-free inputs shared by both sides.

    Attributes:
        X: Dense feature matrix with warm-up ``NaN`` rows removed and a contiguous
            ``RangeIndex`` (so split indices address its rows directly).
        labels: Oracle labels re-indexed to match ``X`` row-for-row.
        timestamps: Per-row epoch timestamps aligned to ``X``.
        n_dropped: Number of warm-up rows dropped from the raw frame.
    """

    X: pd.DataFrame
    labels: pd.DataFrame
    timestamps: np.ndarray
    n_dropped: int


@dataclass(frozen=True)
class SideReport:
    """One side's OOS event metrics, ready for the scorecard.

    Attributes:
        side: ``"high"`` (top turns) or ``"low"`` (bottom turns).
        result: The raw :func:`fit_gbdt_cv` result dict.
    """

    side: str
    result: dict[str, object]


def prepare_data(
    oracle_cfg: OracleConfig = _ORACLE_CFG,
    feature_cfg: FeatureConfig | None = None,
) -> PreparedData:
    """Load SPX 1D, build features + labels, and align them warm-up-free.

    Args:
        oracle_cfg: Oracle configuration for :func:`label_turns`.
        feature_cfg: Feature-bank configuration; defaults to :class:`FeatureConfig`.

    Returns:
        A :class:`PreparedData` with features, labels and timestamps on a common,
        contiguous bar axis.

    Raises:
        FileNotFoundError: If the SPX export is missing.
    """
    if not _SPX_1D.is_file():
        raise FileNotFoundError(f"fit_baseline: missing SPX export {_SPX_1D}")

    feature_cfg = feature_cfg or FeatureConfig()
    logger.info("fit_baseline: loading %s", _SPX_1D)
    df = load_csv(_SPX_1D)

    X, names = build_feature_matrix(df, feature_cfg)
    labels = label_turns(df, oracle_cfg)

    # Drop warm-up rows: any bar with a NaN in any feature column is undecided for
    # the model. Re-index everything to a fresh contiguous axis so CV split indices
    # address the aligned rows directly.
    valid = ~X.isna().any(axis=1).to_numpy()
    n_dropped = int((~valid).sum())
    X_aligned = X.loc[valid].reset_index(drop=True)
    labels_aligned = labels.loc[valid].reset_index(drop=True)
    timestamps = df.loc[valid, "time"].to_numpy(dtype=np.int64)

    logger.info(
        "fit_baseline: %d feature columns, dropped %d warm-up rows, %d bars remain",
        len(names),
        n_dropped,
        len(X_aligned),
    )
    return PreparedData(
        X=X_aligned,
        labels=labels_aligned,
        timestamps=timestamps,
        n_dropped=n_dropped,
    )


def _side_columns(side: str) -> tuple[str, str]:
    """Return the ``(tier_col, weight_col)`` label columns for ``side``.

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
    raise ValueError(f"fit_baseline: unknown side {side!r} (expected 'high'/'low')")


def fit_side(
    data: PreparedData,
    splits: list[Fold],
    side: str,
    cfg: GBDTConfig = _GBDT_CFG,
) -> SideReport:
    """Fit the GBDT baseline for one side and return its OOS result.

    Args:
        data: The aligned inputs from :func:`prepare_data`.
        splits: Purged walk-forward folds over ``data.timestamps``.
        side: ``"high"`` or ``"low"``.
        cfg: Baseline configuration.

    Returns:
        A :class:`SideReport` wrapping the :func:`fit_gbdt_cv` result.
    """
    tier_col, weight_col = _side_columns(side)
    tiers = data.labels[tier_col].to_numpy()
    y = np.isin(tiers, list(_POSITIVE_TIERS)).astype(np.int64)

    # The oracle weight column IS the turn score, which is 0 on every non-turn
    # bar. Used verbatim it would zero the entire negative class and the
    # classifier would collapse to "everything is a turn". So we floor negatives
    # at a unit baseline and let the oracle score *add* confidence emphasis to
    # positives: w = 1 + score on turns, 1 on non-turns.
    oracle_score = data.labels[weight_col].to_numpy(dtype=np.float64)
    sample_weight = np.where(y == 1, 1.0 + oracle_score, 1.0).astype(np.float64)

    n_pos = int(y.sum())
    logger.info(
        "fit_baseline[%s]: %d positives / %d bars (%.3f%%)",
        side,
        n_pos,
        y.shape[0],
        100.0 * n_pos / max(y.shape[0], 1),
    )
    result = fit_gbdt_cv(data.X, y, sample_weight, splits, cfg)
    return SideReport(side=side, result=result)


def build_scorecard(reports: list[SideReport]) -> Scorecard:
    """Assemble a :class:`Scorecard` of pooled OOS rows, one per side.

    Args:
        reports: Per-side fit results.

    Returns:
        A :class:`Scorecard` with one ``(side, SPX, oos)`` row per side.
    """
    card = Scorecard()
    for rep in reports:
        res = rep.result
        card = card.add(
            ScoreRow(
                side=rep.side,
                asset=_ASSET,
                era="oos",
                tp=int(res["oos_tp"]),
                fp=int(res["oos_fp"]),
                fn=int(res["oos_fn"]),
                precision=float(res["oos_precision"]),
                recall=float(res["oos_recall"]),
                f1=float(res["oos_f1"]),
            )
        )
    return card


def _render_report(
    data: PreparedData,
    reports: list[SideReport],
    splits: list[Fold],
    card: Scorecard,
) -> str:
    """Render the full Markdown scorecard document (header + table + notes)."""
    lines: list[str] = []
    lines.append("# cfd10 GBDT baseline — SPX 1D out-of-sample scorecard")
    lines.append("")
    lines.append(
        "First out-of-sample result: a per-side LightGBM turn detector scored "
        "against the forward-looking oracle with purged + embargoed walk-forward "
        "CV. Decision thresholds are tuned on each train fold only; the metric is "
        "event precision/recall/F1 at a "
        f"{_GBDT_CFG.tolerance}-bar tolerance."
    )
    lines.append("")
    lines.append(
        "**Scope:** oracle-vs-detector only. The Pine-signal ('Run 3') comparison "
        "is not possible here — it needs exported Pine signals — so no signal "
        "baseline is reported."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Asset: `{_ASSET}` 1D, {len(data.X)} bars after warm-up "
                 f"(dropped {data.n_dropped}).")
    lines.append(f"- Features: {data.X.shape[1]} columns (default bank).")
    lines.append(
        f"- Oracle: nest={_ORACLE_CFG.scale_nest}, horizon={_ORACLE_CFG.horizon}, "
        f"drawdown_pct={_ORACLE_CFG.drawdown_pct}, "
        f"tau_strong={_ORACLE_CFG.tau_strong}, tau_regular={_ORACLE_CFG.tau_regular}."
    )
    lines.append(
        f"- CV: purged walk-forward, n_folds={_GBDT_CFG.n_folds}, "
        f"label_horizon={_GBDT_CFG.label_horizon}, embargo={_GBDT_CFG.embargo}."
    )
    lines.append(
        f"- GBDT: num_leaves={_GBDT_CFG.num_leaves}, "
        f"n_estimators={_GBDT_CFG.n_estimators}, lr={_GBDT_CFG.learning_rate}, "
        f"scale_pos_weight={_GBDT_CFG.scale_pos_weight}."
    )
    lines.append("")
    lines.append("## Pooled out-of-sample event metrics")
    lines.append("")
    lines.append(card.to_markdown())
    lines.append("")

    # Per-side summary with label balance and per-fold F1 spread.
    lines.append("## Per-side detail")
    lines.append("")
    for rep in reports:
        res = rep.result
        fold_f1s = [round(f.f1, 4) for f in res["folds"]]
        lines.append(
            f"### {rep.side.upper()} ({'top' if rep.side == 'high' else 'bottom'} turns)"
        )
        lines.append("")
        lines.append(
            f"- Label positive rate: {res['positive_rate'] * 100:.3f}% "
            f"({res['n_true']} OOS true events pooled)."
        )
        lines.append(
            f"- Pooled OOS: precision={res['oos_precision']:.4f}, "
            f"recall={res['oos_recall']:.4f}, F1={res['oos_f1']:.4f} "
            f"(tp={res['oos_tp']}, fp={res['oos_fp']}, fn={res['oos_fn']}, "
            f"n_pred={res['n_pred']})."
        )
        lines.append(f"- Mean per-fold F1: {res['mean_fold_f1']:.4f}.")
        lines.append(f"- Per-fold OOS F1: {fold_f1s}.")
        lines.append("")
    return "\n".join(lines)


def run(seed: int = _SEED) -> tuple[Scorecard, list[SideReport]]:
    """Run the full baseline pipeline on SPX 1D and save the scorecard.

    Args:
        seed: Global RNG seed for reproducibility.

    Returns:
        ``(scorecard, reports)`` — the assembled scorecard and the per-side fit
        results.

    Raises:
        FileNotFoundError: If the SPX export is missing.
    """
    set_seed(seed)
    data = prepare_data()
    splits = purged_walk_forward(
        data.timestamps,
        label_horizon=_GBDT_CFG.label_horizon,
        embargo=_GBDT_CFG.embargo,
        n_folds=_GBDT_CFG.n_folds,
    )

    reports = [fit_side(data, splits, side) for side in ("high", "low")]
    card = build_scorecard(reports)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    document = _render_report(data, reports, splits, card)
    _SCORECARD_PATH.write_text(document, encoding="utf-8")
    logger.info("fit_baseline: wrote scorecard to %s", _SCORECARD_PATH)
    return card, reports


def main() -> None:
    """Entry point: run the pipeline and print the scorecard document."""
    card, reports = run()
    print()
    print(_SCORECARD_PATH.read_text(encoding="utf-8"))
    print("-" * 72)
    for rep in reports:
        res = rep.result
        print(
            f"  {rep.side.upper():4s}  OOS F1={res['oos_f1']:.4f}  "
            f"P={res['oos_precision']:.4f}  R={res['oos_recall']:.4f}  "
            f"(n_true={res['n_true']}, n_pred={res['n_pred']}, "
            f"pos_rate={res['positive_rate'] * 100:.3f}%)"
        )
    print(f"  scorecard -> {_SCORECARD_PATH}")


if __name__ == "__main__":
    main()
