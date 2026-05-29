"""Threshold-free RANKING re-evaluation of the pooled GBDT turn detector.

Run a tiny smoke with::

    uv run python pipeline/eval_ranking.py --max-assets 3

(the orchestrator runs the full corpus separately; do NOT run it here unbounded).

Event precision/recall/F1 collapse to zero the instant the detector is
mis-thresholded, which hides a model that *ranks* pivots well. This script
sidesteps the threshold entirely. It collects the pooled GBDT's **out-of-fold**
class-1 probabilities from purged + embargoed walk-forward CV (the same leakage-safe
split the baseline uses) and scores them with the n-weighted ranking metrics in
:mod:`cfd10.eval_module.ranking`:

* **n-weighted Average Precision** - PR-AUC with ``sample_weight`` = the oracle
  pivot n-score, so nailing a heavy (200-scale) pivot counts for more than a light
  one. This is the headline, threshold-free number.
* **precision@k / recall@k / lift@k** for ``k in {25, n_true}`` - top-of-list hit
  rate and concentration over chance.

Every metric is reported **pooled** (all assets) and on the **SPX subset**.

Two ablation knobs probe whether ranking improves with richer / de-biased features:

* ``FeatureConfig(include_overextension=True)`` - append the classic top-tell
  overextension / vol-regime block.
* per-asset z-score **de-bias** - standardise each feature within each asset (fitted
  on the fold's train rows only, applied to test), so a turn on gold and a turn on
  SPX sit on a common scale and cross-asset level differences cannot dominate the
  ranking.

There are NO forward-return or PnL metrics here, by design: the label is the
oracle's structural pivot tier and these scores judge ranking against that label.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from cfd10.cv_module import Fold, purged_walk_forward
from cfd10.data_module import PooledDataset, build_pooled_dataset
from cfd10.eval_module import average_precision, lift_at_k, precision_at_k, recall_at_k
from cfd10.feature_module import FeatureConfig
from cfd10.label_module import OracleConfig
from cfd10.teacher_module import GBDTConfig
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DATA_ROOT: Path = _REPO_ROOT / "data" / "raw_v16"
_OUT_DIR: Path = _REPO_ROOT / "outputs"
_SCORECARD_PATH: Path = _OUT_DIR / "ranking_scorecard.md"

_SPX: str = "SPX"
_SEED: int = 42
_TIMEFRAME: str = "1D"
_K_FIXED: int = 25

# The famous-lows-validated STRUCTURAL oracle (matches pipeline.fit_pooled).
_ORACLE_CFG = OracleConfig(
    scale_nest=(20, 50, 100, 200),
    weight_curve="linear",
    drawdown_pct=0.10,
    horizon=60,
    tau_strong=0.60,
    tau_regular=0.30,
)

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
    embargo=5,
    n_folds=6,
    seed=_SEED,
)

_POSITIVE_TIERS: frozenset[str] = frozenset({"strong", "regular"})

_FEATURE_NAME_WARNING = (
    "X does not have valid feature names, but LGBMClassifier was fitted"
)


# --------------------------------------------------------------------------- #
# Result containers.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RankSlice:
    """Threshold-free ranking metrics over one row slice (pooled or SPX-only).

    Attributes:
        n: Number of scored rows in the slice.
        n_true: Number of positive (strong/regular) labels in the slice.
        ap: Unweighted average precision (PR-AUC).
        ap_weighted: n-weighted average precision (sample_weight = oracle score).
        p_at_25 / r_at_25 / lift_at_25: precision/recall/lift at ``k = 25``.
        p_at_ntrue / r_at_ntrue / lift_at_ntrue: precision/recall/lift at
            ``k = n_true`` (the perfect-recall budget).
    """

    n: int
    n_true: int
    ap: float
    ap_weighted: float
    p_at_25: float
    r_at_25: float
    lift_at_25: float
    p_at_ntrue: float
    r_at_ntrue: float
    lift_at_ntrue: float


@dataclass(frozen=True)
class VariantSideResult:
    """One (variant, side) cell: pooled + SPX ranking slices.

    Attributes:
        variant: Human-readable variant label (e.g. ``"base"``).
        side: ``"high"`` (top turns) or ``"low"`` (bottom turns).
        n_features: Number of feature columns in this variant.
        pooled: All-assets ranking slice.
        spx: SPX-subset ranking slice (``None`` if SPX absent from the pool).
    """

    variant: str
    side: str
    n_features: int
    pooled: RankSlice
    spx: RankSlice | None


# --------------------------------------------------------------------------- #
# Out-of-fold scoring.                                                         #
# --------------------------------------------------------------------------- #


def _predict_proba(
    model: lgb.LGBMClassifier, X: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Class-1 probabilities, silencing the benign LightGBM feature-name warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_FEATURE_NAME_WARNING, category=UserWarning
        )
        proba = model.predict_proba(X)[:, 1]
    return np.ascontiguousarray(proba, dtype=np.float64)


def _zscore_debias(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    g_train: NDArray[np.int64],
    g_test: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-asset z-score features: fit mean/std on train rows, apply to both.

    For every asset present in the training rows the per-column mean and std are
    estimated on that asset's *train* rows only (no test leakage) and used to
    standardise both its train and test rows. Assets/columns with a degenerate
    (zero / non-finite) std are left mean-centred only (unit divisor), and assets
    unseen in train fall back to the global train statistics.

    Args:
        X_train: Train feature rows.
        X_test: Test feature rows.
        g_train: Per-row asset id for ``X_train``.
        g_test: Per-row asset id for ``X_test``.

    Returns:
        ``(X_train_z, X_test_z)`` standardised copies aligned to the inputs.
    """
    x_tr = X_train.copy()
    x_te = X_test.copy()

    # Global train fallback statistics for assets unseen in this fold's train set.
    global_mean = np.nanmean(X_train, axis=0) if X_train.shape[0] else np.zeros(
        X_train.shape[1]
    )
    global_std = np.nanstd(X_train, axis=0) if X_train.shape[0] else np.ones(
        X_train.shape[1]
    )
    global_std = np.where(np.isfinite(global_std) & (global_std > 1e-12), global_std, 1.0)

    seen = set(np.unique(g_train).tolist())
    for asset in np.unique(np.concatenate([g_train, g_test])):
        if asset in seen:
            rows = g_train == asset
            mean = X_train[rows].mean(axis=0)
            std = X_train[rows].std(axis=0)
            std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        else:
            mean, std = global_mean, global_std
        tr_rows = g_train == asset
        te_rows = g_test == asset
        if tr_rows.any():
            x_tr[tr_rows] = (X_train[tr_rows] - mean) / std
        if te_rows.any():
            x_te[te_rows] = (X_test[te_rows] - mean) / std
    return x_tr, x_te


def _oof_scores(
    data: PooledDataset,
    splits: list[Fold],
    side: str,
    debias: bool,
    cfg: GBDTConfig = _GBDT_CFG,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Collect pooled out-of-fold class-1 probabilities for one side.

    Each fold trains a class-balanced GBDT on its train block (oracle sample
    weights, ``scale_pos_weight = n_neg/n_pos``) and scores its held-out test block;
    the test probabilities are scattered back into a pooled OOF array. Test blocks
    are disjoint, so every row that appears in some test block gets exactly one OOF
    score. When ``debias`` is set, features are per-asset z-scored with train-only
    statistics before fitting / scoring.

    Args:
        data: The pooled corpus.
        splits: Purged walk-forward folds over the pooled time axis.
        side: ``"high"`` or ``"low"``.
        debias: Whether to per-asset z-score features (train-fitted).
        cfg: Base GBDT configuration (``scale_pos_weight`` overridden per side).

    Returns:
        ``(oof_scores, y, rank_weight)`` over all pooled rows. ``oof_scores`` is
        ``NaN`` on rows never in any test block; ``y`` is the binary label;
        ``rank_weight`` is the n-weighting sample weight ``w = 1 + oracle_score`` on
        turns, ``1`` on non-turns (see the AP note below).
    """
    tier_col = "top_tier" if side == "high" else "bottom_tier"
    weight_col = "top_weight" if side == "high" else "bottom_weight"

    tiers = data.labels[tier_col].to_numpy()
    y = np.isin(tiers, list(_POSITIVE_TIERS)).astype(np.int64)
    oracle_score = data.labels[weight_col].to_numpy(dtype=np.float64)
    # SAMPLE-WEIGHT CONVENTION (matches the pooled baseline): w = 1 + oracle score
    # on turns, 1 on non-turns. The same vector is the GBDT fit weight AND the
    # ranking (AP) sample weight. Using the RAW oracle score as the AP weight is a
    # trap: it is exactly 0 on every negative row, which zero-weights the entire
    # negative class and makes the weighted average precision degenerate to 1.0.
    # Flooring negatives at 1.0 keeps them in the PR curve while a heavy (large-
    # scale) pivot still carries proportionally more weight than a light one.
    rank_weight = np.where(y == 1, 1.0 + oracle_score, 1.0).astype(np.float64)
    fit_weight = rank_weight

    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    spw = float(n_neg / max(n_pos, 1))
    cfg = replace(cfg, scale_pos_weight=spw)
    set_seed(cfg.seed)

    X_values = np.ascontiguousarray(data.X.to_numpy(dtype=np.float64))
    groups = data.groups
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)

    params = cfg.lgb_params()
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        x_tr = X_values[train_idx]
        x_te = X_values[test_idx]
        if debias:
            x_tr, x_te = _zscore_debias(
                x_tr, x_te, groups[train_idx], groups[test_idx]
            )
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr, y[train_idx], sample_weight=fit_weight[train_idx])
        oof[test_idx] = _predict_proba(model, x_te)
        logger.debug(
            "eval_ranking[%s%s]: fold %d scored %d test rows",
            side,
            " z" if debias else "",
            fold_id,
            test_idx.size,
        )

    return oof, y, rank_weight


# --------------------------------------------------------------------------- #
# Metric assembly.                                                             #
# --------------------------------------------------------------------------- #


def _rank_slice(
    scores: NDArray[np.float64],
    y: NDArray[np.int64],
    weight: NDArray[np.float64],
    keep: NDArray[np.bool_] | None,
) -> RankSlice | None:
    """Compute the threshold-free ranking metrics over an OOF row slice.

    Only rows with a finite OOF score are scored (rows never in a test block are
    dropped). ``keep`` optionally restricts to a subset (e.g. SPX) *before* the
    finite-score filter.

    Args:
        scores: Pooled OOF class-1 probabilities (``NaN`` where unscored).
        y: Binary labels aligned to ``scores``.
        weight: Per-row AP sample weight ``w = 1 + oracle_score`` on turns (``1`` on
            non-turns), aligned to ``scores``.
        keep: Optional boolean mask over all rows; ``None`` scores the full pool.

    Returns:
        The :class:`RankSlice`, or ``None`` if the slice has no scored rows or no
        positives (no PR curve / top-k is defined).
    """
    mask = np.isfinite(scores)
    if keep is not None:
        mask &= keep
    s = scores[mask]
    yy = y[mask].astype(np.int64)
    ww = weight[mask].astype(np.float64)

    n = int(s.shape[0])
    n_true = int(yy.sum())
    if n == 0 or n_true == 0:
        return None

    ap = average_precision(s, yy)
    ap_w = average_precision(s, yy, sample_weight=ww)

    k_fixed = min(_K_FIXED, n)
    k_true = min(max(n_true, 1), n)

    return RankSlice(
        n=n,
        n_true=n_true,
        ap=ap,
        ap_weighted=ap_w,
        p_at_25=precision_at_k(s, yy, k_fixed),
        r_at_25=recall_at_k(s, yy, k_fixed),
        lift_at_25=lift_at_k(s, yy, k_fixed),
        p_at_ntrue=precision_at_k(s, yy, k_true),
        r_at_ntrue=recall_at_k(s, yy, k_true),
        lift_at_ntrue=lift_at_k(s, yy, k_true),
    )


def eval_variant(
    data: PooledDataset,
    splits: list[Fold],
    variant: str,
    debias: bool,
    spx_mask: NDArray[np.bool_] | None,
) -> list[VariantSideResult]:
    """Score both sides for one feature variant (pooled + SPX ranking slices).

    Args:
        data: The pooled corpus for this variant's feature set.
        splits: Purged walk-forward folds over the pooled time axis.
        variant: Human-readable variant label.
        debias: Whether to per-asset z-score features (train-fitted).
        spx_mask: Boolean mask over pooled rows selecting SPX (``None`` if absent).

    Returns:
        One :class:`VariantSideResult` per side (``high`` then ``low``).
    """
    results: list[VariantSideResult] = []
    for side in ("high", "low"):
        scores, y, weight = _oof_scores(data, splits, side, debias)
        pooled = _rank_slice(scores, y, weight, None)
        spx = _rank_slice(scores, y, weight, spx_mask) if spx_mask is not None else None
        if pooled is None:
            logger.warning(
                "eval_ranking[%s/%s]: no scored positives pooled; skipping", variant, side
            )
            continue
        logger.info(
            "eval_ranking[%s/%s]: pooled AP=%.4f (n-weighted=%.4f), "
            "P@25=%.3f R@n_true=%.3f (n_true=%d)",
            variant,
            side,
            pooled.ap,
            pooled.ap_weighted,
            pooled.p_at_25,
            pooled.r_at_ntrue,
            pooled.n_true,
        )
        results.append(
            VariantSideResult(
                variant=variant,
                side=side,
                n_features=data.X.shape[1],
                pooled=pooled,
                spx=spx,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Report.                                                                      #
# --------------------------------------------------------------------------- #


def _slice_row(label: str, sl: RankSlice | None) -> str:
    """Render one Markdown table row for a (scope) ranking slice."""
    if sl is None:
        return f"| {label} | - | - | - | - | - | - | - | - |"
    return (
        f"| {label} | {sl.n_true} | {sl.ap:.4f} | {sl.ap_weighted:.4f} | "
        f"{sl.p_at_25:.3f} | {sl.r_at_25:.3f} | {sl.lift_at_25:.2f} | "
        f"{sl.p_at_ntrue:.3f} | {sl.lift_at_ntrue:.2f} |"
    )


def _render_report(
    data: PooledDataset,
    results: list[VariantSideResult],
    spx_present: bool,
    issues: list[str],
) -> str:
    """Render the full threshold-free ranking scorecard (ASCII only)."""
    lines: list[str] = []
    lines.append("# cfd10 threshold-free RANKING scorecard - pooled GBDT detector")
    lines.append("")
    lines.append(
        "Out-of-fold ranking quality of the pooled GBDT turn detector, scored with "
        "n-weighted average precision and top-k hit rates (no decision threshold). "
        "Scores come from purged + embargoed walk-forward CV; each row is scored on "
        "its held-out fold only. NO forward-return or PnL metrics - the label is the "
        "oracle's structural pivot tier."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- Pool: {data.n_assets} `{_TIMEFRAME}` assets, {data.n_rows} rows after "
        "per-asset warm-up removal."
    )
    lines.append(
        f"- STRUCTURAL oracle: nest={_ORACLE_CFG.scale_nest}, "
        f"horizon={_ORACLE_CFG.horizon}, drawdown_pct={_ORACLE_CFG.drawdown_pct}, "
        f"tau_strong={_ORACLE_CFG.tau_strong}, tau_regular={_ORACLE_CFG.tau_regular}."
    )
    lines.append(
        f"- CV: purged walk-forward, n_folds={_GBDT_CFG.n_folds}, "
        f"label_horizon={_GBDT_CFG.label_horizon}, embargo={_GBDT_CFG.embargo}, "
        "pooled_groups=per-asset."
    )
    lines.append(
        "- AP n-weighting: sample_weight = 1 + oracle pivot n-score on turns (1 on "
        "non-turns), so heavy (large-scale) pivots count more while negatives still "
        "weigh in. `lift` is precision@k / base positive rate (1.0 = chance)."
    )
    lines.append(f"- k_fixed = {_K_FIXED}; k = n_true is the perfect-recall budget.")
    if not spx_present:
        lines.append("- NOTE: SPX absent from the pool; SPX rows are empty.")
    if issues:
        lines.append("")
        lines.append("**Build issues / subset notes:**")
        for msg in issues:
            lines.append(f"- {msg}")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append(
        "- `base`: default 36-feature bank. "
        "`+overext`: FeatureConfig(include_overextension=True). "
        "`+overext+zdebias`: same features, per-asset train-fitted z-score de-bias."
    )
    lines.append("")

    # One table per side, rows = (variant, scope).
    for side in ("high", "low"):
        name = "top" if side == "high" else "bottom"
        side_results = [r for r in results if r.side == side]
        lines.append(f"## {side.upper()} ({name} turns)")
        lines.append("")
        lines.append(
            "| variant / scope | n_true | AP | AP (n-wt) | P@25 | R@25 | lift@25 | "
            "P@n_true | lift@n_true |"
        )
        lines.append("| :-- | --: | --: | --: | --: | --: | --: | --: | --: |")
        for res in side_results:
            lines.append(
                _slice_row(f"{res.variant} ({res.n_features}f) / POOL", res.pooled)
            )
            lines.append(_slice_row(f"{res.variant} / SPX", res.spx))
        lines.append("")

    lines.append("## Reading the result")
    lines.append("")
    lines.append(
        "The headline is **AP (n-wt)**: threshold-free, and weighted so heavy "
        "(large-scale) pivots dominate. Compare `+overext` and `+overext+zdebias` "
        "against `base` per side - a higher n-weighted AP means the richer / "
        "de-biased features rank the structurally important pivots better, even "
        "where event-F1 looked flat."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #


def run(
    seed: int = _SEED, max_assets: int | None = None
) -> list[VariantSideResult]:
    """Build the pool(s), collect OOF scores per variant/side, save the scorecard.

    Three variants are evaluated: ``base`` (default bank), ``+overext``
    (overextension block on), and ``+overext+zdebias`` (overextension on plus a
    per-asset train-fitted z-score de-bias).

    Args:
        seed: Global RNG seed for reproducibility.
        max_assets: Optional cap on the pooled asset count (forwarded to
            :func:`build_pooled_dataset`). Use a small value for a smoke run.

    Returns:
        The flat list of :class:`VariantSideResult` across variants and sides.

    Raises:
        FileNotFoundError: If the data root is missing.
    """
    set_seed(seed)
    if not _DATA_ROOT.is_dir():
        raise FileNotFoundError(f"eval_ranking: missing data root {_DATA_ROOT}")

    base_cfg = FeatureConfig()
    oe_cfg = FeatureConfig(include_overextension=True)

    data_base, issues = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=base_cfg,
        oracle_cfg=_ORACLE_CFG,
        timeframe=_TIMEFRAME,
        max_assets=max_assets,
    )
    data_oe, _ = build_pooled_dataset(
        _DATA_ROOT,
        feature_cfg=oe_cfg,
        oracle_cfg=_ORACLE_CFG,
        timeframe=_TIMEFRAME,
        max_assets=max_assets,
    )
    for msg in issues:
        logger.info("eval_ranking: build note - %s", msg)

    spx_present = _SPX in data_base.asset_names
    spx_mask_base = data_base.subset_mask(_SPX) if spx_present else None
    spx_mask_oe = data_oe.subset_mask(_SPX) if _SPX in data_oe.asset_names else None

    splits_base = purged_walk_forward(
        data_base.timestamps,
        label_horizon=_GBDT_CFG.label_horizon,
        embargo=_GBDT_CFG.embargo,
        n_folds=_GBDT_CFG.n_folds,
        pooled_groups=data_base.groups,
    )
    splits_oe = purged_walk_forward(
        data_oe.timestamps,
        label_horizon=_GBDT_CFG.label_horizon,
        embargo=_GBDT_CFG.embargo,
        n_folds=_GBDT_CFG.n_folds,
        pooled_groups=data_oe.groups,
    )

    results: list[VariantSideResult] = []
    results += eval_variant(data_base, splits_base, "base", False, spx_mask_base)
    results += eval_variant(data_oe, splits_oe, "+overext", False, spx_mask_oe)
    results += eval_variant(
        data_oe, splits_oe, "+overext+zdebias", True, spx_mask_oe
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    document = _render_report(data_base, results, spx_present, issues)
    _SCORECARD_PATH.write_text(document, encoding="utf-8")
    logger.info("eval_ranking: wrote scorecard to %s", _SCORECARD_PATH)
    return results


def main() -> None:
    """Entry point: parse ``--max-assets``, run the ranking eval, print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Cap the pooled asset count (use a small value for a smoke run).",
    )
    parser.add_argument(
        "--seed", type=int, default=_SEED, help="Global RNG seed."
    )
    args = parser.parse_args()

    results = run(seed=args.seed, max_assets=args.max_assets)
    print()
    print(f"  ranking scorecard -> {_SCORECARD_PATH}")
    print("-" * 72)
    for res in results:
        pooled = res.pooled
        print(
            f"  {res.variant:18s} {res.side.upper():4s} "
            f"AP={pooled.ap:.4f} AP_nwt={pooled.ap_weighted:.4f} "
            f"P@25={pooled.p_at_25:.3f} R@n_true={pooled.r_at_ntrue:.3f} "
            f"(n_true={pooled.n_true})"
        )


if __name__ == "__main__":
    main()
