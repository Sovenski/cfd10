"""QA harness: validate the turn oracle against famous S&P 500 market bottoms.

Run with::

    uv run python pipeline/qa_labels.py

The script loads the long SPX daily ``raw_v16`` export, runs
:func:`cfd10.label_module.label_turns` with the default :class:`OracleConfig`, and
**asserts** that each of four canonical bear-market bottoms

    * 2009-03-09  (GFC trough)
    * 2020-03-23  (COVID crash trough)
    * 2002-10-09  (dot-com bear trough)
    * 2008-11-20  (the major GFC interim low)

falls on a ``bottom_tier`` of ``"strong"`` or ``"regular"``. Because the oracle's
strict multi-scale extreme can sit one or two sessions away from the exact printed
daily low (and TradingView stamps a bar at its UTC session open, so the calendar
date can be off by a day), the check accepts the best bottom tier within a small
``+/-MATCH_TOLERANCE_BARS`` window around the bar nearest each target date.

On success it writes an overview plot to ``outputs/qa_labels_spx1d.png`` marking
the detected strong/regular bottoms and the four famous lows, then prints a
summary. Any failed assertion raises (non-zero exit) so the orchestrator notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write a PNG, never open a window.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfd10.data_module import load_csv
from cfd10.label_module import OracleConfig, label_turns
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_SPX_1D: Path = _REPO_ROOT / "data" / "raw_v16" / "SP_SPX, 1D_a20e0.csv"
_OUT_DIR: Path = _REPO_ROOT / "outputs"
_PLOT_PATH: Path = _OUT_DIR / "qa_labels_spx1d.png"

# Famous S&P 500 bottoms the oracle must flag (as bottom 'strong'/'regular').
_FAMOUS_LOWS: tuple[str, ...] = (
    "2009-03-09",
    "2020-03-23",
    "2002-10-09",
    "2008-11-20",
)

# Sessions of slack when matching a calendar date to the oracle's strict extreme.
_MATCH_TOLERANCE_BARS: int = 3

_PASS_TIERS: frozenset[str] = frozenset({"strong", "regular"})


@dataclass(frozen=True)
class FamousLowCheck:
    """Result of validating one famous-low date against the oracle.

    Attributes:
        label: The target date string (e.g. ``"2009-03-09"``).
        bar_index: Index of the bar nearest ``label`` in the loaded frame.
        bar_date: The actual calendar date of ``bar_index``.
        best_tier: The strongest bottom tier within the match window
            (``"strong"`` > ``"regular"`` > ``"none"``).
        best_score: The maximum bottom score within the match window.
        passed: Whether ``best_tier`` is in :data:`_PASS_TIERS`.
    """

    label: str
    bar_index: int
    bar_date: pd.Timestamp
    best_tier: str
    best_score: float
    passed: bool


def epoch_to_dates(time_s: np.ndarray) -> pd.Series:
    """Convert canonical epoch-second timestamps to dates (negative epochs -> NaT).

    The canonical schema keeps ``time`` as raw ``int64`` epoch seconds and tolerates
    negative (pre-1970) values for the 1871-era SPX history, which pandas datetimes
    cannot represent cleanly. Here we map only the **non-negative** timestamps to
    naive ``datetime64[ns]`` and leave the rest as ``NaT``.

    Args:
        time_s: 1-D array of epoch seconds (may contain negatives).

    Returns:
        A ``datetime64[ns]`` :class:`pandas.Series` aligned to ``time_s`` with
        ``NaT`` wherever the source epoch was negative.
    """
    nonneg = time_s >= 0
    clipped = np.where(nonneg, time_s, 0).astype("int64")
    converted = pd.to_datetime(pd.Series(clipped), unit="s", utc=True).dt.tz_localize(None)
    return converted.where(pd.Series(nonneg, index=converted.index), pd.NaT)


def _nearest_bar(dates: pd.Series, target: pd.Timestamp) -> int:
    """Index of the dated bar closest to ``target`` (ignoring NaT bars)."""
    valid = dates.dropna()
    if valid.empty:
        raise ValueError("qa_labels: no dated bars available to match against")
    return int((valid - target).abs().idxmin())


def check_famous_low(
    label: str,
    dates: pd.Series,
    bottom_score: np.ndarray,
    bottom_tier: np.ndarray,
    tolerance: int = _MATCH_TOLERANCE_BARS,
) -> FamousLowCheck:
    """Validate a single famous-low date against the oracle's bottom labels.

    Args:
        label: Target date string.
        dates: Per-bar dates (NaT for pre-1970 bars).
        bottom_score: Per-bar bottom scores from :func:`label_turns`.
        bottom_tier: Per-bar bottom tiers from :func:`label_turns`.
        tolerance: Half-width (bars) of the acceptance window around the nearest
            bar.

    Returns:
        A :class:`FamousLowCheck` with the best tier/score in the window and the
        pass flag.
    """
    target = pd.Timestamp(label)
    idx = _nearest_bar(dates, target)
    lo = max(idx - tolerance, 0)
    hi = min(idx + tolerance + 1, bottom_score.shape[0])

    window_tiers = set(bottom_tier[lo:hi])
    if "strong" in window_tiers:
        best_tier = "strong"
    elif "regular" in window_tiers:
        best_tier = "regular"
    else:
        best_tier = "none"
    best_score = float(np.max(bottom_score[lo:hi]))

    return FamousLowCheck(
        label=label,
        bar_index=idx,
        bar_date=dates.iloc[idx],
        best_tier=best_tier,
        best_score=best_score,
        passed=best_tier in _PASS_TIERS,
    )


def _plot_overview(
    df: pd.DataFrame,
    dates: pd.Series,
    labels: pd.DataFrame,
    checks: list[FamousLowCheck],
    out_path: Path,
) -> None:
    """Render close price with detected bottoms and the famous lows; save a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dated = dates.notna().to_numpy()
    x = dates.to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)
    bottom_tier = labels["bottom_tier"].to_numpy()

    strong = dated & (bottom_tier == "strong")
    regular = dated & (bottom_tier == "regular")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x[dated], close[dated], color="#444444", lw=0.7, label="SPX close")
    ax.scatter(
        x[regular], close[regular], s=14, c="#1f77b4", alpha=0.6,
        label="regular bottom", zorder=3,
    )
    ax.scatter(
        x[strong], close[strong], s=26, c="#d62728", alpha=0.85,
        label="strong bottom", zorder=4,
    )

    for chk in checks:
        bx = x[chk.bar_index]
        by = close[chk.bar_index]
        ax.annotate(
            chk.label,
            xy=(bx, by),
            xytext=(0, -38),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#2ca02c",
            arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.0},
        )

    ax.set_yscale("log")
    ax.set_title(
        "cfd10 oracle QA — SPX 1D bottom labels vs. famous bear-market lows\n"
        "(log price; green arrows = asserted famous lows)"
    )
    ax.set_xlabel("date")
    ax.set_ylabel("close (log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("qa_labels: wrote plot to %s", out_path)


def run_qa(cfg: OracleConfig | None = None) -> list[FamousLowCheck]:
    """Load SPX 1D, label turns, validate famous lows, and write the QA plot.

    Args:
        cfg: Oracle configuration; defaults to :class:`OracleConfig` (the tuned
            defaults under which all four famous lows clear ``"regular"``).

    Returns:
        The per-low :class:`FamousLowCheck` results.

    Raises:
        FileNotFoundError: If the SPX export is missing.
        AssertionError: If any famous low fails to land on a strong/regular bottom.
    """
    cfg = cfg or OracleConfig()
    if not _SPX_1D.is_file():
        raise FileNotFoundError(f"qa_labels: missing SPX export {_SPX_1D}")

    logger.info("qa_labels: loading %s", _SPX_1D)
    df = load_csv(_SPX_1D)
    dates = epoch_to_dates(df["time"].to_numpy())

    labels = label_turns(df, cfg)
    bottom_score = labels["bottom_score"].to_numpy(dtype=np.float64)
    bottom_tier = labels["bottom_tier"].to_numpy()

    checks = [
        check_famous_low(label, dates, bottom_score, bottom_tier)
        for label in _FAMOUS_LOWS
    ]

    for chk in checks:
        logger.info(
            "qa_labels: %s -> bar %d (%s): best bottom tier=%s score=%.3f [%s]",
            chk.label,
            chk.bar_index,
            chk.bar_date.date(),
            chk.best_tier,
            chk.best_score,
            "PASS" if chk.passed else "FAIL",
        )

    _plot_overview(df, dates, labels, checks, _PLOT_PATH)

    failures = [c.label for c in checks if not c.passed]
    assert not failures, (
        "qa_labels: these famous lows did not land on a strong/regular bottom "
        f"within +/-{_MATCH_TOLERANCE_BARS} bars: {failures}"
    )
    return checks


def main() -> None:
    """Entry point: run the QA and print a compact summary table."""
    checks = run_qa()
    print("\ncfd10 oracle QA — famous SPX bottoms")
    print("-" * 64)
    for chk in checks:
        print(
            f"  {chk.label}  bar={chk.bar_index:>6d}  date={chk.bar_date.date()}  "
            f"tier={chk.best_tier:<8s} score={chk.best_score:.3f}  "
            f"{'PASS' if chk.passed else 'FAIL'}"
        )
    print("-" * 64)
    print(f"  all {len(checks)} famous lows passed; plot -> {_PLOT_PATH}")


if __name__ == "__main__":
    main()
