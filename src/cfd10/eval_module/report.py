"""Scorecard: per-(side, asset, era) evaluation rows with Markdown rendering.

Evaluation results are sliced along three axes — the trade *side* (e.g. ``long``
/ ``short``), the *asset* (e.g. ``SPX``), and the *era* (a named time regime,
e.g. ``1990-2007`` or ``oos``). :class:`ScoreRow` holds one such slice's headline
detector scores plus the confusion counts and any deflation diagnostics;
:class:`Scorecard` collects the rows and renders them as a stable, sorted
GitHub-flavoured Markdown table via :meth:`Scorecard.to_markdown`.

The row is a frozen dataclass (immutable, hashable identity over its key), and
:meth:`Scorecard.add` returns a new :class:`Scorecard` rather than mutating in
place, keeping the container value-like and safe to share.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from cfd10.eval_module.metrics import prf_from_counts
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "ScoreRow",
    "Scorecard",
]


@dataclass(frozen=True)
class ScoreRow:
    """One evaluation slice keyed by ``(side, asset, era)``.

    Attributes:
        side: Trade side, e.g. ``"long"`` / ``"short"``.
        asset: Instrument symbol, e.g. ``"SPX"``.
        era: Named time regime / split, e.g. ``"1990-2007"`` or ``"oos"``.
        tp: True-positive event count.
        fp: False-positive event count.
        fn: False-negative event count.
        precision: Event precision in ``[0, 1]``.
        recall: Event recall in ``[0, 1]``.
        f1: Event F1 in ``[0, 1]``.
        deflated: Optional deflated metric (e.g. DSR / haircut Sharpe); ``None``
            when not computed for this slice.
        n_trials: Optional number of trials behind ``deflated`` (for provenance).
    """

    side: str
    asset: str
    era: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    deflated: float | None = None
    n_trials: int | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the ``(side, asset, era)`` sort/identity key."""
        return (self.side, self.asset, self.era)

    @classmethod
    def from_counts(
        cls,
        side: str,
        asset: str,
        era: str,
        tp: int,
        fp: int,
        fn: int,
        deflated: float | None = None,
        n_trials: int | None = None,
    ) -> ScoreRow:
        """Build a row from confusion counts, deriving precision/recall/F1.

        Args:
            side: Trade side.
            asset: Instrument symbol.
            era: Named time regime / split.
            tp: True-positive count.
            fp: False-positive count.
            fn: False-negative count.
            deflated: Optional deflated metric for the slice.
            n_trials: Optional trial count behind ``deflated``.

        Returns:
            A populated :class:`ScoreRow` with precision/recall/F1 computed via
            :func:`cfd10.eval_module.metrics.prf_from_counts`.
        """
        precision, recall, f1 = prf_from_counts(tp, fp, fn)
        return cls(
            side=side,
            asset=asset,
            era=era,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            deflated=deflated,
            n_trials=n_trials,
        )


# Markdown table header (column order is the public contract of the report).
_COLUMNS: tuple[str, ...] = (
    "side",
    "asset",
    "era",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "deflated",
)


@dataclass(frozen=True)
class Scorecard:
    """An immutable collection of :class:`ScoreRow` results.

    Attributes:
        rows: Tuple of score rows. Use :meth:`add` / :meth:`extend` to derive a
            new scorecard with extra rows (the instance itself is never mutated).
    """

    rows: tuple[ScoreRow, ...] = field(default_factory=tuple)

    def add(self, row: ScoreRow) -> Scorecard:
        """Return a new scorecard with ``row`` appended.

        Args:
            row: The score row to add.

        Returns:
            A new :class:`Scorecard` instance including ``row``.
        """
        return replace(self, rows=(*self.rows, row))

    def extend(self, rows: tuple[ScoreRow, ...]) -> Scorecard:
        """Return a new scorecard with ``rows`` appended.

        Args:
            rows: Score rows to add.

        Returns:
            A new :class:`Scorecard` instance including ``rows``.
        """
        return replace(self, rows=(*self.rows, *rows))

    def sorted_rows(self) -> list[ScoreRow]:
        """Return the rows ordered by ``(side, asset, era)`` for stable output."""
        return sorted(self.rows, key=lambda r: r.key)

    @staticmethod
    def _fmt_float(value: float) -> str:
        """Format a metric float to 4 decimals."""
        return f"{value:.4f}"

    def _fmt_deflated(self, row: ScoreRow) -> str:
        """Render the deflated cell, blank when absent."""
        if row.deflated is None:
            return ""
        return self._fmt_float(row.deflated)

    def to_markdown(self) -> str:
        """Render the scorecard as a GitHub-flavoured Markdown table.

        Rows are sorted by ``(side, asset, era)`` so the output is deterministic
        regardless of insertion order. A scorecard with no rows renders the
        header and alignment row only.

        Returns:
            The Markdown table as a single string (no trailing newline).
        """
        header = "| " + " | ".join(_COLUMNS) + " |"
        # Right-align the numeric columns; left-align the three key columns.
        aligns = []
        for col in _COLUMNS:
            aligns.append(":--" if col in ("side", "asset", "era") else "--:")
        sep = "| " + " | ".join(aligns) + " |"

        lines = [header, sep]
        for row in self.sorted_rows():
            cells = [
                row.side,
                row.asset,
                row.era,
                str(row.tp),
                str(row.fp),
                str(row.fn),
                self._fmt_float(row.precision),
                self._fmt_float(row.recall),
                self._fmt_float(row.f1),
                self._fmt_deflated(row),
            ]
            lines.append("| " + " | ".join(cells) + " |")

        logger.debug("Scorecard.to_markdown: rendered %d rows", len(self.rows))
        return "\n".join(lines)
