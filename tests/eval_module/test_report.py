"""Tests for ``cfd10.eval_module.report`` — the per-slice Scorecard.

Checks the count-derived row factory, the immutable add/extend semantics, and a
deterministic, sorted Markdown rendering with and without deflation values.
"""

from __future__ import annotations

import math

from cfd10.eval_module.report import Scorecard, ScoreRow


def test_score_row_from_counts_derives_prf() -> None:
    """``ScoreRow.from_counts`` computes precision/recall/F1 from the triple."""
    row = ScoreRow.from_counts("long", "SPX", "oos", tp=3, fp=2, fn=1)
    assert math.isclose(row.precision, 0.6, abs_tol=1e-12)
    assert math.isclose(row.recall, 0.75, abs_tol=1e-12)
    assert math.isclose(row.f1, 2.0 / 3.0, abs_tol=1e-12)
    assert row.key == ("long", "SPX", "oos")


def test_scorecard_add_is_immutable() -> None:
    """``add`` returns a new scorecard; the original is unchanged."""
    empty = Scorecard()
    row = ScoreRow.from_counts("long", "SPX", "era1", 1, 0, 0)
    one = empty.add(row)
    assert len(empty.rows) == 0
    assert len(one.rows) == 1
    two = one.add(ScoreRow.from_counts("short", "SPX", "era1", 2, 1, 1))
    assert len(one.rows) == 1
    assert len(two.rows) == 2


def test_to_markdown_is_sorted_and_well_formed() -> None:
    """Rows render sorted by (side, asset, era) with a valid header/separator."""
    card = Scorecard().extend(
        (
            ScoreRow.from_counts("short", "SPX", "era2", 1, 1, 1),
            ScoreRow.from_counts("long", "SPX", "era1", 4, 0, 0),
            ScoreRow.from_counts("long", "BTC", "era1", 2, 2, 0),
        )
    )
    md = card.to_markdown()
    lines = md.splitlines()

    # Header + separator + 3 data rows.
    assert len(lines) == 5
    assert lines[0].startswith("| side | asset | era |")
    # Separator row uses Markdown alignment markers.
    assert set(lines[1].replace("|", "").replace(" ", "")) <= set(":-")

    # Data rows are sorted: ('long','BTC','era1') < ('long','SPX','era1')
    #                       < ('short','SPX','era2').
    data = lines[2:]
    assert data[0].split("|")[1].strip() == "long"
    assert data[0].split("|")[2].strip() == "BTC"
    assert data[1].split("|")[2].strip() == "SPX"
    assert data[2].split("|")[1].strip() == "short"


def test_to_markdown_renders_deflated_and_blank() -> None:
    """The deflated cell shows a 4-dp value when set and is blank otherwise."""
    card = Scorecard().extend(
        (
            ScoreRow.from_counts("long", "SPX", "era1", 3, 1, 1, deflated=1.2345),
            ScoreRow.from_counts("long", "SPX", "era2", 3, 1, 1),
        )
    )
    md = card.to_markdown()
    lines = md.splitlines()
    # era1 row carries the deflated value formatted to 4 decimals.
    assert "1.2345" in lines[2]
    # era2 row leaves the deflated cell empty (trailing "|  |").
    last_cell = lines[3].rstrip().rsplit("|", 2)[1].strip()
    assert last_cell == ""


def test_empty_scorecard_renders_header_only() -> None:
    """An empty scorecard renders just the header and alignment rows."""
    md = Scorecard().to_markdown()
    lines = md.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("| side |")
