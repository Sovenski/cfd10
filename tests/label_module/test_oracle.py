"""Tests for ``cfd10.label_module.oracle`` — the forward-looking turn oracle.

The oracle is *ground truth*: it is allowed to look forward, but only within
``cfg.horizon`` bars of the bar being labelled. These tests pin three load-bearing
properties of that contract:

(a) **Scale selectivity.** A deep V-bottom that is simultaneously a 200-bar and a
    50-bar extreme (and reverses far past the drawdown threshold) scores *much*
    higher than a shallow wiggle that is only a 2-bar local extreme.
(b) **Monotone weight curve.** ``weight(n)`` increases with the nest scale, and the
    smallest scale is ~0 relative to the largest, for both ``"linear"`` and
    ``"quadratic"`` curves.
(c) **Causal horizon.** Truncating the series anywhere at or beyond ``t + horizon``
    leaves the label at bar ``t`` unchanged (no leakage past the horizon).

Plus the frozen-config / registry-factory contract shared by every cfd10 module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfd10.label_module import (
    ORACLE_SIDE_FACTORY,
    OracleConfig,
    get_side_fn,
    label_turns,
    register_side,
)

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "top_score",
    "bottom_score",
    "top_tier",
    "bottom_tier",
    "top_weight",
    "bottom_weight",
)


# --------------------------------------------------------------------------- #
# Synthetic-frame builders.                                                   #
# --------------------------------------------------------------------------- #


def _ohlcv_from_close(close: np.ndarray, t0: int = 0) -> pd.DataFrame:
    """Wrap a close path into a canonical OHLCV frame (flat bars, unit volume)."""
    n = close.shape[0]
    time = np.arange(t0, t0 + n, dtype=np.int64)
    return pd.DataFrame(
        {
            "time": time,
            "open": close.astype(np.float64),
            "high": close.astype(np.float64),
            "low": close.astype(np.float64),
            "close": close.astype(np.float64),
            "volume": np.ones(n, dtype=np.float64),
        }
    )


def _deep_v_series() -> tuple[pd.DataFrame, int, int]:
    """A long flat plateau, one deep V-bottom, then a noisy 2-bar wiggle.

    Layout (indices):
        - ``[0, 600)``    : flat plateau at 100 with tiny noise (no extremes).
        - ``bottom``      : a single deep V trough far below the plateau, framed
          by wide shoulders so it is the strict minimum of a >200-bar window, and
          followed by a recovery that blows past any sane ``drawdown_pct``.
        - ``wiggle``      : a shallow down-tick that is only a 2-bar local minimum
          (its neighbours are barely higher), with no large reversal.

    Returns ``(df, bottom_idx, wiggle_idx)``.
    """
    rng = np.random.default_rng(0)
    plateau = 100.0 + rng.normal(0.0, 0.02, size=600)

    # Deep V: descend over ~120 bars to a sharp trough, then recover over ~140
    # bars well above the trough (a >100% bounce off the low).
    down = np.linspace(100.0, 40.0, 120)
    up = np.linspace(40.0, 130.0, 140)
    v = np.concatenate([down, up[1:]])  # share the trough sample once
    bottom_idx = 600 + (len(down) - 1)

    # Long flat tail so the bottom keeps a full forward horizon, then a tiny
    # 2-bar dip that is a strict local min over its immediate neighbours only.
    tail = 130.0 + rng.normal(0.0, 0.02, size=400)
    wiggle_idx = 600 + len(v) + 200
    close = np.concatenate([plateau, v, tail])
    # Carve a shallow 2-bar dip directly at wiggle_idx: a strict local min over
    # its immediate neighbours, with a sub-threshold (< drawdown_pct) reversal.
    close[wiggle_idx] = close[wiggle_idx - 1] - 0.5
    close[wiggle_idx + 1] = close[wiggle_idx] + 0.4

    df = _ohlcv_from_close(close)
    return df, bottom_idx, wiggle_idx


# --------------------------------------------------------------------------- #
# (config) frozen dataclass + registry/factory contract.                      #
# --------------------------------------------------------------------------- #


def test_config_is_frozen() -> None:
    """``OracleConfig`` is immutable (frozen dataclass)."""
    cfg = OracleConfig()
    try:
        cfg.horizon = 5  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 - we only assert it raises.
        assert exc.__class__.__name__ in {"FrozenInstanceError", "AttributeError"}
    else:  # pragma: no cover - mutation must fail.
        raise AssertionError("OracleConfig should be frozen")


def test_registry_factory_roundtrip() -> None:
    """A side reducer registered via the decorator resolves through the factory."""

    @register_side("unit_test_probe_side")
    def _probe(values: np.ndarray, cfg: OracleConfig) -> np.ndarray:  # noqa: ANN001
        return np.zeros_like(values)

    try:
        assert "unit_test_probe_side" in ORACLE_SIDE_FACTORY
        assert get_side_fn("unit_test_probe_side") is _probe
    finally:
        ORACLE_SIDE_FACTORY.pop("unit_test_probe_side", None)


# --------------------------------------------------------------------------- #
# (b) monotone, bottom-heavy weight curve.                                    #
# --------------------------------------------------------------------------- #


def test_weight_increases_with_scale_linear() -> None:
    """Linear curve: weight(200) > weight(50) > weight(2), and weight(2) ~ 0."""
    cfg = OracleConfig(scale_nest=(2, 50, 200), weight_curve="linear")
    w2, w50, w200 = cfg.weight(2), cfg.weight(50), cfg.weight(200)

    assert w200 > w50 > w2
    # The smallest scale is negligible next to the largest.
    assert w2 <= 1e-9
    assert w2 < 0.05 * w200


def test_weight_increases_with_scale_quadratic() -> None:
    """Quadratic curve is also monotone and even more bottom-suppressing."""
    cfg = OracleConfig(scale_nest=(2, 50, 200), weight_curve="quadratic")
    w2, w50, w200 = cfg.weight(2), cfg.weight(50), cfg.weight(200)

    assert w200 > w50 > w2
    assert w2 <= 1e-9

    # Quadratic suppresses the *middle* rank harder than linear does (relative to
    # the top rank): the n=50 share of the top weight is strictly smaller.
    lin = OracleConfig(scale_nest=(2, 50, 200), weight_curve="linear")
    assert (w50 / w200) < (lin.weight(50) / lin.weight(200))


def test_weight_unknown_scale_raises() -> None:
    """Asking for a scale outside the nest is a programming error."""
    cfg = OracleConfig(scale_nest=(20, 50, 100, 200))
    try:
        cfg.weight(7)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("weight() of an out-of-nest scale should raise")


def test_unknown_weight_curve_raises() -> None:
    """An unregistered weight curve name is rejected."""
    cfg = OracleConfig(weight_curve="nope")
    try:
        cfg.weight(cfg.scale_nest[0])
    except (KeyError, ValueError):
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown weight_curve should raise")


# --------------------------------------------------------------------------- #
# label_turns output schema.                                                  #
# --------------------------------------------------------------------------- #


def test_label_turns_schema_and_alignment() -> None:
    """Output has the documented columns, is index-aligned, and tiers are valid."""
    df, _, _ = _deep_v_series()
    cfg = OracleConfig()
    out = label_turns(df, cfg)

    assert list(out.columns) == list(_REQUIRED_COLUMNS)
    assert out.index.equals(df.index)
    assert len(out) == len(df)

    for score_col in ("top_score", "bottom_score", "top_weight", "bottom_weight"):
        vals = out[score_col].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(vals)), f"{score_col} must be finite"
        assert vals.min() >= -1e-9 and vals.max() <= 1.0 + 1e-9, (
            f"{score_col} must be a normalized score in [0, 1]"
        )

    # score doubles as the sample weight.
    np.testing.assert_allclose(
        out["top_score"].to_numpy(), out["top_weight"].to_numpy()
    )
    np.testing.assert_allclose(
        out["bottom_score"].to_numpy(), out["bottom_weight"].to_numpy()
    )

    valid = {"strong", "regular", "none"}
    assert set(out["top_tier"].unique()) <= valid
    assert set(out["bottom_tier"].unique()) <= valid


# --------------------------------------------------------------------------- #
# (a) deep multi-scale V-bottom >> shallow 2-bar wiggle.                       #
# --------------------------------------------------------------------------- #


def test_deep_v_outscores_shallow_wiggle() -> None:
    """A genuine 200-and-50-bar V-bottom scores far above a 2-bar-only wiggle."""
    df, bottom_idx, wiggle_idx = _deep_v_series()
    cfg = OracleConfig()
    out = label_turns(df, cfg)

    deep = float(out["bottom_score"].iloc[bottom_idx])
    wiggle = float(out["bottom_score"].iloc[wiggle_idx])

    assert deep > 0.5, f"deep V-bottom should score high, got {deep}"
    assert wiggle < 0.2, f"shallow 2-bar wiggle should score low, got {wiggle}"
    assert deep > 3.0 * max(wiggle, 1e-6), "deep bottom must dominate the wiggle"

    # And it should clear at least the 'regular' tier.
    assert out["bottom_tier"].iloc[bottom_idx] in {"strong", "regular"}
    assert out["bottom_tier"].iloc[wiggle_idx] == "none"


def test_top_side_mirrors_bottom() -> None:
    """A deep inverted-V (peak) is detected on the top side, symmetric to bottoms."""
    df_v, bottom_idx, _ = _deep_v_series()
    # Reflect the close path to turn the V-bottom into an inverted-V peak.
    close = df_v["close"].to_numpy(dtype=np.float64)
    peak_close = float(close.max()) + 1.0 - close
    df_peak = _ohlcv_from_close(peak_close)

    cfg = OracleConfig()
    out = label_turns(df_peak, cfg)

    top = float(out["top_score"].iloc[bottom_idx])
    assert top > 0.5, f"reflected V should be a strong top, got {top}"
    assert out["top_tier"].iloc[bottom_idx] in {"strong", "regular"}


# --------------------------------------------------------------------------- #
# (c) causal horizon: no leakage past t + horizon.                            #
# --------------------------------------------------------------------------- #


def test_causal_horizon_truncation_invariant() -> None:
    """Truncating at/after ``t + horizon`` does not change the label at ``t``.

    For every candidate bar ``t``, the oracle may only consult bars up to
    ``t + horizon``. Therefore cutting the series at ``t + horizon + k`` (k >= 0)
    must leave the entire label row at ``t`` byte-for-byte identical.
    """
    df, bottom_idx, _ = _deep_v_series()
    cfg = OracleConfig(horizon=60)
    full = label_turns(df, cfg)

    t = bottom_idx
    # Cut exactly at the horizon edge (keep bars [0, t + horizon]).
    cut_edge = df.iloc[: t + cfg.horizon + 1].copy()
    cut_extra = df.iloc[: t + cfg.horizon + 25].copy()

    for cut in (cut_edge, cut_extra):
        truncated = label_turns(cut, cfg)
        for col in ("bottom_score", "bottom_tier", "bottom_weight"):
            assert truncated[col].iloc[t] == full[col].iloc[t], (
                f"label at t={t} changed under truncation for column {col!r}: "
                f"{truncated[col].iloc[t]!r} != {full[col].iloc[t]!r}"
            )


def test_no_forward_window_yields_zero_score() -> None:
    """The last ``horizon`` bars cannot be confirmed, so their scores are 0."""
    df, _, _ = _deep_v_series()
    cfg = OracleConfig(horizon=60)
    out = label_turns(df, cfg)

    tail = out.iloc[len(df) - cfg.horizon :]
    assert np.all(tail["top_score"].to_numpy() == 0.0)
    assert np.all(tail["bottom_score"].to_numpy() == 0.0)
    assert set(tail["top_tier"].unique()) <= {"none"}
    assert set(tail["bottom_tier"].unique()) <= {"none"}
