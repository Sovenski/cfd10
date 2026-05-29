"""Tests for ``cfd10.feature_module.momentum``.

These tests pin the Pine semantics of ``speculatores_v15_presets_gold.pine``
lines 443-453 (per-side momentum / momentum-velocity block) for a single
lookback ``L``. The feature layer surfaces insufficient-history bars as ``NaN``
rather than Pine's ``nz``-swallowed ``0.0`` (the ``nz`` is re-applied later by the
assembly step); the warm-up assertions below lock that contract in.
"""

from __future__ import annotations

import numpy as np

from cfd10.feature_module.momentum import (
    _mom_divergence_scalar,
    _mom_velocity_scalar,
    _price_return_scalar,
    mom_divergence,
    mom_velocity,
    price_return,
)


def _random_market(n: int = 400, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive close series and a non-negative volume series."""
    rng = np.random.default_rng(seed)
    close = (100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))).astype(np.float64)
    volume = rng.integers(1, 10_000, size=n).astype(np.float64)
    return close, volume


def test_price_return_reference_small_fixture() -> None:
    """``price_return`` equals ``(close - close[L]) / close[L]`` bar by bar (L443)."""
    close = np.array([10.0, 11.0, 9.0, 12.0, 15.0], dtype=np.float64)
    out = price_return(close, L=2)
    # i < 2 -> warm-up -> NaN; i >= 2 -> (close[i] - close[i-2]) / close[i-2].
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2], (9.0 - 10.0) / 10.0)
    np.testing.assert_allclose(out[3], (12.0 - 11.0) / 11.0)
    np.testing.assert_allclose(out[4], (15.0 - 9.0) / 9.0)


def test_mom_divergence_reference_small_fixture() -> None:
    """``mom_divergence = price_ret * vol_ret`` with ``max(volume[L], 1)`` (L444-445)."""
    close = np.array([10.0, 11.0, 9.0, 12.0, 15.0], dtype=np.float64)
    # Include a zero historical volume to exercise the max(., 1) denominator clamp.
    volume = np.array([0.0, 100.0, 200.0, 50.0, 400.0], dtype=np.float64)
    L = 2
    out = mom_divergence(close, volume, L)

    assert np.isnan(out[0]) and np.isnan(out[1])
    # i = 2: price_ret = (9 - 10)/10 = -0.1; vol_ret = (200 - 0)/max(0, 1) = 200.
    pr2 = (9.0 - 10.0) / 10.0
    vr2 = (200.0 - 0.0) / max(0.0, 1.0)
    np.testing.assert_allclose(out[2], pr2 * vr2)
    # i = 3: price_ret = (12 - 11)/11; vol_ret = (50 - 100)/max(100, 1).
    pr3 = (12.0 - 11.0) / 11.0
    vr3 = (50.0 - 100.0) / max(100.0, 1.0)
    np.testing.assert_allclose(out[3], pr3 * vr3)
    # i = 4: price_ret = (15 - 9)/9; vol_ret = (400 - 200)/max(200, 1).
    pr4 = (15.0 - 9.0) / 9.0
    vr4 = (400.0 - 200.0) / max(200.0, 1.0)
    np.testing.assert_allclose(out[4], pr4 * vr4)


def test_mom_velocity_reference_small_fixture() -> None:
    """``mom_velocity = price_ret - price_ret[1]`` (L446); needs two valid returns."""
    close = np.array([10.0, 11.0, 9.0, 12.0, 15.0], dtype=np.float64)
    L = 2
    pr = price_return(close, L)
    out = mom_velocity(close, L)

    # price_ret valid from i = L = 2, so velocity valid only from i = L + 1 = 3.
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    np.testing.assert_allclose(out[3], pr[3] - pr[2])
    np.testing.assert_allclose(out[4], pr[4] - pr[3])


def test_warmup_is_nan_for_all_three() -> None:
    """The first ``L`` bars of price_return/mom_divergence and first ``L+1`` of
    mom_velocity are NaN; everything after is finite for clean data."""
    close, volume = _random_market()
    L = 14

    pr = price_return(close, L)
    md = mom_divergence(close, volume, L)
    mv = mom_velocity(close, L)

    assert np.all(np.isnan(pr[:L]))
    assert np.all(np.isnan(md[:L]))
    assert np.all(np.isnan(mv[: L + 1]))

    assert np.all(np.isfinite(pr[L:]))
    assert np.all(np.isfinite(md[L:]))
    assert np.all(np.isfinite(mv[L + 1 :]))


def test_outputs_are_float_arrays_aligned_to_input() -> None:
    """All three return float64 ``np.ndarray`` of the same length as the input."""
    close, volume = _random_market(n=123)
    L = 5
    for arr in (
        price_return(close, L),
        mom_divergence(close, volume, L),
        mom_velocity(close, L),
    ):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (123,)


def test_finite_values_are_bounded_and_consistent() -> None:
    """Post-warm-up values are finite and the documented identities hold.

    ``mom_divergence == price_return * vol_ret`` and
    ``mom_velocity[i] == price_return[i] - price_return[i-1]`` are re-derived
    independently here to guard against drift in the vectorized arithmetic.
    """
    close, volume = _random_market(seed=3)
    L = 8

    pr = price_return(close, L)
    md = mom_divergence(close, volume, L)
    mv = mom_velocity(close, L)

    # All post-warm-up outputs are finite (real-world bounded-range/finiteness).
    assert np.all(np.isfinite(pr[L:]))
    assert np.all(np.isfinite(md[L:]))
    assert np.all(np.isfinite(mv[L + 1 :]))

    # Independent recomputation of vol_ret and the two product/diff identities.
    vol_back = volume[:-L]
    denom = np.maximum(vol_back, 1.0)
    vol_ret = (volume[L:] - vol_back) / denom
    np.testing.assert_allclose(md[L:], pr[L:] * vol_ret, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(mv[L + 1 :], pr[L + 1 :] - pr[L:-1], atol=1e-12, rtol=0.0)


def test_scalar_and_vectorized_agree() -> None:
    """Scalar per-bar references agree with the vectorized API to 1e-9.

    NaN warm-up bars are compared with ``equal_nan`` so the warm-up regions must
    coincide as well as the finite values.
    """
    close, volume = _random_market(n=300, seed=99)
    for L in (1, 2, 5, 21):
        n = close.shape[0]

        vec_pr = price_return(close, L)
        sca_pr = np.array(
            [_price_return_scalar(close, L, i) for i in range(n)], dtype=np.float64
        )
        np.testing.assert_allclose(vec_pr, sca_pr, atol=1e-9, rtol=0.0, equal_nan=True)

        vec_md = mom_divergence(close, volume, L)
        sca_md = np.array(
            [_mom_divergence_scalar(close, volume, L, i) for i in range(n)],
            dtype=np.float64,
        )
        np.testing.assert_allclose(vec_md, sca_md, atol=1e-9, rtol=0.0, equal_nan=True)

        vec_mv = mom_velocity(close, L)
        sca_mv = np.array(
            [_mom_velocity_scalar(close, L, i) for i in range(n)], dtype=np.float64
        )
        np.testing.assert_allclose(vec_mv, sca_mv, atol=1e-9, rtol=0.0, equal_nan=True)


def test_constant_series_gives_zero_return() -> None:
    """A flat close series yields exactly zero price-return (no spurious drift)."""
    close = np.full(50, 42.0, dtype=np.float64)
    out = price_return(close, L=10)
    assert np.all(np.isnan(out[:10]))
    np.testing.assert_array_equal(out[10:], np.zeros(40, dtype=np.float64))
