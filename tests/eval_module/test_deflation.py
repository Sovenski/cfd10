"""Tests for ``cfd10.eval_module.deflation`` — DSR haircut and CSCV PBO.

These pin the qualitative guarantees of the multiple-testing guards:

* the expected-maximum benchmark grows with the number of trials;
* ``deflated_metric`` falls below the observed value as trials accumulate, in
  both the haircut and DSR-probability modes;
* ``pbo_cscv`` returns a probability in ``[0, 1]`` and separates an overfit
  (random-noise) configuration matrix from a genuinely-skilled one.
"""

from __future__ import annotations

import numpy as np

from cfd10.eval_module.deflation import (
    deflated_metric,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
)


def test_expected_max_sharpe_zero_for_single_trial() -> None:
    """One mean-zero trial has expected maximum 0; more trials raise it."""
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(10) > 0.0
    assert expected_max_sharpe(1000) > expected_max_sharpe(10)


def test_expected_max_scales_with_dispersion() -> None:
    """The benchmark scales linearly with the cross-trial dispersion."""
    base = expected_max_sharpe(50, trials_std=1.0)
    scaled = expected_max_sharpe(50, trials_std=2.0)
    assert np.isclose(scaled, 2.0 * base, rtol=1e-9)
    # Zero dispersion collapses the benchmark to 0.
    assert expected_max_sharpe(50, trials_std=0.0) == 0.0


def test_deflated_haircut_below_observed_when_many_trials() -> None:
    """Haircut mode: deflated < observed for n_trials > 1 and monotone in trials."""
    observed = 2.0
    single = deflated_metric(observed, n_trials=1)
    many = deflated_metric(observed, n_trials=500)
    huge = deflated_metric(observed, n_trials=5000)
    # A single trial applies no haircut.
    assert np.isclose(single, observed, rtol=1e-12)
    # More trials -> strictly smaller deflated metric, all below observed.
    assert many < observed
    assert huge < many


def test_deflated_dsr_probability_below_observed_and_in_unit_interval() -> None:
    """DSR mode returns a probability in [0,1] that shrinks as trials grow."""
    observed = 1.5
    dsr_few = deflated_metric(observed, n_trials=2, n_obs=250)
    dsr_many = deflated_metric(observed, n_trials=2000, n_obs=250)
    for value in (dsr_few, dsr_many):
        assert 0.0 <= value <= 1.0
    # The DSR probability is below the raw Sharpe (different scale, but the
    # requirement is "deflated < observed") and decreases with more trials.
    assert dsr_few < observed
    assert dsr_many < dsr_few


def test_probabilistic_sharpe_ratio_monotone_in_observed() -> None:
    """PSR rises with the observed Sharpe and lies in [0, 1]."""
    low = probabilistic_sharpe_ratio(0.1, benchmark_sr=0.0, n_obs=200)
    high = probabilistic_sharpe_ratio(1.0, benchmark_sr=0.0, n_obs=200)
    assert 0.0 <= low <= high <= 1.0
    # An observed Sharpe equal to the benchmark gives PSR == 0.5.
    assert np.isclose(
        probabilistic_sharpe_ratio(0.5, benchmark_sr=0.5, n_obs=200), 0.5, atol=1e-9
    )


def test_pbo_in_unit_interval() -> None:
    """PBO is a probability on synthetic matrices regardless of structure."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        mat = rng.normal(size=(120, 8))
        pbo = pbo_cscv(mat, n_splits=8)
        assert 0.0 <= pbo <= 1.0


def test_pbo_high_for_pure_noise() -> None:
    """Pure-noise configurations have no persistent edge -> PBO near 0.5+.

    With i.i.d. noise the in-sample winner is essentially random out-of-sample,
    so its OoS rank is uniform and PBO should sit around one half (well away
    from 0). We assert a loose lower bound to avoid flakiness.
    """
    rng = np.random.default_rng(42)
    mat = rng.normal(size=(200, 20))
    pbo = pbo_cscv(mat, n_splits=10)
    assert 0.0 <= pbo <= 1.0
    assert pbo > 0.2


def test_pbo_low_for_dominant_configuration() -> None:
    """One configuration that dominates every period is not overfit -> PBO == 0.

    Configuration 0 has a large positive mean every period; the rest are noise.
    It wins in-sample and stays on top out-of-sample in every CSCV split, so its
    OoS rank is always the best and the overfitting probability is zero.
    """
    rng = np.random.default_rng(1)
    mat = rng.normal(scale=0.1, size=(200, 10))
    mat[:, 0] += 5.0  # Dominant, persistent edge.
    pbo = pbo_cscv(mat, n_splits=10)
    assert pbo == 0.0


def test_pbo_rejects_bad_shape_and_odd_splits() -> None:
    """Input validation: 1-D matrix, single config, and odd n_splits all raise."""
    rng = np.random.default_rng(5)
    good = rng.normal(size=(100, 4))
    for bad_call in (
        lambda: pbo_cscv(rng.normal(size=100), n_splits=4),  # 1-D
        lambda: pbo_cscv(rng.normal(size=(100, 1)), n_splits=4),  # single config
        lambda: pbo_cscv(good, n_splits=5),  # odd splits
        lambda: pbo_cscv(good, n_splits=3),  # odd splits
    ):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for invalid pbo_cscv input")


def test_deflated_metric_rejects_bad_trials() -> None:
    """``n_trials < 1`` is rejected."""
    try:
        deflated_metric(1.0, n_trials=0)
    except ValueError:
        return
    raise AssertionError("n_trials=0 should have raised ValueError")
