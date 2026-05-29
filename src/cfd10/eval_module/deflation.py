"""Multiple-testing deflation: deflated metric and CSCV overfitting probability.

Selecting the best of many backtested configurations inflates the winner's
apparent skill — the more trials, the higher the best in-sample score even with
no true edge. This module supplies two complementary guards from the
Bailey / Lopez de Prado line of work:

* :func:`deflated_metric` haircuts an observed Sharpe-like statistic by the score
  a skill-free researcher would *expect* to obtain as the maximum over
  ``n_trials`` independent trials. The deflated value is strictly below the
  observed one once more than one trial is run, and falls as the trial count
  grows.
* :func:`pbo_cscv` estimates the Probability of Backtest Overfitting via
  Combinatorially-Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado &
  Zhu, 2017): the probability that the configuration chosen as best in-sample
  underperforms the median out-of-sample. It returns a value in ``[0, 1]``.

Supporting pieces (:func:`expected_max_sharpe`, :func:`probabilistic_sharpe_ratio`)
are exposed because they are the named building blocks of the Deflated Sharpe
Ratio and are independently useful.

References
----------
Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
Bailey, D., Borwein, J., Lopez de Prado, M. & Zhu, Q. (2017). "The Probability of
Backtest Overfitting." *Journal of Computational Finance*.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_metric",
    "pbo_cscv",
]

# Euler-Mascheroni constant, used in the expected-maximum approximation.
_EULER_MASCHERONI: float = 0.5772156649015329


def expected_max_sharpe(n_trials: int, trials_std: float = 1.0) -> float:
    """Expected maximum of ``n_trials`` skill-free (mean-zero) Sharpe estimates.

    Uses the standard extreme-value approximation for the maximum of ``N`` i.i.d.
    standard normals (Bailey & Lopez de Prado, 2014):

    ``E[max] ~ trials_std * [(1 - g) * Z^{-1}(1 - 1/N) + g * Z^{-1}(1 - 1/(N e))]``

    where ``g`` is the Euler-Mascheroni constant, ``Z^{-1}`` the standard-normal
    quantile and ``e`` Euler's number. For ``N == 1`` the expected maximum is
    ``0`` (a single mean-zero draw has zero expectation).

    Args:
        n_trials: Number of independent trials/configurations, ``>= 1``.
        trials_std: Cross-trial standard deviation of the Sharpe estimates
            (the dispersion of skill-free trial outcomes); defaults to ``1.0``
            (the standardised case). Must be ``>= 0``.

    Returns:
        The expected maximum Sharpe under the multiple-testing null, on the same
        scale as ``trials_std``.

    Raises:
        ValueError: If ``n_trials < 1`` or ``trials_std < 0``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if trials_std < 0.0:
        raise ValueError(f"trials_std must be >= 0, got {trials_std}")
    if n_trials == 1 or trials_std == 0.0:
        return 0.0

    g = _EULER_MASCHERONI
    n = float(n_trials)
    q1 = float(norm.ppf(1.0 - 1.0 / n))
    q2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    return float(trials_std * ((1.0 - g) * q1 + g * q2))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio: ``P(true SR > benchmark_sr)``.

    The PSR (Bailey & Lopez de Prado) is the probability that the true Sharpe
    exceeds ``benchmark_sr`` given an ``observed_sr`` estimated from ``n_obs``
    returns with the supplied higher moments:

    ``PSR = Z[ (observed - benchmark) * sqrt(n - 1)
                / sqrt(1 - skew*observed + (kurtosis - 1)/4 * observed^2) ]``

    Args:
        observed_sr: Estimated (non-annualised) Sharpe ratio.
        benchmark_sr: Threshold Sharpe to test against (e.g. the deflated
            benchmark from :func:`expected_max_sharpe`).
        n_obs: Number of return observations, ``>= 2``.
        skew: Skewness of the returns (``0`` for normal).
        kurtosis: Kurtosis of the returns (``3`` for normal).

    Returns:
        A probability in ``[0, 1]``.

    Raises:
        ValueError: If ``n_obs < 2`` or the variance term is non-positive.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    var_term = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    if var_term <= 0.0:
        raise ValueError(
            f"non-positive PSR variance term ({var_term}); check skew/kurtosis"
        )
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1.0) / math.sqrt(var_term)
    return float(norm.cdf(z))


def deflated_metric(
    observed: float,
    n_trials: int,
    trials_std: float = 1.0,
    n_obs: int | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflate an observed Sharpe-like metric for multiple-testing selection bias.

    The observed statistic is compared against the benchmark a skill-free
    researcher would expect as the best of ``n_trials`` trials,
    ``benchmark = expected_max_sharpe(n_trials, trials_std)``. Two modes:

    * **Haircut (default, ``n_obs is None``):** return ``observed - benchmark`` —
      the metric net of the multiple-testing inflation, on the input scale. This
      is monotonically decreasing in ``n_trials`` and equals ``observed`` for a
      single trial.
    * **Deflated Sharpe Ratio (``n_obs`` given):** return the probability that the
      true Sharpe beats the inflated benchmark,
      ``probabilistic_sharpe_ratio(observed, benchmark, n_obs, skew, kurtosis)`` —
      a value in ``[0, 1]`` that also shrinks toward ``0`` as ``n_trials`` grows.

    In both modes more trials means a harsher penalty, so the returned value is
    strictly below ``observed`` whenever ``n_trials > 1`` (and ``trials_std > 0``).

    Args:
        observed: The observed metric (e.g. the best configuration's Sharpe).
        n_trials: Number of trials/configurations searched, ``>= 1``.
        trials_std: Cross-trial dispersion of skill-free outcomes (default ``1.0``).
        n_obs: If given, return the DSR probability over this many return
            observations; otherwise return the haircut statistic.
        skew: Returns skewness, used only in DSR mode.
        kurtosis: Returns kurtosis, used only in DSR mode.

    Returns:
        The deflated metric: a haircut statistic (default) or a DSR probability
        in ``[0, 1]`` (when ``n_obs`` is supplied).

    Raises:
        ValueError: If ``n_trials < 1`` or (DSR mode) ``n_obs < 2``.
    """
    benchmark = expected_max_sharpe(n_trials, trials_std)

    if n_obs is None:
        deflated = float(observed - benchmark)
        logger.debug(
            "deflated_metric[haircut]: observed=%.4f benchmark=%.4f -> %.4f "
            "(n_trials=%d)",
            observed,
            benchmark,
            deflated,
            n_trials,
        )
        return deflated

    dsr = probabilistic_sharpe_ratio(observed, benchmark, n_obs, skew, kurtosis)
    logger.debug(
        "deflated_metric[DSR]: observed=%.4f benchmark=%.4f n_obs=%d -> %.4f "
        "(n_trials=%d)",
        observed,
        benchmark,
        n_obs,
        dsr,
        n_trials,
    )
    return dsr


def _column_performance(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-configuration in/out-of-sample performance for a row-block.

    The performance statistic is the Sharpe-like mean-over-std of each column
    (configuration) across the block's rows. Columns whose dispersion is zero
    fall back to the raw mean so constant series remain comparable rather than
    collapsing to ``NaN``.

    Args:
        block: ``(rows, n_configs)`` slice of the performance matrix.

    Returns:
        A length-``n_configs`` array of per-configuration scores.
    """
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1) if block.shape[0] > 1 else np.zeros(block.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(std > 0.0, mean / std, mean)
    return np.asarray(score, dtype=np.float64)


def pbo_cscv(
    perf_matrix: NDArray[np.float64],
    n_splits: int = 10,
) -> float:
    """Probability of Backtest Overfitting via CSCV.

    Implements Combinatorially-Symmetric Cross-Validation (Bailey et al., 2017).
    The ``(T, N)`` performance matrix (``T`` time observations x ``N``
    configurations) is split row-wise into ``n_splits`` equal contiguous blocks.
    For every way to choose ``n_splits / 2`` blocks as the in-sample (IS) set
    (the rest out-of-sample, OoS):

    1. pick the configuration with the best IS performance;
    2. find that configuration's OoS performance rank among all ``N``;
    3. map the relative rank ``w in (0, 1)`` to the logit ``lambda = ln(w/(1-w))``.

    PBO is the fraction of combinations whose ``lambda <= 0`` — i.e. the IS-best
    configuration lands in the bottom half OoS, the signature of overfitting.

    Args:
        perf_matrix: ``(T, N)`` array of per-period, per-configuration performance
            (e.g. returns). Needs ``N >= 2`` configurations and enough rows to form
            ``n_splits`` blocks of at least one row each.
        n_splits: Number of CSCV blocks ``S`` (must be even and ``>= 2``); the
            default ``10`` gives ``C(10, 5) = 252`` combinations.

    Returns:
        The probability of backtest overfitting, a float in ``[0, 1]``.

    Raises:
        ValueError: If ``perf_matrix`` is not 2-D with ``N >= 2``, if ``n_splits``
            is odd / ``< 2``, or if there are too few rows to split.
    """
    matrix = np.ascontiguousarray(perf_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"perf_matrix must be 2-D, got shape {matrix.shape}")
    n_obs, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError(f"need at least 2 configurations, got {n_configs}")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    if n_obs < n_splits:
        raise ValueError(
            f"need at least n_splits={n_splits} rows, got {n_obs}; "
            "reduce n_splits or supply more observations"
        )

    # Equal contiguous blocks; any remainder rows are dropped from the tail so
    # every block has the same length (CSCV assumes balanced sub-matrices).
    block_len = n_obs // n_splits
    usable = block_len * n_splits
    blocks = [
        matrix[b * block_len : (b + 1) * block_len, :] for b in range(n_splits)
    ]
    if usable != n_obs:
        logger.debug(
            "pbo_cscv: dropping %d tail rows to balance %d blocks of %d",
            n_obs - usable,
            n_splits,
            block_len,
        )

    all_blocks = set(range(n_splits))
    half = n_splits // 2
    logits: list[float] = []
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = sorted(all_blocks.difference(is_blocks))
        is_mat = np.concatenate([blocks[b] for b in is_blocks], axis=0)
        oos_mat = np.concatenate([blocks[b] for b in oos_blocks], axis=0)

        is_perf = _column_performance(is_mat)
        oos_perf = _column_performance(oos_mat)

        # Best configuration in-sample (lowest index breaks ties).
        best = int(np.argmax(is_perf))

        # Relative OoS rank of the IS-best config in (0, 1). Average ranks over
        # ties; "rank" counts configs with strictly lower OoS performance.
        oos_best = oos_perf[best]
        less = int(np.sum(oos_perf < oos_best))
        equal = int(np.sum(oos_perf == oos_best))
        # Average rank position (1-based) among equals, then map to (0, 1).
        avg_rank = less + (equal + 1) / 2.0
        omega = avg_rank / (n_configs + 1.0)
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)
        logits.append(math.log(omega / (1.0 - omega)))

    logit_arr = np.asarray(logits, dtype=np.float64)
    pbo = float(np.mean(logit_arr <= 0.0))
    logger.info(
        "pbo_cscv: PBO=%.4f over %d combinations (S=%d, N=%d)",
        pbo,
        logit_arr.shape[0],
        n_splits,
        n_configs,
    )
    return pbo
