"""Purged + embargoed walk-forward cross-validation (López de Prado).

This module implements the one CV scheme that is sound for serially correlated,
overlapping-label financial series: an *expanding* walk-forward split with
**purging** and **embargoing** (López de Prado, *Advances in Financial Machine
Learning*, ch. 7).

Why a plain ``KFold`` / contiguous split is wrong here
------------------------------------------------------
Each training label spans a forward window ``[t, t + label_horizon]`` (e.g. "did
price turn within the next ``label_horizon`` bars?"). If a training bar sits
within ``label_horizon`` of the test block, its label *window overlaps the test
period* — the model is trained on information it is then scored on. That is
**leakage**. Two corrections remove it:

* **Purge** — drop every training bar whose label window ``[t, t + h]`` overlaps
  the test block's timeline.
* **Embargo** — additionally drop ``embargo`` bars immediately *after* each test
  block from training, because returns are serially correlated and a bar right
  after the block is statistically entangled with it.

The forbidden span around a test block ``[block_start, block_end]`` is therefore

    [block_start - label_horizon, block_end + embargo]

measured in **bar positions along the relevant timeline**. For a single,
unit-spaced timeline a bar offset equals a timestamp offset, so the band is also
exact in timestamp terms.

Pooled (multi-asset) timelines
-------------------------------
When ``pooled_groups`` is given (a per-row asset id), the *split* still happens
on the single pooled time axis (so a fold's test block spans all assets at once),
but **purge and embargo are computed within each asset's own bar clock**. A
five-bar embargo means five bars *of that asset*, not five rows of the
interleaved pool, so one asset's history cannot leak across another's test bars.

Determinism
-----------
The routine performs a stable sort on ``timestamps`` and uses
:func:`numpy.array_split`; given identical inputs it returns byte-identical
folds. Returned indices always refer to the *original* row order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "purged_walk_forward",
    "LeakageError",
]

# A single fold: (train_idx, test_idx), both 1-D int64 arrays of original rows.
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]


class LeakageError(RuntimeError):
    """Raised if a constructed fold violates the purge/embargo contract.

    This is a defensive self-check: it should never fire for correct inputs. It
    exists so a future refactor that silently reintroduces leakage fails loudly
    rather than producing optimistically biased CV scores.
    """


# --------------------------------------------------------------------------- #
# Input validation.                                                            #
# --------------------------------------------------------------------------- #


def _validate(
    timestamps: NDArray[np.int64],
    label_horizon: int,
    embargo: int,
    n_folds: int,
    pooled_groups: NDArray[np.int64] | None,
) -> NDArray[np.int64]:
    """Validate arguments and return ``timestamps`` as a 1-D ``int64`` array.

    Args:
        timestamps: Per-row event times (any integer epoch; may be negative).
        label_horizon: Forward label span in bars; must be ``>= 0``.
        embargo: Post-block embargo in bars; must be ``>= 0``.
        n_folds: Number of walk-forward test blocks; must be ``>= 2`` and at most
            the number of samples.
        pooled_groups: Optional per-row asset id, same length as ``timestamps``.

    Returns:
        ``timestamps`` coerced to a contiguous 1-D ``int64`` array.

    Raises:
        ValueError: If any argument is out of its allowed domain or the array
            shapes are inconsistent.
    """
    ts = np.ascontiguousarray(timestamps)
    if ts.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape {ts.shape}")
    n = ts.shape[0]
    if n == 0:
        raise ValueError("timestamps is empty")
    if label_horizon < 0:
        raise ValueError(f"label_horizon must be >= 0, got {label_horizon}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_folds > n:
        raise ValueError(f"n_folds={n_folds} exceeds number of samples {n}")
    if pooled_groups is not None:
        groups = np.ascontiguousarray(pooled_groups)
        if groups.ndim != 1:
            raise ValueError(f"pooled_groups must be 1-D, got shape {groups.shape}")
        if groups.shape[0] != n:
            raise ValueError(
                f"pooled_groups length {groups.shape[0]} != timestamps length {n}"
            )
    return ts.astype(np.int64, copy=False)


# --------------------------------------------------------------------------- #
# Forbidden-band construction.                                                 #
# --------------------------------------------------------------------------- #


def _forbidden_mask_single(
    n_sorted: int,
    block_lo: int,
    block_hi: int,
    label_horizon: int,
    embargo: int,
) -> NDArray[np.bool_]:
    """Mark sorted positions inside ``[lo - h, hi + embargo]`` as forbidden.

    Operates in the single pooled timeline's *sorted-position* space, where a
    one-position step is one bar. The test block itself is also marked (it is
    removed from train regardless), so the returned mask is exactly the set of
    sorted positions a training bar may **not** occupy for this fold.

    Args:
        n_sorted: Number of samples (length of the sorted axis).
        block_lo: First sorted position of the test block (inclusive).
        block_hi: Last sorted position of the test block (inclusive).
        label_horizon: Forward label span in bars.
        embargo: Post-block embargo in bars.

    Returns:
        Boolean mask over sorted positions; ``True`` == forbidden for training.
    """
    lo = max(0, block_lo - label_horizon)
    hi = min(n_sorted - 1, block_hi + embargo)
    mask = np.zeros(n_sorted, dtype=np.bool_)
    mask[lo : hi + 1] = True
    return mask


def _forbidden_mask_pooled(
    groups_sorted: NDArray[np.int64],
    is_test: NDArray[np.bool_],
    label_horizon: int,
    embargo: int,
) -> NDArray[np.bool_]:
    """Build the forbidden mask per asset, in each asset's own bar clock.

    For every asset the test block maps to a contiguous range of that asset's
    own sorted positions (the global block is contiguous, so its restriction to
    a single asset is too). The purge/embargo band ``[lo - h, hi + embargo]`` is
    then taken in *that asset's* position space, so the exclusion zone is sized
    in bars of the asset rather than rows of the interleaved pool.

    Args:
        groups_sorted: Asset id per sorted position (aligned to the sorted axis).
        is_test: Boolean mask over sorted positions selecting this fold's block.
        label_horizon: Forward label span in bars (per asset).
        embargo: Post-block embargo in bars (per asset).

    Returns:
        Boolean mask over sorted positions; ``True`` == forbidden for training.
    """
    n_sorted = groups_sorted.shape[0]
    forbidden = np.zeros(n_sorted, dtype=np.bool_)
    # The test block is always removed from train; start from it and grow the
    # purge/embargo halo within each asset.
    forbidden |= is_test

    for asset in np.unique(groups_sorted):
        asset_positions = np.flatnonzero(groups_sorted == asset)
        # This asset's local test positions (0-based within its own timeline).
        local_is_test = is_test[asset_positions]
        if not local_is_test.any():
            continue
        local_test_pos = np.flatnonzero(local_is_test)
        lo = int(local_test_pos.min()) - label_horizon
        hi = int(local_test_pos.max()) + embargo
        lo = max(0, lo)
        hi = min(asset_positions.shape[0] - 1, hi)
        # Map the local band back to global sorted positions and forbid them.
        banned_global = asset_positions[lo : hi + 1]
        forbidden[banned_global] = True
    return forbidden


# --------------------------------------------------------------------------- #
# Self-check.                                                                  #
# --------------------------------------------------------------------------- #


def _assert_no_leak(
    train_sorted: NDArray[np.int64],
    forbidden: NDArray[np.bool_],
    fold_id: int,
) -> None:
    """Raise :class:`LeakageError` if any train position is forbidden.

    Args:
        train_sorted: Sorted positions assigned to training in this fold.
        forbidden: Forbidden-position mask for this fold.
        fold_id: Index of the fold (for the error message).

    Raises:
        LeakageError: If the training set intersects the forbidden band.
    """
    if train_sorted.size and forbidden[train_sorted].any():
        leaked = int(forbidden[train_sorted].sum())
        raise LeakageError(
            f"fold {fold_id}: {leaked} training bars fall inside the "
            "purge/embargo band — leakage was not removed"
        )


# --------------------------------------------------------------------------- #
# Public API.                                                                  #
# --------------------------------------------------------------------------- #


def purged_walk_forward(
    timestamps: NDArray[np.int64],
    label_horizon: int,
    embargo: int,
    n_folds: int,
    pooled_groups: NDArray[np.int64] | None = None,
) -> list[Fold]:
    """Build purged + embargoed expanding walk-forward folds.

    The samples are ordered by ``timestamps`` and partitioned into ``n_folds``
    contiguous test blocks that advance through time. For each fold the training
    set is every other sample **minus** the purge/embargo band: any sample whose
    label window ``[t, t + label_horizon]`` overlaps the test block, plus the
    ``embargo`` bars immediately following the block.

    With ``pooled_groups`` the partition is still taken on the pooled time axis,
    but purging and embargoing are applied *within each asset's own bar clock* so
    assets cannot leak across one another.

    Args:
        timestamps: Per-row event times (integer epochs; may be negative and need
            not be unique or pre-sorted). Returned indices refer to these rows in
            their original order.
        label_horizon: Forward label span in bars; ``>= 0``.
        embargo: Number of bars after each test block to exclude from training;
            ``>= 0``.
        n_folds: Number of walk-forward test blocks; ``2 <= n_folds <= len``.
        pooled_groups: Optional per-row asset id (same length as ``timestamps``).
            When given, purge/embargo are computed per asset.

    Returns:
        A list of ``n_folds`` ``(train_idx, test_idx)`` tuples. Both arrays are
        sorted 1-D ``int64`` indices into the original rows, are disjoint, and
        the test blocks are time-ordered across folds.

    Raises:
        ValueError: If any argument is out of its allowed domain (see
            :func:`_validate`).
        LeakageError: If the internal self-check detects residual leakage (should
            never happen for valid inputs).
    """
    ts = _validate(timestamps, label_horizon, embargo, n_folds, pooled_groups)
    n = ts.shape[0]

    # Stable ascending order by timestamp; ``order[k]`` is the original row at
    # sorted position ``k``. A stable sort makes ties (duplicate timestamps)
    # resolve by original order, which keeps the result deterministic.
    order = np.argsort(ts, kind="stable")

    groups_sorted: NDArray[np.int64] | None = None
    if pooled_groups is not None:
        groups_sorted = np.ascontiguousarray(pooled_groups)[order].astype(
            np.int64, copy=False
        )

    # Contiguous, near-equal test blocks over the sorted axis.
    test_blocks = np.array_split(np.arange(n), n_folds)

    folds: list[Fold] = []
    for fold_id, block_positions in enumerate(test_blocks):
        block_positions = np.ascontiguousarray(block_positions)
        if block_positions.size == 0:
            # Possible only when n_folds == n is not requested; guarded by
            # validation, but keep the loop total honest.
            continue
        block_lo = int(block_positions[0])
        block_hi = int(block_positions[-1])

        is_test = np.zeros(n, dtype=np.bool_)
        is_test[block_positions] = True

        if groups_sorted is None:
            forbidden = _forbidden_mask_single(
                n, block_lo, block_hi, label_horizon, embargo
            )
        else:
            forbidden = _forbidden_mask_pooled(
                groups_sorted, is_test, label_horizon, embargo
            )

        # Train = everything that is neither test nor inside the forbidden band.
        train_sorted = np.flatnonzero(~forbidden & ~is_test)
        _assert_no_leak(train_sorted, forbidden, fold_id)

        # Map sorted positions back to original row indices; sort so the returned
        # arrays are ascending in original-index order (deterministic, tidy).
        train_idx = np.sort(order[train_sorted]).astype(np.int64, copy=False)
        test_idx = np.sort(order[block_positions]).astype(np.int64, copy=False)
        folds.append((train_idx, test_idx))

        logger.debug(
            "fold %d: test=%d train=%d (block sorted-pos [%d, %d])",
            fold_id,
            test_idx.size,
            train_idx.size,
            block_lo,
            block_hi,
        )

    logger.info(
        "purged_walk_forward: %d folds over %d samples (h=%d, embargo=%d, pooled=%s)",
        len(folds),
        n,
        label_horizon,
        embargo,
        pooled_groups is not None,
    )
    return folds
