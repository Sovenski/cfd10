"""Tests for ``cfd10.cv_module.splits`` — purged + embargoed walk-forward CV.

THE LEAKAGE TEST IS THE POINT. The whole reason this harness exists is that a
naive contiguous train/test split lets the label of a training bar peek into the
test block (a label spanning ``[t, t + label_horizon]`` overlaps the test
window) and lets a training bar sit immediately *after* the test block where it
is statistically entangled with it (serial correlation). López de Prado's
*purging* removes the former and *embargoing* removes the latter.

The forbidden timestamp band around a test block ``[test_start, test_end]`` is

    [test_start - label_horizon, test_end + embargo]

(in timestamp terms; for unit-spaced single-asset timelines a bar offset equals
a timestamp offset). For every fold we assert that **no** training index lands
inside that band. We additionally:

* build a case where a naive contiguous split *would* admit a leaking neighbour
  and assert :func:`purged_walk_forward` drops it;
* pool two interleaved assets and assert neither asset's bar straddles the
  embargo/purge zone of the *other* asset's test bars within a fold;
* assert byte-for-byte determinism across repeated calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfd10.cv_module import purged_walk_forward
from cfd10.cv_module.splits import LeakageError


# --------------------------------------------------------------------------- #
# Helpers.                                                                     #
# --------------------------------------------------------------------------- #


def _assert_partition(
    folds: list[tuple[np.ndarray, np.ndarray]],
    n: int,
    timestamps: np.ndarray | None = None,
) -> None:
    """Train/test indices are disjoint, in-range, and test blocks are ordered.

    The walk-forward ordering invariant is "successive test blocks advance in
    *time*". Returned indices refer to original rows, so when ``timestamps`` is a
    shuffling of the time axis we must check ordering in timestamp space; for the
    unit-spaced ``arange`` fixtures row index and timestamp coincide.
    """
    prev_test_time_max = -np.inf
    for train_idx, test_idx in folds:
        assert train_idx.dtype.kind == "i"
        assert test_idx.dtype.kind == "i"
        # In range.
        assert train_idx.min(initial=0) >= 0
        assert test_idx.min(initial=0) >= 0
        assert test_idx.max(initial=0) < n
        if train_idx.size:
            assert train_idx.max() < n
        # Disjoint.
        assert np.intersect1d(train_idx, test_idx).size == 0
        # Returned arrays are sorted, unique (by original row index).
        assert np.all(np.diff(train_idx) > 0) if train_idx.size > 1 else True
        assert np.all(np.diff(test_idx) > 0) if test_idx.size > 1 else True
        # Walk-forward: test blocks advance in time (checked in timestamp space).
        if timestamps is None:
            block_time_min = float(test_idx.min())
            block_time_max = float(test_idx.max())
        else:
            block_time_min = float(timestamps[test_idx].min())
            block_time_max = float(timestamps[test_idx].max())
        assert block_time_min > prev_test_time_max
        prev_test_time_max = block_time_max


def _assert_no_leak_single_timeline(
    timestamps: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    label_horizon: int,
    embargo: int,
) -> None:
    """No train timestamp lies in ``[t_start - h, t_end + embargo]`` of its block.

    Works on a *single* monotonically increasing, unit-spaced timeline where a
    one-bar step equals a one-unit timestamp step, so the bar-defined purge /
    embargo translate exactly to a timestamp band.
    """
    for train_idx, test_idx in folds:
        if test_idx.size == 0:
            continue
        test_start = float(timestamps[test_idx].min())
        test_end = float(timestamps[test_idx].max())
        lo = test_start - label_horizon
        hi = test_end + embargo
        train_ts = timestamps[train_idx].astype(float)
        inside = (train_ts >= lo) & (train_ts <= hi)
        assert not inside.any(), (
            f"leakage: {int(inside.sum())} train bars fall in forbidden band "
            f"[{lo}, {hi}] around test block [{test_start}, {test_end}]"
        )


# --------------------------------------------------------------------------- #
# Core leakage contract (single asset).                                        #
# --------------------------------------------------------------------------- #


def test_no_leak_in_forbidden_band_unit_timeline() -> None:
    """For every fold, no train bar lies in the purge/embargo band."""
    n = 200
    timestamps = np.arange(n, dtype=np.int64)
    label_horizon = 5
    embargo = 3
    folds = purged_walk_forward(
        timestamps, label_horizon=label_horizon, embargo=embargo, n_folds=5
    )
    assert len(folds) == 5
    _assert_partition(folds, n)
    _assert_no_leak_single_timeline(timestamps, folds, label_horizon, embargo)


def test_no_leak_with_negative_epochs() -> None:
    """Negative, unit-spaced epochs (early SPX bars) still respect the band."""
    n = 120
    timestamps = np.arange(-50, -50 + n, dtype=np.int64)
    label_horizon = 7
    embargo = 4
    folds = purged_walk_forward(
        timestamps, label_horizon=label_horizon, embargo=embargo, n_folds=4
    )
    _assert_partition(folds, n)
    _assert_no_leak_single_timeline(timestamps, folds, label_horizon, embargo)


def test_naive_contiguous_split_would_leak_but_purged_excludes_it() -> None:
    """A neighbour a naive split keeps is the exact bar purging must drop.

    The naive contiguous split puts everything before the test block into train.
    The training bars in the last ``label_horizon`` positions before the test
    block, and the first ``embargo`` after it, are precisely the leaking
    neighbours. We assert those specific indices are present in the naive split
    yet absent from the purged split.
    """
    n = 100
    timestamps = np.arange(n, dtype=np.int64)
    label_horizon = 6
    embargo = 5
    n_folds = 4
    folds = purged_walk_forward(
        timestamps, label_horizon=label_horizon, embargo=embargo, n_folds=n_folds
    )

    # Inspect an interior fold (not the first) so a post-block embargo region of
    # train bars can exist if the design ever keeps later data.
    for train_idx, test_idx in folds:
        train_set = set(train_idx.tolist())
        test_start = int(test_idx.min())
        test_end = int(test_idx.max())

        # The naive contiguous "train = all bars before test_start" split would
        # include these immediately-preceding bars; each has a label window
        # [t, t + label_horizon] that reaches into the test block -> must purge.
        leaking_pre = range(max(0, test_start - label_horizon), test_start)
        for t in leaking_pre:
            assert t not in train_set, (
                f"purge failed: pre-block bar {t} (label window reaches test "
                f"block starting {test_start}) leaked into train"
            )

        # Any train bar in the embargo region just after the block must be gone.
        embargo_zone = range(test_end + 1, test_end + 1 + embargo)
        for t in embargo_zone:
            assert t not in train_set, (
                f"embargo failed: post-block bar {t} within {embargo} bars of "
                f"test end {test_end} leaked into train"
            )

    # And a sanity check that the naive split really *would* have kept a leaker:
    # for the second fold, the bar at (test_start - 1) is a valid earlier index.
    _, second_test = folds[1]
    leaker = int(second_test.min()) - 1
    assert 0 <= leaker < n
    naive_train_before = leaker in set(range(int(second_test.min())))
    assert naive_train_before, "fixture sanity: naive split should contain the leaker"


def test_purge_window_is_label_horizon_wide() -> None:
    """The bar exactly ``label_horizon`` before the block start is the boundary.

    A train bar at ``test_start - label_horizon`` has a label window ending at
    ``test_start`` (touching the block) and must be purged; a bar one earlier is
    safe and (if otherwise eligible) may remain.
    """
    n = 150
    timestamps = np.arange(n, dtype=np.int64)
    label_horizon = 10
    embargo = 0
    folds = purged_walk_forward(
        timestamps, label_horizon=label_horizon, embargo=embargo, n_folds=3
    )
    for train_idx, test_idx in folds:
        train_set = set(train_idx.tolist())
        test_start = int(test_idx.min())
        boundary = test_start - label_horizon
        if boundary >= 0:
            assert boundary not in train_set, (
                f"boundary bar {boundary} (label touches block start "
                f"{test_start}) must be purged"
            )
        safe = test_start - label_horizon - 1
        if safe >= 0:
            # The bar one step earlier does not touch the block; it should be a
            # legal training bar (it precedes the block and clears the purge).
            assert safe in train_set, (
                f"bar {safe} is outside the purge window and should remain in "
                "train"
            )


# --------------------------------------------------------------------------- #
# Pooled multi-asset timelines.                                                #
# --------------------------------------------------------------------------- #


def test_pooled_groups_no_cross_asset_straddle() -> None:
    """Two interleaved assets: no asset's bar straddles the other's band.

    Rows alternate A, B, A, B, ... but each asset has its own internal,
    unit-spaced timeline. Purge/embargo are defined *within* each asset, so when
    we look at a fold we must verify, per asset, that none of that asset's train
    bars fall in the forbidden band around that asset's own test bars.
    """
    per_asset = 100
    # Asset A on even rows, asset B on odd rows; interleaved in the pooled order.
    groups = np.empty(2 * per_asset, dtype=np.int64)
    groups[0::2] = 0
    groups[1::2] = 1
    # Per-asset timelines are unit-spaced; the pooled timestamp is the row order
    # so the global walk-forward sees a single increasing axis.
    timestamps = np.arange(2 * per_asset, dtype=np.int64)

    label_horizon = 4
    embargo = 3
    folds = purged_walk_forward(
        timestamps,
        label_horizon=label_horizon,
        embargo=embargo,
        n_folds=5,
        pooled_groups=groups,
    )
    _assert_partition(folds, 2 * per_asset)

    for train_idx, test_idx in folds:
        for asset in (0, 1):
            # Restrict to this asset's own bars and its own (unit-spaced) clock.
            asset_mask_all = groups == asset
            # Map global indices -> position within this asset's sorted timeline.
            asset_global = np.flatnonzero(asset_mask_all)
            pos_of_global = {int(g): p for p, g in enumerate(asset_global)}

            test_asset = [i for i in test_idx.tolist() if groups[i] == asset]
            train_asset = [i for i in train_idx.tolist() if groups[i] == asset]
            if not test_asset:
                continue
            test_pos = np.array([pos_of_global[i] for i in test_asset])
            lo = int(test_pos.min()) - label_horizon
            hi = int(test_pos.max()) + embargo
            for gi in train_asset:
                p = pos_of_global[gi]
                assert not (lo <= p <= hi), (
                    f"asset {asset}: train bar (global {gi}, asset-pos {p}) "
                    f"straddles forbidden asset-positions [{lo}, {hi}]"
                )


def test_pooled_groups_partition_covers_each_asset() -> None:
    """Pooled folds keep each asset's train/test indices within that asset."""
    per_asset = 60
    groups = np.empty(2 * per_asset, dtype=np.int64)
    groups[0::2] = 7
    groups[1::2] = 9
    timestamps = np.arange(2 * per_asset, dtype=np.int64)
    folds = purged_walk_forward(
        timestamps,
        label_horizon=3,
        embargo=2,
        n_folds=4,
        pooled_groups=groups,
    )
    _assert_partition(folds, 2 * per_asset)
    # Every test block should contain bars from both assets (interleaved input),
    # confirming the split happens on the pooled timeline, not per asset.
    for _, test_idx in folds:
        seen = set(groups[test_idx].tolist())
        assert seen == {7, 9}, f"expected both assets in test block, saw {seen}"


# --------------------------------------------------------------------------- #
# Determinism & validation.                                                    #
# --------------------------------------------------------------------------- #


def test_determinism_identical_folds() -> None:
    """Identical inputs yield byte-identical folds across repeated calls."""
    timestamps = np.arange(300, dtype=np.int64)
    kwargs = dict(label_horizon=5, embargo=4, n_folds=6)
    a = purged_walk_forward(timestamps, **kwargs)
    b = purged_walk_forward(timestamps, **kwargs)
    assert len(a) == len(b)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_determinism_pooled() -> None:
    """Determinism also holds for the pooled-groups path."""
    n = 240
    groups = np.tile(np.array([0, 1, 2], dtype=np.int64), n // 3)
    timestamps = np.arange(n, dtype=np.int64)
    kwargs = dict(label_horizon=4, embargo=3, n_folds=5, pooled_groups=groups)
    a = purged_walk_forward(timestamps, **kwargs)
    b = purged_walk_forward(timestamps, **kwargs)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_unsorted_timestamps_accepted_and_sorted_internally() -> None:
    """Shuffled input still produces a valid, leak-free walk-forward.

    The harness must order by timestamp internally; the returned indices refer
    to the *original* rows. We verify partition + leak-freedom after mapping
    each returned index back through the (shuffled) timestamp array.
    """
    n = 160
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    timestamps = perm.astype(np.int64)  # a shuffling of 0..n-1
    label_horizon = 5
    embargo = 2
    folds = purged_walk_forward(
        timestamps, label_horizon=label_horizon, embargo=embargo, n_folds=4
    )
    _assert_partition(folds, n, timestamps=timestamps)
    # The forbidden-band check is defined in timestamp space, so it is invariant
    # to the row permutation.
    _assert_no_leak_single_timeline(timestamps, folds, label_horizon, embargo)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(label_horizon=-1, embargo=0, n_folds=3),
        dict(label_horizon=0, embargo=-1, n_folds=3),
        dict(label_horizon=0, embargo=0, n_folds=1),
        dict(label_horizon=0, embargo=0, n_folds=0),
    ],
)
def test_invalid_arguments_raise(kwargs: dict) -> None:
    """Out-of-domain horizon / embargo / fold counts raise ``ValueError``."""
    timestamps = np.arange(50, dtype=np.int64)
    with pytest.raises(ValueError):
        purged_walk_forward(timestamps, **kwargs)


def test_too_many_folds_raises() -> None:
    """Requesting more folds than samples raises ``ValueError``."""
    timestamps = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError):
        purged_walk_forward(timestamps, label_horizon=0, embargo=0, n_folds=10)


def test_mismatched_groups_length_raises() -> None:
    """``pooled_groups`` of the wrong length raises ``ValueError``."""
    timestamps = np.arange(20, dtype=np.int64)
    groups = np.zeros(19, dtype=np.int64)
    with pytest.raises(ValueError):
        purged_walk_forward(
            timestamps, label_horizon=0, embargo=0, n_folds=3, pooled_groups=groups
        )


def test_leakage_error_is_exported() -> None:
    """The defensive ``LeakageError`` type is importable (self-check guard)."""
    assert issubclass(LeakageError, Exception)
