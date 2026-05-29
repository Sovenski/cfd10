"""Windowed sequence dataset for the TCN teacher (group-boundary safe).

The GBDT baseline scores a single bar's feature vector; the TCN instead reads a
*window* of ``W`` consecutive bars. On the **pooled** corpus that raises one
correctness hazard: a naive sliding window straddling the boundary between two
stacked assets would feed bars from asset A and asset B into one sample — a leak
across instruments that the per-asset feature/label construction worked hard to
avoid. :class:`WindowDataset` forbids it: a window ending at row ``i`` is emitted
**only if all ``W`` of its bars share one group id**.

What each item is
-----------------
For a valid window ending at pooled row ``i`` the dataset yields
``(window, target, weight)`` where:

* ``window`` is a ``(W, F)`` float tensor of the standardized feature bars
  ``[i - W + 1 .. i]`` (oldest first; the model reads the last row as "now");
* ``target`` is the ``(2,)`` per-side binary label **at the window's last bar**
  (``[top, bottom]``, ``1`` == oracle turn);
* ``weight`` is the ``(2,)`` per-side sample weight under the project convention
  ``w = 1 + oracle_score`` on turns else ``1`` (the raw ``*_weight`` column is
  ``0`` on non-turns and must never be used as the weight directly).

Standardization (no leakage)
----------------------------
Features are standardized with a **train-split** mean / std passed in by the
caller (:func:`compute_feature_stats` over the train rows), never statistics of
the whole corpus, so test windows are scaled by train-only moments. After
standardization any residual ``NaN`` is imputed to ``0`` (the post-standardization
mean), so a feature that was missing contributes nothing rather than poisoning the
convolution.

Public API
----------
:class:`FeatureStats` (frozen) + :func:`compute_feature_stats`; :func:`make_labels`
(tiers + per-side weight under the convention); :class:`WindowDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "FeatureStats",
    "compute_feature_stats",
    "make_labels",
    "WindowDataset",
    "POSITIVE_TIERS",
]

# Oracle tiers counted as a positive turn (y = 1).
POSITIVE_TIERS: frozenset[str] = frozenset({"strong", "regular"})

# Side -> (tier column, weight column) in the oracle label frame. Index 0 is the
# top (high) side, index 1 the bottom (low) side — the model's output order.
_SIDE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("top_tier", "top_weight"),
    ("bottom_tier", "bottom_weight"),
)

_EPS: float = 1e-8


# --------------------------------------------------------------------------- #
# Train-split feature standardization (leakage-safe).                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature standardization moments (computed on the train split only).

    Attributes:
        mean: Per-feature mean, shape ``(F,)`` (NaNs ignored in the estimate).
        std: Per-feature standard deviation, shape ``(F,)``, floored away from 0
            so division is safe for (near-)constant features.
    """

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the moment shapes match."""
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"FeatureStats: mean shape {self.mean.shape} != std shape "
                f"{self.std.shape}"
            )

    @property
    def n_features(self) -> int:
        """Number of features the stats cover."""
        return int(self.mean.shape[0])


def compute_feature_stats(
    X: pd.DataFrame, train_idx: NDArray[np.int64]
) -> FeatureStats:
    """Compute per-feature mean / std over the **train rows only**.

    NaNs are ignored when estimating the moments (``nanmean`` / ``nanstd``); the
    std is floored at :data:`_EPS` so a (near-)constant column does not blow up the
    standardization. Using only ``train_idx`` keeps test statistics out of the
    scaling — the same no-leakage rule the baseline follows for thresholding.

    Args:
        X: The pooled feature matrix.
        train_idx: Row indices of the training split (into ``X``).

    Returns:
        The :class:`FeatureStats` to pass to :class:`WindowDataset`.

    Raises:
        ValueError: If ``train_idx`` is empty.
    """
    if train_idx.size == 0:
        raise ValueError("compute_feature_stats: train_idx is empty")
    values = X.to_numpy(dtype=np.float64)[train_idx]
    # All-NaN columns would make nanmean/nanstd emit a RuntimeWarning and NaNs;
    # they are imputed to 0 downstream, so treat their moments as 0 / 1 here.
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std = np.where(std < _EPS, 1.0, std)
    return FeatureStats(
        mean=np.ascontiguousarray(mean, dtype=np.float64),
        std=np.ascontiguousarray(std, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Label / weight construction (the sample-weight convention lives here).       #
# --------------------------------------------------------------------------- #


def make_labels(labels: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build per-side binary targets and convention-correct sample weights.

    For each side (top, bottom): ``y = 1`` iff the oracle tier is in
    :data:`POSITIVE_TIERS` (``strong`` / ``regular``). The sample weight follows
    the project convention

        ``w = where(y == 1, 1.0 + oracle_score, 1.0)``

    where ``oracle_score`` is the side's ``*_weight`` column (the turn score, ``0``
    on non-turns). The raw column is **never** used as the weight directly — that
    would zero the negative class.

    Args:
        labels: The oracle label frame (must contain the tier / weight columns
            for both sides).

    Returns:
        ``(y, w)`` each shaped ``(N, 2)`` (column 0 = top, column 1 = bottom),
        ``float64``.

    Raises:
        KeyError: If a required tier / weight column is missing.
    """
    n = len(labels)
    y = np.zeros((n, 2), dtype=np.float64)
    w = np.ones((n, 2), dtype=np.float64)
    for side, (tier_col, weight_col) in enumerate(_SIDE_COLUMNS):
        if tier_col not in labels.columns or weight_col not in labels.columns:
            raise KeyError(
                f"make_labels: label frame missing {tier_col!r}/{weight_col!r}"
            )
        tiers = labels[tier_col].to_numpy()
        y_side = np.isin(tiers, list(POSITIVE_TIERS)).astype(np.float64)
        oracle_score = labels[weight_col].to_numpy(dtype=np.float64)
        # Convention: floor negatives at 1, let the score add emphasis on turns.
        w_side = np.where(y_side == 1.0, 1.0 + oracle_score, 1.0)
        y[:, side] = y_side
        w[:, side] = w_side
    return y, w


# --------------------------------------------------------------------------- #
# Valid-window enumeration (the group-boundary guarantee).                     #
# --------------------------------------------------------------------------- #


def _valid_window_ends(
    groups: NDArray[np.int64], window: int
) -> NDArray[np.int64]:
    """Return the end-row of every window whose ``window`` bars share one group.

    A window ending at row ``i`` spans ``[i - window + 1, i]``. It is valid iff
    every bar in that span has the same group id. Because the pooled corpus stacks
    each asset's rows contiguously, a span is single-group exactly when the bar
    ``window - 1`` rows back has the same group id as the end bar **and** no group
    change occurs inside the span. We test this directly (vectorised) rather than
    assuming contiguity, so the guarantee holds even for an arbitrarily ordered
    ``groups`` array.

    Args:
        groups: Per-row integer group id, shape ``(N,)``.
        window: Window length ``W`` (``>= 1``).

    Returns:
        Sorted end-row indices ``i`` (``window - 1 <= i < N``) of every valid
        window, as an ``int64`` array (possibly empty).
    """
    n = groups.shape[0]
    if window < 1:
        raise ValueError(f"_valid_window_ends: window must be >= 1, got {window}")
    if n < window:
        return np.empty(0, dtype=np.int64)

    # diff[k] == 0 iff rows k and k+1 share a group. A window [s, i] is single
    # group iff there is no boundary among diff[s .. i-1], i.e. the count of
    # boundaries in that range is 0. Use a prefix sum of boundary indicators for
    # an O(N) test over every candidate end.
    boundary = (groups[1:] != groups[:-1]).astype(np.int64)  # length N-1
    prefix = np.concatenate([[0], np.cumsum(boundary)])  # length N, prefix[k]=sum(boundary[:k])

    ends = np.arange(window - 1, n, dtype=np.int64)
    starts = ends - (window - 1)
    # Boundaries strictly inside the span are boundary[starts .. ends-1] ->
    # prefix[ends] - prefix[starts]. Zero means the whole span is one group.
    inside = prefix[ends] - prefix[starts]
    valid = ends[inside == 0]
    return np.ascontiguousarray(valid, dtype=np.int64)


# --------------------------------------------------------------------------- #
# The dataset.                                                                 #
# --------------------------------------------------------------------------- #


class WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Sliding-window view of the pooled corpus that never crosses an asset.

    Each item is a ``(window, target, weight)`` triple for one valid window
    (see the module docstring). Features are standardized with the supplied
    train-split :class:`FeatureStats` and NaN-imputed to ``0`` *after*
    standardization. The set of windows can be restricted to a CV split by
    passing ``allowed_rows`` (only windows whose **end bar** lies in that set are
    materialised), which is how a train / val / test fold selects its samples.

    The dataset pre-standardizes the whole feature matrix once at construction
    (cheap, vectorised) and slices windows on demand, so ``__getitem__`` is a
    light tensor copy.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        labels: pd.DataFrame,
        groups: NDArray[np.int64],
        window: int,
        stats: FeatureStats,
        allowed_rows: NDArray[np.int64] | None = None,
    ) -> None:
        """Materialise the valid-window index for a (sub)set of the corpus.

        Args:
            X: Pooled feature matrix, shape ``(N, F)``.
            labels: Oracle label frame row-aligned to ``X``.
            groups: Per-row group id aligned to ``X``, shape ``(N,)``.
            window: Window length ``W`` (``>= 1``).
            stats: Train-split standardization moments (``F`` features).
            allowed_rows: Optional row indices (a CV split); only windows whose
                **end bar** is in this set are kept. ``None`` keeps every valid
                window.

        Raises:
            ValueError: If shapes are inconsistent or ``window`` is invalid.
        """
        n, n_features = X.shape
        if groups.shape[0] != n:
            raise ValueError(
                f"WindowDataset: groups length {groups.shape[0]} != n_rows {n}"
            )
        if len(labels) != n:
            raise ValueError(
                f"WindowDataset: labels length {len(labels)} != n_rows {n}"
            )
        if stats.n_features != n_features:
            raise ValueError(
                f"WindowDataset: stats has {stats.n_features} features, X has "
                f"{n_features}"
            )
        if window < 1:
            raise ValueError(f"WindowDataset: window must be >= 1, got {window}")

        self.window = int(window)
        self.n_features = int(n_features)

        # Standardize once (train-split moments), then impute residual NaN -> 0.
        # 0 is the post-standardization mean, so an imputed feature is neutral.
        raw = X.to_numpy(dtype=np.float32)
        mean = stats.mean.astype(np.float32)
        std = stats.std.astype(np.float32)
        standardized = (raw - mean) / std
        standardized = np.nan_to_num(
            standardized, nan=0.0, posinf=0.0, neginf=0.0
        )
        self._features: NDArray[np.float32] = np.ascontiguousarray(standardized)

        y, w = make_labels(labels)
        self._targets: NDArray[np.float32] = np.ascontiguousarray(y, dtype=np.float32)
        self._weights: NDArray[np.float32] = np.ascontiguousarray(w, dtype=np.float32)
        self._groups: NDArray[np.int64] = np.ascontiguousarray(groups, dtype=np.int64)

        ends = _valid_window_ends(self._groups, self.window)
        if allowed_rows is not None:
            allowed = np.zeros(n, dtype=np.bool_)
            allowed[np.ascontiguousarray(allowed_rows, dtype=np.int64)] = True
            ends = ends[allowed[ends]]
        self._ends: NDArray[np.int64] = np.ascontiguousarray(ends, dtype=np.int64)

        logger.debug(
            "WindowDataset: W=%d, %d valid windows (of %d rows, %d features)%s",
            self.window,
            self._ends.shape[0],
            n,
            self.n_features,
            "" if allowed_rows is None else f" restricted to {allowed_rows.size} rows",
        )

    def __len__(self) -> int:
        """Number of valid windows in the dataset."""
        return int(self._ends.shape[0])

    @property
    def end_indices(self) -> NDArray[np.int64]:
        """The pooled end-row index of each window (aligned to item order).

        Useful for mapping model predictions back to original bar positions for
        event scoring: item ``k`` predicts the turn at bar ``end_indices[k]``.
        """
        return self._ends

    def group_of_item(self, item: int) -> int:
        """Return the (single) group id of window ``item``'s bars.

        Args:
            item: Index into the dataset (``0 <= item < len(self)``).

        Returns:
            The group id shared by all ``W`` bars of the window.
        """
        end = int(self._ends[item])
        return int(self._groups[end])

    def window_groups(self, item: int) -> NDArray[np.int64]:
        """Return the group ids of every bar in window ``item`` (for assertions).

        Args:
            item: Index into the dataset.

        Returns:
            The ``(W,)`` group ids of the window's bars. They are guaranteed equal
            by construction; this exposes them so tests can verify the contract.
        """
        end = int(self._ends[item])
        start = end - self.window + 1
        return np.ascontiguousarray(self._groups[start : end + 1], dtype=np.int64)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return the ``(window, target, weight)`` triple for one window.

        Args:
            item: Index into the dataset (``0 <= item < len(self)``).

        Returns:
            ``(window, target, weight)``:
                ``window`` — ``(W, F)`` float32 standardized feature bars;
                ``target`` — ``(2,)`` float32 per-side binary label at the last
                    bar (top, bottom);
                ``weight`` — ``(2,)`` float32 per-side sample weight.
        """
        end = int(self._ends[item])
        start = end - self.window + 1
        window = self._features[start : end + 1]  # (W, F)
        target = self._targets[end]  # (2,)
        weight = self._weights[end]  # (2,)
        return (
            torch.from_numpy(np.ascontiguousarray(window)),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(np.ascontiguousarray(weight)),
        )
