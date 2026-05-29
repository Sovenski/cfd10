"""Isotonic probability calibration for the TCN turn teacher.

A neural classifier's raw ``sigmoid(logit)`` outputs are usually *uncalibrated* —
the score ranks turns well but its absolute value is not a faithful probability,
which matters when a single decision threshold must be chosen and shared across
folds / distilled into the student. This module fits a **per-side isotonic
regression** (a monotone, non-parametric map ``raw_prob -> calibrated_prob``) on a
**held-out** fold, so calibration never sees the data it is applied to.

Isotonic (vs Platt scaling) is the natural fit here: it is monotone (preserves the
model's ranking, so event recall/precision at a rank are unchanged) yet flexible
enough to correct the systematic over/under-confidence sparse-label BCE induces.

Public API
----------
:func:`predict_proba` (run a fitted model over a dataset -> per-side raw probs +
their end-bar indices and targets); :class:`SideCalibrator` (one side's fitted
isotonic map); :func:`fit_calibrators` (fit both sides on a held fold);
:func:`apply_calibrators`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from cfd10.teacher_module.models import TCNTurnModel
from cfd10.teacher_module.train import resolve_device
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "ProbaResult",
    "SideCalibrator",
    "predict_proba",
    "fit_calibrators",
    "apply_calibrators",
]

_N_SIDES: int = 2


# --------------------------------------------------------------------------- #
# Inference over a dataset.                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbaResult:
    """Per-side raw probabilities from a model over a (sub)set of windows.

    Attributes:
        proba: Raw ``sigmoid(logit)`` per side, shape ``(M, 2)`` (col 0 = top,
            col 1 = bottom), ``float64``.
        targets: Per-side binary targets aligned to ``proba``, shape ``(M, 2)``.
        end_idx: Pooled end-row index of each window, shape ``(M,)`` — maps a row
            of ``proba`` back to the bar it scores (for event evaluation).
    """

    proba: NDArray[np.float64]
    targets: NDArray[np.float64]
    end_idx: NDArray[np.int64]


def predict_proba(
    model: TCNTurnModel,
    dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    item_idx: NDArray[np.int64] | None = None,
    batch_size: int = 512,
    device: str | torch.device | None = None,
) -> ProbaResult:
    """Run ``model`` over ``dataset`` and return per-side raw turn probabilities.

    The model is evaluated with gradients disabled in ``eval`` mode; the per-side
    ``sigmoid`` of the logits is returned together with the targets and (when the
    dataset exposes ``end_indices``) the pooled end-bar index of each window, so
    callers can score events against the oracle.

    Args:
        model: A fitted TCN.
        dataset: A windowed dataset (full corpus or a split) yielding
            ``(window, target, weight)``.
        item_idx: Optional item indices selecting a split; ``None`` scores all.
        batch_size: Inference batch size.
        device: Explicit device, or ``None`` to auto-select.

    Returns:
        The :class:`ProbaResult`.
    """
    dev = resolve_device(device)
    model = model.to(dev)
    model.eval()

    sub: Dataset[tuple[Tensor, Tensor, Tensor]]
    if item_idx is not None:
        order = np.ascontiguousarray(item_idx, dtype=np.int64)
        sub = Subset(dataset, order.tolist())
    else:
        order = None
        sub = dataset

    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        sub, batch_size=batch_size, shuffle=False
    )
    proba_blocks: list[NDArray[np.float64]] = []
    target_blocks: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for window, target, _weight in loader:
            window = window.to(dev)
            logits = model(window)
            proba = torch.sigmoid(logits).to("cpu", dtype=torch.float64).numpy()
            proba_blocks.append(np.ascontiguousarray(proba))
            target_blocks.append(
                np.ascontiguousarray(target.to(torch.float64).numpy())
            )

    proba_arr = (
        np.concatenate(proba_blocks)
        if proba_blocks
        else np.empty((0, _N_SIDES), dtype=np.float64)
    )
    target_arr = (
        np.concatenate(target_blocks)
        if target_blocks
        else np.empty((0, _N_SIDES), dtype=np.float64)
    )

    # Map each scored item back to its pooled end-bar index, honouring item_idx.
    end_idx = _resolve_end_indices(dataset, order, proba_arr.shape[0])
    return ProbaResult(proba=proba_arr, targets=target_arr, end_idx=end_idx)


def _resolve_end_indices(
    dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    order: NDArray[np.int64] | None,
    n_scored: int,
) -> NDArray[np.int64]:
    """Return the pooled end-bar index per scored item (or ``-1`` if unavailable).

    Args:
        dataset: The dataset that was scored.
        order: The item order used (``None`` == natural order).
        n_scored: Number of scored items (for the fallback shape).

    Returns:
        End-bar indices aligned to the scored items, ``int64``.
    """
    ends = getattr(dataset, "end_indices", None)
    if ends is None:
        return np.full(n_scored, -1, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    if order is None:
        return ends
    return np.ascontiguousarray(ends[order], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Per-side isotonic calibrator.                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SideCalibrator:
    """One side's fitted isotonic ``raw_prob -> calibrated_prob`` map.

    Attributes:
        side: The side this calibrator was fit for (``0`` = top, ``1`` = bottom).
        model: The fitted :class:`sklearn.isotonic.IsotonicRegression`, or
            ``None`` if the held fold lacked both classes (then calibration is the
            identity — nothing reliable to fit).
    """

    side: int
    model: IsotonicRegression | None

    def transform(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw probabilities through the isotonic fit (identity if unfit).

        Args:
            raw: Raw probabilities in ``[0, 1]``, any shape.

        Returns:
            Calibrated probabilities, same shape, clipped to ``[0, 1]``.
        """
        if self.model is None:
            return np.clip(np.ascontiguousarray(raw, dtype=np.float64), 0.0, 1.0)
        flat = np.ascontiguousarray(raw, dtype=np.float64).reshape(-1)
        out = self.model.predict(flat)
        return np.clip(out, 0.0, 1.0).reshape(np.asarray(raw).shape)


def fit_calibrators(result: ProbaResult) -> list[SideCalibrator]:
    """Fit a per-side isotonic calibrator on a held fold's raw probs / targets.

    For each side an :class:`sklearn.isotonic.IsotonicRegression` (clipped to
    ``[0, 1]``, ``out_of_bounds="clip"``) is fit mapping the model's raw
    probability to the binary target. A side whose held fold has only one class
    (all turns or no turns) cannot be calibrated meaningfully and falls back to the
    identity map.

    Args:
        result: Raw probabilities + targets from :func:`predict_proba` on the
            **held calibration fold** (kept disjoint from the data calibration is
            later applied to).

    Returns:
        A list of two :class:`SideCalibrator` (top, bottom).
    """
    calibrators: list[SideCalibrator] = []
    for side in range(_N_SIDES):
        raw = result.proba[:, side]
        tgt = result.targets[:, side]
        unique = np.unique(tgt)
        if raw.size == 0 or unique.size < 2:
            logger.warning(
                "fit_calibrators[side=%d]: held fold has <2 classes (or empty); "
                "using identity calibration",
                side,
            )
            calibrators.append(SideCalibrator(side=side, model=None))
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw, tgt)
        calibrators.append(SideCalibrator(side=side, model=iso))
        logger.info(
            "fit_calibrators[side=%d]: isotonic fit on %d samples (pos_rate=%.4f)",
            side,
            raw.size,
            float(tgt.mean()),
        )
    return calibrators


def apply_calibrators(
    calibrators: list[SideCalibrator], proba: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply per-side calibrators to a ``(M, 2)`` raw-probability matrix.

    Args:
        calibrators: The two fitted side calibrators (top, bottom).
        proba: Raw per-side probabilities, shape ``(M, 2)``.

    Returns:
        Calibrated per-side probabilities, shape ``(M, 2)``.

    Raises:
        ValueError: If ``proba`` does not have two columns or the calibrator count
            mismatches.
    """
    if proba.ndim != 2 or proba.shape[1] != _N_SIDES:
        raise ValueError(
            f"apply_calibrators: expected (M, {_N_SIDES}) proba, got {proba.shape}"
        )
    if len(calibrators) != _N_SIDES:
        raise ValueError(
            f"apply_calibrators: expected {_N_SIDES} calibrators, got "
            f"{len(calibrators)}"
        )
    out = np.empty_like(proba, dtype=np.float64)
    for side, cal in enumerate(calibrators):
        out[:, side] = cal.transform(proba[:, side])
    return out
