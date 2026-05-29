"""Training loop for the TCN turn teacher (weighted BCE, AdamW, early stop).

This fits the dilated causal TCN against the oracle turn labels. The objective is
a **doubly weighted** binary cross-entropy on the two side logits (top, bottom):

* a class ``pos_weight`` per side (negatives vastly outnumber the sparse turns,
  so each positive's gradient is scaled by ``#neg / #pos``), times
* the **per-sample** oracle weight under the project convention
  (``1 + oracle_score`` on turns, ``1`` otherwise), so a strong turn pulls harder
  than a marginal one.

The loss uses ``BCEWithLogitsLoss(reduction="none", pos_weight=...)`` and folds in
the per-sample weights by multiply-then-mean — numerically stable (log-sum-exp
inside) and exactly the class+sample weighting the baseline uses for LightGBM.

Optimisation is **AdamW** with weight decay; training **early-stops** on the held
validation split (lowest weighted val loss, with patience), restoring the best
weights. It is device-agnostic (``cuda`` if available else ``cpu``) and uses
**AMP autocast + GradScaler on CUDA only** (a no-op on CPU), so the same code path
runs locally on CPU and fast on a Colab T4/L4.

Public API
----------
:class:`TrainConfig` (frozen) and :func:`train_teacher`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from cfd10.teacher_module.models import TCNConfig, TCNTurnModel
from cfd10.utils.logging_conf import get_logger
from cfd10.utils.seed import set_seed

logger = get_logger(__name__)

__all__ = [
    "TrainConfig",
    "resolve_device",
    "train_teacher",
]


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration for the TCN teacher.

    Attributes:
        epochs: Maximum training epochs (early stopping may stop sooner).
        batch_size: Mini-batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (L2 on weights).
        patience: Early-stopping patience in epochs (epochs with no val-loss
            improvement before stopping). ``0`` disables early stopping.
        max_pos_weight: Upper clamp on the per-side class ``pos_weight`` so an
            extremely sparse side does not produce a gigantic gradient scale.
        grad_clip_norm: Max global gradient norm (``<= 0`` disables clipping).
        num_workers: DataLoader worker processes (``0`` = main process; safest on
            Windows / inside the smoke test).
        use_amp: Enable mixed-precision autocast + GradScaler **on CUDA** (ignored
            on CPU). Speeds up the Colab GPU run.
        seed: RNG seed applied before model init and training.
    """

    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    max_pos_weight: float = 50.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    use_amp: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the training knobs."""
        if self.epochs < 1:
            raise ValueError(f"TrainConfig: epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(
                f"TrainConfig: batch_size must be >= 1, got {self.batch_size}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"TrainConfig: lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"TrainConfig: weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.patience < 0:
            raise ValueError(f"TrainConfig: patience must be >= 0, got {self.patience}")
        if self.max_pos_weight <= 0.0:
            raise ValueError(
                f"TrainConfig: max_pos_weight must be > 0, got {self.max_pos_weight}"
            )
        if self.num_workers < 0:
            raise ValueError(
                f"TrainConfig: num_workers must be >= 0, got {self.num_workers}"
            )


# --------------------------------------------------------------------------- #
# Device + helpers.                                                            #
# --------------------------------------------------------------------------- #


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the compute device (CUDA if available, else CPU).

    Args:
        device: An explicit device (string or ``torch.device``); ``None`` selects
            ``cuda`` when available and ``cpu`` otherwise.

    Returns:
        The resolved :class:`torch.device`.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_dataset(
    dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    idx: NDArray[np.int64] | None,
) -> Dataset[tuple[Tensor, Tensor, Tensor]]:
    """Wrap ``dataset`` in a :class:`Subset` when ``idx`` is given, else return it.

    Args:
        dataset: The full windowed dataset.
        idx: Item indices selecting a split, or ``None`` to use the whole dataset.

    Returns:
        The (possibly subset) dataset.
    """
    if idx is None:
        return dataset
    return Subset(dataset, np.ascontiguousarray(idx, dtype=np.int64).tolist())


def _compute_pos_weight(
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    n_sides: int,
    max_pos_weight: float,
    device: torch.device,
) -> Tensor:
    """Estimate the per-side class ``pos_weight = #neg / #pos`` from the loader.

    Iterates the training targets once to count positives / negatives per side.
    A side with no positives falls back to ``pos_weight = 1`` (nothing to up-weight);
    the ratio is clamped to ``max_pos_weight`` so an ultra-sparse side cannot
    dominate the gradient.

    Args:
        loader: The training data loader (yields ``(window, target, weight)``).
        n_sides: Number of output sides (2).
        max_pos_weight: Upper clamp on the ratio.
        device: Device to place the returned tensor on.

    Returns:
        A ``(n_sides,)`` float tensor of clamped class pos-weights.
    """
    pos = torch.zeros(n_sides, dtype=torch.float64)
    total = 0
    for _window, target, _weight in loader:
        pos += target.sum(dim=0).to(torch.float64)
        total += target.shape[0]
    neg = float(total) - pos
    # Guard zero-positive sides: ratio 1.0 (no special up-weighting).
    ratio = torch.where(pos > 0, neg / torch.clamp(pos, min=1.0), torch.ones_like(pos))
    ratio = torch.clamp(ratio, min=1.0, max=max_pos_weight)
    logger.info(
        "train_teacher: per-side pos_weight=%s (pos=%s of %d)",
        [round(r, 2) for r in ratio.tolist()],
        [int(p) for p in pos.tolist()],
        total,
    )
    return ratio.to(device=device, dtype=torch.float32)


def _run_epoch(
    model: TCNTurnModel,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    grad_clip_norm: float,
) -> float:
    """Run one epoch (train if ``optimizer`` given, else eval) and return mean loss.

    The loss is the per-sample-weighted BCE: ``BCEWithLogitsLoss(reduction="none",
    pos_weight=...)`` elementwise, multiplied by the per-side sample weights, then
    averaged over the weight mass (a weighted mean), so the reported number is
    comparable across splits of different size.

    Args:
        model: The TCN.
        loader: Data loader for the split.
        criterion: The configured ``BCEWithLogitsLoss`` (``reduction="none"``).
        device: Compute device.
        optimizer: AdamW for training, or ``None`` for a no-grad eval pass.
        scaler: CUDA ``GradScaler`` (used only when training under AMP), else
            ``None``.
        use_amp: Whether AMP autocast is active (CUDA only).
        grad_clip_norm: Max global grad norm (``<= 0`` disables clipping).

    Returns:
        The weighted-mean loss over the split.
    """
    training = optimizer is not None
    model.train(training)
    amp_device = device.type if (use_amp and device.type == "cuda") else "cpu"
    amp_enabled = use_amp and device.type == "cuda"

    loss_sum = 0.0
    weight_sum = 0.0
    with torch.set_grad_enabled(training):
        for window, target, weight in loader:
            window = window.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=amp_device, enabled=amp_enabled):
                logits = model(window)
                per_elem = criterion(logits, target)  # (N, 2), reduction="none"
                weighted = per_elem * weight
                # Weighted mean over the batch's total weight mass (stable scale).
                batch_weight = weight.sum()
                loss = weighted.sum() / torch.clamp(batch_weight, min=1.0)

            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            # Accumulate in float for a stable epoch average (detach from graph).
            loss_sum += float(weighted.sum().detach().to("cpu"))
            weight_sum += float(batch_weight.detach().to("cpu"))

    return loss_sum / max(weight_sum, 1.0)


# --------------------------------------------------------------------------- #
# Public API.                                                                 #
# --------------------------------------------------------------------------- #


def train_teacher(
    dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    train_idx: NDArray[np.int64] | None,
    val_idx: NDArray[np.int64] | None,
    model_cfg: TCNConfig,
    train_cfg: TrainConfig,
    device: str | torch.device | None = None,
) -> TCNTurnModel:
    """Fit a TCN turn teacher on ``train_idx`` with early stopping on ``val_idx``.

    Builds the model from ``model_cfg``, trains with AdamW under the doubly
    weighted BCE (class ``pos_weight`` x per-sample oracle weight), and — when a
    validation split is given — early-stops on the lowest weighted val loss
    (patience from ``train_cfg``), restoring the best weights before returning.

    The routine is device-agnostic and uses AMP (autocast + GradScaler) on CUDA
    only, so the identical call runs on CPU locally and fast on a Colab GPU.

    Args:
        dataset: The full :class:`cfd10.teacher_module.datasets.WindowDataset`
            (or any dataset yielding ``(window, target, weight)`` with
            ``target``/``weight`` shaped ``(2,)``).
        train_idx: Item indices (into ``dataset``) for the training split;
            ``None`` uses the whole dataset.
        val_idx: Item indices for the validation split; ``None`` disables early
            stopping (the model trains for the full ``epochs``).
        model_cfg: TCN architecture configuration.
        train_cfg: Training configuration (optimiser, batch, early stop, AMP).
        device: Explicit device, or ``None`` to auto-select CUDA/CPU.

    Returns:
        The fitted :class:`cfd10.teacher_module.models.TCNTurnModel` (on
        ``device``, in ``eval`` mode, with the best validation weights restored
        when a val split was provided).

    Raises:
        ValueError: If the training split is empty.
    """
    set_seed(train_cfg.seed)
    dev = resolve_device(device)

    train_ds = _as_dataset(dataset, train_idx)
    if len(train_ds) == 0:  # type: ignore[arg-type]
        raise ValueError("train_teacher: training split is empty")
    has_val = val_idx is not None and len(_as_dataset(dataset, val_idx)) > 0  # type: ignore[arg-type]

    # Drop a trailing singleton batch in training so any BatchNorm-like layer is
    # safe; with plain conv layers it is harmless but keeps batches well-formed.
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        drop_last=len(train_ds) > train_cfg.batch_size,  # type: ignore[arg-type]
    )

    model = TCNTurnModel(model_cfg).to(dev)
    pos_weight = _compute_pos_weight(
        train_loader, model_cfg.n_outputs, train_cfg.max_pos_weight, dev
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    amp_on = train_cfg.use_amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=amp_on)

    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor]] | None = None
    if has_val:
        val_loader = DataLoader(
            _as_dataset(dataset, val_idx),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
        )

    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_no_improve = 0

    for epoch in range(train_cfg.epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            dev,
            optimizer,
            scaler,
            train_cfg.use_amp,
            train_cfg.grad_clip_norm,
        )

        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, criterion, dev, None, None, train_cfg.use_amp, 0.0
            )
            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            logger.info(
                "train_teacher: epoch %d/%d train_loss=%.5f val_loss=%.5f%s",
                epoch + 1,
                train_cfg.epochs,
                train_loss,
                val_loss,
                " *" if improved else "",
            )
            if train_cfg.patience > 0 and epochs_no_improve >= train_cfg.patience:
                logger.info(
                    "train_teacher: early stop at epoch %d (no val improvement "
                    "for %d epochs, best val_loss=%.5f)",
                    epoch + 1,
                    epochs_no_improve,
                    best_val,
                )
                break
        else:
            logger.info(
                "train_teacher: epoch %d/%d train_loss=%.5f",
                epoch + 1,
                train_cfg.epochs,
                train_loss,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
