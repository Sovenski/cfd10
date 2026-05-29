"""Dilated causal TCN: the deep per-side turn detector (teacher model).

This is the neural teacher the project trains against the forward-looking turn
oracle — a small **Temporal Convolutional Network** (Bai et al., 2018). It reads
a window of ``W`` bars by ``F`` features and emits **two logits** (a *top* turn
logit and a *bottom* turn logit) for the window's last bar, so one network scores
both sides at once and shares the temporal representation between them.

Why a TCN (not an RNN / plain CNN)
----------------------------------
* **Causal** — every output at time ``t`` depends only on inputs ``<= t`` (left
  padding, cropped to be exact), so there is no look-ahead leakage inside a
  window. Combined with the windower never crossing an asset boundary and the
  purged CV, the leakage contract holds end to end.
* **Dilated** — stacking dilations ``[1, 2, 4, 8]`` grows the receptive field
  exponentially (``1 + 2 * (k - 1) * sum(dilations)`` bars for kernel ``k``),
  so a handful of residual blocks see a multi-week context cheaply.
* **Residual** — each block adds its (gated, dropped) transform to a 1x1
  projection of its input, which stabilises optimisation of the deeper stack.

Capacity
--------
The defaults (``channels=24``, 4 blocks, kernel 3) sit at ~32k parameters — well
under the 100k budget — small enough that a T4/L4 trains it in minutes yet
expressive enough to beat the GBDT baseline. :func:`TCNTurnModel.count_parameters`
reports the exact trainable count.

Public API
----------
:class:`TCNConfig` (frozen) — architecture hyper-parameters with validation.
:class:`TCNTurnModel` — the ``nn.Module`` (input ``(N, W, F)``, output
``(N, 2)``). The model registry / factory (:func:`register_model`,
:func:`ModelFactory`) live in :mod:`cfd10.teacher_module.models`; this module
registers ``"tcn"`` on import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor, nn

from cfd10.teacher_module.models.registry import register_model
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "TCNConfig",
    "TCNTurnModel",
]

# Two output logits: index 0 = top (high) turn, index 1 = bottom (low) turn.
_N_SIDES: int = 2

_DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)


# --------------------------------------------------------------------------- #
# Configuration.                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TCNConfig:
    """Immutable TCN architecture configuration.

    Attributes:
        in_features: Number of input feature channels ``F`` (the feature-bank
            width). Set by the caller from the data.
        channels: Hidden channel width used by every residual block. The dominant
            capacity knob; keep it small (the default keeps the net < 100k params).
        kernel_size: Temporal kernel width of each causal convolution (``>= 2``;
            odd or even both fine — causality is enforced by explicit cropping).
        dilations: Per-block dilation factors (one residual block per entry). The
            receptive field grows with their sum, so ``(1, 2, 4, 8)`` already sees
            a multi-week daily context.
        dropout: Dropout probability applied inside each residual block (``[0, 1)``).
        n_outputs: Number of output logits; fixed at 2 (top, bottom) but kept
            configurable for clarity.
    """

    in_features: int
    channels: int = 48
    kernel_size: int = 3
    dilations: tuple[int, ...] = field(default=_DEFAULT_DILATIONS)
    dropout: float = 0.1
    n_outputs: int = _N_SIDES

    def __post_init__(self) -> None:
        """Validate the architecture knobs."""
        if self.in_features < 1:
            raise ValueError(f"TCNConfig: in_features must be >= 1, got {self.in_features}")
        if self.channels < 1:
            raise ValueError(f"TCNConfig: channels must be >= 1, got {self.channels}")
        if self.kernel_size < 2:
            raise ValueError(
                f"TCNConfig: kernel_size must be >= 2, got {self.kernel_size}"
            )
        if len(self.dilations) == 0:
            raise ValueError("TCNConfig: dilations must be non-empty")
        if any(d < 1 for d in self.dilations):
            raise ValueError(f"TCNConfig: dilations must be >= 1, got {self.dilations}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"TCNConfig: dropout must be in [0, 1), got {self.dropout}")
        if self.n_outputs < 1:
            raise ValueError(f"TCNConfig: n_outputs must be >= 1, got {self.n_outputs}")

    @property
    def receptive_field(self) -> int:
        """Number of past bars the last output depends on.

        For a stack of dilated causal convolutions of kernel ``k`` the receptive
        field is ``1 + (k - 1) * sum(dilations)`` (a single conv layer per block).

        Returns:
            The receptive field in bars.
        """
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


# --------------------------------------------------------------------------- #
# Causal building blocks.                                                     #
# --------------------------------------------------------------------------- #


class _Chomp1d(nn.Module):
    """Crop the right end of a tensor to restore causality after left padding.

    A ``Conv1d`` with ``padding=p`` pads *both* ends; to make the convolution
    strictly causal we pad only the left, which a symmetric pad over-achieves by
    ``p`` extra steps on the right. This module removes exactly those ``p`` steps
    so the output length equals the input length and depends only on past inputs.
    """

    def __init__(self, chomp_size: int) -> None:
        """Store how many trailing steps to drop.

        Args:
            chomp_size: Number of right-end timesteps to remove (``>= 0``).
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        """Drop the last ``chomp_size`` timesteps along the time axis.

        Args:
            x: Tensor shaped ``(N, C, L)``.

        Returns:
            ``x[..., : L - chomp_size]`` (contiguous), or ``x`` unchanged if
            ``chomp_size == 0``.
        """
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """One residual dilated-causal block: Conv -> ReLU -> Dropout, then + skip.

    The block keeps the channel width fixed at ``channels`` and the sequence
    length unchanged (causal padding + chomp), so blocks stack uniformly. A 1x1
    convolution projects the input for the residual add when the in/out channel
    counts differ (only the first block, mapping ``F -> channels``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        """Build the convolution, activation, dropout, and residual projection.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count (the hidden width).
            kernel_size: Temporal kernel width.
            dilation: Dilation factor for this block.
            dropout: Dropout probability applied after the activation.
        """
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.chomp = _Chomp1d(pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 1x1 projection only when the residual needs a channel match.
        self.downsample: nn.Module
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()
        self.out_relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated dilated conv and add the (projected) residual.

        Args:
            x: Tensor shaped ``(N, in_channels, L)``.

        Returns:
            Tensor shaped ``(N, out_channels, L)`` (length preserved).
        """
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = self.downsample(x)
        return self.out_relu(out + res)


# --------------------------------------------------------------------------- #
# The model.                                                                  #
# --------------------------------------------------------------------------- #


class TCNTurnModel(nn.Module):
    """A small dilated causal TCN emitting top/bottom turn logits per window.

    The forward pass takes a batch of windows ``(N, W, F)`` (the layout
    :class:`cfd10.teacher_module.datasets.WindowDataset` yields), transposes to
    the ``Conv1d`` channel-first layout, runs the residual dilated stack, takes
    the representation at the **last timestep** (the window's most recent bar —
    the bar being labelled), and maps it through a linear head to two logits.

    Per the codebase convention the constructor takes a single ``cfg`` object so
    the model is fully config-driven.
    """

    def __init__(self, cfg: TCNConfig) -> None:
        """Build the dilated residual stack and the linear output head.

        Args:
            cfg: The architecture configuration.
        """
        super().__init__()
        self.cfg = cfg

        blocks: list[nn.Module] = []
        in_ch = cfg.in_features
        for dilation in cfg.dilations:
            blocks.append(
                _TemporalBlock(
                    in_channels=in_ch,
                    out_channels=cfg.channels,
                    kernel_size=cfg.kernel_size,
                    dilation=dilation,
                    dropout=cfg.dropout,
                )
            )
            in_ch = cfg.channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(cfg.channels, cfg.n_outputs)

        logger.debug(
            "TCNTurnModel: in_features=%d channels=%d blocks=%d kernel=%d "
            "receptive_field=%d params=%d",
            cfg.in_features,
            cfg.channels,
            len(cfg.dilations),
            cfg.kernel_size,
            cfg.receptive_field,
            self.count_parameters(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of windows to top/bottom turn logits.

        Args:
            x: Windows shaped ``(N, W, F)`` — ``N`` windows of ``W`` bars by
                ``F`` features (the dataset's native layout).

        Returns:
            Logits shaped ``(N, n_outputs)`` (``n_outputs == 2``: top, bottom).
            Apply ``torch.sigmoid`` to obtain per-side turn probabilities.

        Raises:
            ValueError: If ``x`` is not a 3-D tensor or its feature dimension does
                not match ``cfg.in_features``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TCNTurnModel.forward: expected 3-D input (N, W, F), got shape "
                f"{tuple(x.shape)}"
            )
        if x.shape[-1] != self.cfg.in_features:
            raise ValueError(
                f"TCNTurnModel.forward: feature dim {x.shape[-1]} != "
                f"in_features {self.cfg.in_features}"
            )
        # (N, W, F) -> (N, F, W) for Conv1d (channels = features, length = time).
        x = x.transpose(1, 2)
        out = self.network(x)
        # Representation at the last (most recent) timestep -> (N, channels).
        last = out[:, :, -1]
        return self.head(last)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the model's parameter count.

        Args:
            trainable_only: If ``True`` (default) count only parameters with
                ``requires_grad``; otherwise count all parameters.

        Returns:
            The number of (trainable) parameters.
        """
        params = (
            p for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
        return int(sum(p.numel() for p in params))


# Register the TCN under the factory name ``"tcn"`` on import.
@register_model("tcn")
def _build_tcn(cfg: TCNConfig) -> TCNTurnModel:
    """Factory builder for the TCN teacher (registered as ``"tcn"``).

    Args:
        cfg: The :class:`TCNConfig` architecture configuration.

    Returns:
        An initialised :class:`TCNTurnModel`.
    """
    return TCNTurnModel(cfg)
