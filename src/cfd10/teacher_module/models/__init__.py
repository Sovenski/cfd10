"""Neural turn-detector models for the cfd10 teacher layer.

This package holds the deep teachers trained against the turn oracle, exposed
through a small name -> builder factory so config-driven pipelines resolve a model
by name. Its first member is the dilated causal :class:`TCNTurnModel` (registered
as ``"tcn"``), which reads a window of bars-by-features and emits top/bottom turn
logits.

Importing this package registers all built-in models, so
``ModelFactory("tcn")(cfg)`` works without importing the concrete module.

Public API
----------
Models
    :class:`TCNConfig` (frozen) and :class:`TCNTurnModel`.
Registry / factory
    :data:`MODEL_REGISTRY`, :func:`register_model`, :func:`ModelFactory`.
"""

from __future__ import annotations

# Import the concrete models for their registration side effects, then re-export.
from cfd10.teacher_module.models.registry import (
    MODEL_REGISTRY,
    ModelBuilder,
    ModelFactory,
    register_model,
)
from cfd10.teacher_module.models.tcn import TCNConfig, TCNTurnModel

__all__ = [
    "TCNConfig",
    "TCNTurnModel",
    "ModelBuilder",
    "MODEL_REGISTRY",
    "register_model",
    "ModelFactory",
]
