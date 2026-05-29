"""Model registry + factory for the neural turn-detector teachers.

A tiny name -> builder registry mirroring the rest of the codebase's factory
pattern (datasets, features, splitters, baselines). A *builder* maps a frozen
config object to an initialised ``torch.nn.Module``; registering it under a name
lets config-driven pipelines resolve a model without importing its class.

Kept in its own module (not ``__init__``) so concrete model modules can import
:func:`register_model` at definition time without a circular import through the
package ``__init__``.

Public API
----------
:data:`MODEL_REGISTRY`, :func:`register_model`, :func:`ModelFactory`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from torch import nn

from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

__all__ = [
    "ModelBuilder",
    "MODEL_REGISTRY",
    "register_model",
    "ModelFactory",
]

# A model builder: maps a config object to an initialised module. The config type
# is intentionally loose (each model defines its own frozen config dataclass).
ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {}

_B = TypeVar("_B", bound=ModelBuilder)


def register_model(name: str) -> Callable[[_B], _B]:
    """Register a model builder under ``name``.

    Args:
        name: Unique registry key (e.g. ``"tcn"``).

    Returns:
        A decorator recording the builder in :data:`MODEL_REGISTRY` and returning
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(builder: _B) -> _B:
        if name in MODEL_REGISTRY:
            raise ValueError(f"register_model: duplicate registration for {name!r}")
        MODEL_REGISTRY[name] = builder
        logger.debug("register_model: registered %s", name)
        return builder

    return decorator


def ModelFactory(name: str) -> ModelBuilder:  # noqa: N802 (factory naming)
    """Resolve a registered model builder by name.

    Args:
        name: Registry key used with :func:`register_model`.

    Returns:
        The registered builder callable (``cfg -> nn.Module``).

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "<empty>"
        raise KeyError(f"ModelFactory: unknown model {name!r}; known: {known}") from exc
