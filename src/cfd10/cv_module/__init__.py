"""cfd10 cross-validation layer: leakage-safe walk-forward splitters.

This package provides the cross-validation harness for serially correlated,
overlapping-label market data. Its centrepiece is
:func:`purged_walk_forward` — López de Prado's *purged + embargoed* expanding
walk-forward, the only split that does not let a training label window peek into
the test block (purge) or sit in the serially-correlated bars right after it
(embargo). With ``pooled_groups`` the purge/embargo are applied within each
asset's own bar clock so assets cannot leak across one another.

Public API
----------
Splitters
    :func:`purged_walk_forward` and the defensive :class:`LeakageError`.
Registry / factory
    :data:`SPLITTER_REGISTRY`, :func:`register_splitter`, :func:`SplitterFactory`
    resolve named splitter callables for config-driven pipelines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from cfd10.cv_module.splits import (
    Fold,
    LeakageError,
    purged_walk_forward,
)
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

# A splitter: maps integer timestamps (+ kwargs) to a list of train/test folds.
# The signature is intentionally loose (``*args``/``**kwargs``) because distinct
# splitters take different parameters; callers resolve a name and pass that
# splitter's documented arguments.
Splitter = Callable[..., list[Fold]]

SPLITTER_REGISTRY: dict[str, Splitter] = {}

_S = TypeVar("_S", bound=Splitter)


def register_splitter(name: str) -> Callable[[_S], _S]:
    """Register a CV splitter under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator that records the splitter in :data:`SPLITTER_REGISTRY` and
        returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: _S) -> _S:
        if name in SPLITTER_REGISTRY:
            raise ValueError(f"register_splitter: duplicate registration for {name!r}")
        SPLITTER_REGISTRY[name] = fn
        logger.debug("register_splitter: registered %s", name)
        return fn

    return decorator


def SplitterFactory(name: str) -> Splitter:  # noqa: N802 (factory naming)
    """Resolve a registered splitter by name.

    Args:
        name: Registry key used with :func:`register_splitter`.

    Returns:
        The registered splitter callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return SPLITTER_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(SPLITTER_REGISTRY)) or "<empty>"
        raise KeyError(
            f"SplitterFactory: unknown splitter {name!r}; known: {known}"
        ) from exc


# Populate the registry with the built-in splitter.
register_splitter("purged_walk_forward")(purged_walk_forward)

__all__ = [
    "purged_walk_forward",
    "LeakageError",
    "Fold",
    "Splitter",
    "SPLITTER_REGISTRY",
    "register_splitter",
    "SplitterFactory",
]
