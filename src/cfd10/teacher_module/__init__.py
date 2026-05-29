"""cfd10 teacher layer: the supervised turn detectors trained on the oracle.

This package holds the models trained against the forward-looking turn oracle.
Its first member is the **GBDT baseline** — the yardstick every later (TCN /
distilled) teacher must beat, and the project's first genuinely out-of-sample
result. :func:`fit_gbdt_cv` fits a per-side LightGBM classifier across
leakage-safe purged walk-forward folds, tunes the decision threshold on the train
fold only, and reports pooled out-of-sample event precision / recall / F1.

Public API
----------
Baseline
    :class:`GBDTConfig` (frozen) and :class:`FoldResult`; :func:`fit_gbdt_cv`.
Registry / factory
    :data:`TEACHER_REGISTRY`, :func:`register_teacher`, :func:`TeacherFactory`
    resolve named baseline fitters for config-driven pipelines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pandas as pd
from numpy.typing import NDArray

from cfd10.cv_module import Fold
from cfd10.teacher_module.baseline import (
    FoldResult,
    GBDTConfig,
    fit_gbdt_cv,
)
from cfd10.utils.logging_conf import get_logger

logger = get_logger(__name__)

# A baseline fitter: (X, y, sample_weight, splits, cfg) -> metrics dict. The
# signature mirrors :func:`fit_gbdt_cv` so any registered teacher is a drop-in.
TeacherFitter = Callable[
    [pd.DataFrame, NDArray, NDArray, list[Fold], GBDTConfig],
    dict[str, object],
]

TEACHER_REGISTRY: dict[str, TeacherFitter] = {}

_T = TypeVar("_T", bound=TeacherFitter)


def register_teacher(name: str) -> Callable[[_T], _T]:
    """Register a baseline fitter under ``name``.

    Args:
        name: Unique registry key.

    Returns:
        A decorator recording the fitter in :data:`TEACHER_REGISTRY` and returning
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: _T) -> _T:
        if name in TEACHER_REGISTRY:
            raise ValueError(f"register_teacher: duplicate registration for {name!r}")
        TEACHER_REGISTRY[name] = fn
        logger.debug("register_teacher: registered %s", name)
        return fn

    return decorator


def TeacherFactory(name: str) -> TeacherFitter:  # noqa: N802 (factory naming)
    """Resolve a registered baseline fitter by name.

    Args:
        name: Registry key used with :func:`register_teacher`.

    Returns:
        The registered fitter callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return TEACHER_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(TEACHER_REGISTRY)) or "<empty>"
        raise KeyError(f"TeacherFactory: unknown teacher {name!r}; known: {known}") from exc


# Populate the registry with the built-in baseline.
register_teacher("gbdt")(fit_gbdt_cv)

__all__ = [
    "GBDTConfig",
    "FoldResult",
    "fit_gbdt_cv",
    "TeacherFitter",
    "TEACHER_REGISTRY",
    "register_teacher",
    "TeacherFactory",
]
