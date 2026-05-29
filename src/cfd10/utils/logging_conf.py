"""Module-level logger factory."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger.

    Idempotent: attaches a single stream handler the first time a given logger
    name is requested, so repeated calls do not duplicate log lines.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A logger writing INFO+ to stderr with a timestamped format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
