"""
Centralized logging for the multi-agent system.

Usage:
    from .logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("something happened", extra={...})
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False

# Default format — structured enough for production, readable for demos
_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    stream: Optional[object] = None,
) -> None:
    """
    Configure root logging once.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger("agentic_system")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the agentic_system namespace.
    Ensures setup has been called.
    """
    if not _CONFIGURED:
        setup_logging()
    # Normalize: accept both "agents" and "agentic_system.agents"
    if not name.startswith("agentic_system"):
        name = f"agentic_system.{name}"
    return logging.getLogger(name)
