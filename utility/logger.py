"""Compatibility shim exposing the project logger as ``utility.logger``.

The vendored :mod:`LLM` package (imported from another project) expects the
shared logger to live in ``utility/logger.py``, while in this repository it is
defined in :mod:`utility.utility`. This module re-exports it so both import
paths resolve to the very same logger instance.
"""

from utility.utility import Logger, bold, italic, logger

__all__ = ["Logger", "bold", "italic", "logger"]
