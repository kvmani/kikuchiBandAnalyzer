"""Top-level package for Kikuchi Band Analyzer utilities."""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"

try:
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    __version__ = "0.0.0"
    _LOGGER.warning("VERSION file not found at %s; using %s.", _VERSION_FILE, __version__)
except OSError as exc:
    __version__ = "0.0.0"
    _LOGGER.warning(
        "Failed to read VERSION file at %s: %s; using %s.",
        _VERSION_FILE,
        exc,
        __version__,
    )
