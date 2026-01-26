"""Application metadata for EBSD Scan Comparator packaging and distribution."""

from __future__ import annotations

import logging
from pathlib import Path


def _read_version_file(path: Path, fallback: str = "0.0.0") -> str:
    """Read the application version from a text file.

    Parameters:
        path: Path to the VERSION file.
        fallback: Version string to use if the file cannot be read.

    Returns:
        Version string.
    """

    logger = logging.getLogger(__name__)
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("VERSION file not found at %s; using %s.", path, fallback)
    except OSError as exc:
        logger.warning("Failed to read VERSION file at %s: %s; using %s.", path, exc, fallback)
    return fallback

APP_NAME = "EBSD Scan Comparator"
APP_SHORT_NAME = "EBSDScanComparator"
APP_ID = "{FB48909A-2AFD-4403-93F9-12A76104BE2E}"
_VERSION_FILE = Path(__file__).resolve().parent / "VERSION"
APP_VERSION = _read_version_file(_VERSION_FILE, fallback="1.0.0")
APP_DESCRIPTION = "GUI for comparing EBSD scan maps and patterns."
APP_AUTHOR = "KikuchiBandAnalyzer contributors"
APP_PUBLISHER = "KikuchiBandAnalyzer"
APP_WEBSITE = "https://github.com/kvmani/kikuchiBandAnalyzer"
APP_COPYRIGHT = "Copyright (c) 2025 KikuchiBandAnalyzer contributors"
APP_LICENSE = "MIT"
APP_EXE_NAME = "EBSD_Scan_Comparator.exe"
APP_ICON_PATH = "assets/icons/ebsd_app.ico"
