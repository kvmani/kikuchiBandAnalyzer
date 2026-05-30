"""Package facade for band detection strategies."""

from __future__ import annotations

from strategies import (
    BandDetectionStrategy,
    LineTrimmer,
    RectangularAreaBandDetector,
    gaussian,
    strtobool,
)

__all__ = [
    "BandDetectionStrategy",
    "LineTrimmer",
    "RectangularAreaBandDetector",
    "gaussian",
    "strtobool",
]
