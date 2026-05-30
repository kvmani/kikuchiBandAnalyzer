"""Production package entry point for the band-width pipeline."""

from __future__ import annotations

from pathlib import Path

from kikuchiBandAnalyzer.band_width.config import load_band_width_config
from KikuchiBandWidthAutomator import BandWidthAutomator as _LegacyBandWidthAutomator


class BandWidthAutomator(_LegacyBandWidthAutomator):
    """Run the YAML-driven Kikuchi band-width workflow.

    Parameters:
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str | Path = "bandDetectorOptionsHcp.yml") -> None:
        """Initialize the workflow after validating its configuration.

        Parameters:
            config_path: Path to the YAML configuration file.
        """

        load_band_width_config(config_path)
        super().__init__(config_path=str(config_path))
