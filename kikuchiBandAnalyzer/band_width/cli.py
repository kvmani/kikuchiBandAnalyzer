"""Command line interface for the band-width pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from kikuchiBandAnalyzer.band_width.config import load_band_width_config
from KikuchiBandWidthAutomator import BandWidthAutomator


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the band-width pipeline argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Run Kikuchi band-width analysis from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        default="bandDetectorOptionsHcp.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force debug logging for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the band-width pipeline CLI.

    Parameters:
        argv: Optional command-line arguments.

    Returns:
        None.
    """

    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config)
    config = load_band_width_config(config_path)
    if args.debug or bool(config.get("debug", False)):
        logging.getLogger().setLevel(logging.DEBUG)
    automator = BandWidthAutomator(config_path=str(config_path))
    automator.run()


if __name__ == "__main__":
    main()
