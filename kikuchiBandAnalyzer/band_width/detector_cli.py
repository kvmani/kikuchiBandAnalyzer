"""Command line interface for standalone band detection on image fixtures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from configLoader import load_config
import utilities as ut
from kikuchiBandAnalyzer.band_width.detector import (
    KikuchiBatchProcessor,
    load_ebsd_data,
    prepare_json_input,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone detector argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Run standalone Kikuchi band detection on image or numpy data."
    )
    parser.add_argument("--source", required=True, help="Image file, image folder, or .npy source.")
    parser.add_argument("--annotations", required=True, help="JSON annotation file.")
    parser.add_argument("--config", required=True, help="Band detector YAML configuration.")
    parser.add_argument("--raw-output", default="bandOutputData.csv", help="Raw CSV output path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--filtered-output",
        default="filtered_band_data.csv",
        help="Filtered CSV output path.",
    )
    parser.add_argument("--tile-rows", type=int, default=1, help="Rows when tiling one image.")
    parser.add_argument("--tile-cols", type=int, default=1, help="Columns when tiling one image.")
    parser.add_argument(
        "--tile-from-single",
        action="store_true",
        help="Repeat one JSON annotation for all loaded patterns.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run standalone band detection.

    Parameters:
        argv: Optional command-line arguments.

    Returns:
        None.
    """

    args = build_arg_parser().parse_args(argv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = load_config(args.config)
    if args.debug:
        config["debug"] = True
        config["plot_band_detection"] = True
        config.setdefault("plot_band_detection_condition", "False")
        logging.info(
            "Debug mode enabled: interactive band diagnostic plots will be shown "
            "for detected candidates. Close plot windows to continue processing."
        )
    ebsd_data = load_ebsd_data(args.source, tile_rows=args.tile_rows, tile_cols=args.tile_cols)
    n_patterns = int(np.prod(ebsd_data.shape[:2]))
    json_input = prepare_json_input(
        args.annotations,
        n_patterns=n_patterns,
        tile_from_single=bool(args.tile_from_single),
    )
    processor = KikuchiBatchProcessor(
        ebsd_data,
        json_input,
        config=config,
        desired_hkl=config.get("desired_hkl", "110"),
    )
    results = processor.process()
    if args.json_output:
        ut.save_results_to_json(results, path=args.json_output)
    ut.save_results_to_csv(
        results,
        raw_path=args.raw_output,
        filtered_path=args.filtered_output,
    )
    logging.info("Wrote standalone detector outputs to %s and %s.", args.raw_output, args.filtered_output)


if __name__ == "__main__":
    main()
