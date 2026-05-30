"""Convert the bundled HKL/Oxford CTF fixture into TSL-style ANG and OH5 files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kikuchiBandAnalyzer.io.hkl_ctf_to_tsl import convert_hkl_ctf_fixture_to_tsl


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the fixture converter.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Convert testData/hkl_ctf_test_data into DA-compatible ANG/H5/OH5 outputs."
    )
    parser.add_argument(
        "--ctf",
        type=Path,
        default=Path("testData/hkl_ctf_test_data/Subset.ctf"),
        help="Path to the HKL/Oxford CTF file.",
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=Path("testData/hkl_ctf_test_data/Binned_2x2"),
        help="Directory containing HKL pattern TIFF images.",
    )
    parser.add_argument(
        "--reference-h5",
        type=Path,
        default=Path("testData/DA.h5"),
        help="DA HDF5 file used as the TSL metadata template.",
    )
    parser.add_argument(
        "--reference-ang",
        type=Path,
        default=Path("testData/DA.ang"),
        help="DA ANG file used as the ANG header template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testData/hkl_ctf_test_data/converted_tsl"),
        help="Directory for generated ANG/H5/OH5 files.",
    )
    parser.add_argument(
        "--scan-name",
        default="Subset_HKL_TSL",
        help="Top-level scan name and output file stem.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    """Run the HKL fixture conversion from command-line arguments.

    Returns:
        None.
    """

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    result = convert_hkl_ctf_fixture_to_tsl(
        ctf_path=args.ctf,
        pattern_dir=args.patterns,
        reference_h5_path=args.reference_h5,
        reference_ang_path=args.reference_ang,
        output_dir=args.output_dir,
        scan_name=args.scan_name,
    )
    logging.info("Wrote H5: %s", result.h5_path)
    logging.info("Wrote OH5: %s", result.oh5_path)
    logging.info("Wrote ANG: %s", result.ang_path)


if __name__ == "__main__":
    main()
