"""Headless comparison runner for EBSD scans."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.field_selection import resolve_scalar_fields
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import alignment_from_config
from kikuchiBandAnalyzer.ebsd_compare.simulated import SimulatedScanFactory
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging, load_yaml_config


class CompareExporter:
    """Export comparison maps for configured scalar fields."""

    def __init__(self, config: Dict, logger: logging.Logger) -> None:
        """Initialize the exporter.

        Parameters:
            config: Configuration dictionary.
            logger: Logger instance.
        """

        self._config = config
        self._logger = logger

    def export(self, scan_a: Path, scan_b: Path, output_dir: Path) -> None:
        """Export map comparisons to PNG and CSV.

        Parameters:
            scan_a: Path to scan A.
            scan_b: Path to scan B.
            output_dir: Output directory for exports.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        field_aliases = self._config.get("field_aliases", {})
        dataset_a = OH5ScanFileReader.from_path(scan_a, field_aliases=field_aliases)
        dataset_b = OH5ScanFileReader.from_path(scan_b, field_aliases=field_aliases)
        try:
            self.export_datasets(dataset_a, dataset_b, output_dir)
        finally:
            dataset_a.close()
            dataset_b.close()

    def export_datasets(self, dataset_a, dataset_b, output_dir: Path) -> None:
        """Export comparison maps for preloaded datasets.

        Parameters:
            dataset_a: ScanDataset for scan A.
            dataset_b: ScanDataset for scan B.
            output_dir: Output directory for exports.

        Returns:
            None.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        alignment = alignment_from_config(self._config.get("alignment", {}), self._logger)
        engine = ComparisonEngine(
            dataset_a, dataset_b, self._config, self._logger, alignment=alignment
        )
        mode = self._config.get("display", {}).get("map_diff_mode", "delta")
        scalar_fields, _ = resolve_scalar_fields(
            self._config,
            dataset_a.catalog,
            dataset_b.catalog,
            logger=self._logger,
        )
        if not scalar_fields:
            self._logger.warning("No scalar fields available for export.")
            return
        for field in scalar_fields:
            maps = engine.map_triplet(field, mode)
            self._export_map_triplet(output_dir, field, maps)

    def _export_map_triplet(
        self, output_dir: Path, field: str, maps: Dict[str, np.ndarray]
    ) -> None:
        """Export a map triplet to PNG and CSV.

        Parameters:
            output_dir: Output directory.
            field: Field name.
            maps: Map triplet dictionary.
        """

        for key, array in maps.items():
            png_path = output_dir / f"{field}_{key}.png"
            csv_path = output_dir / f"{field}_{key}.csv"
            plt.figure(figsize=(4, 4))
            cmap = "coolwarm" if key == "D" else "viridis"
            plt.imshow(array, cmap=cmap)
            plt.title(f"{field} {key}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            np.savetxt(csv_path, array, delimiter=",")
            self._logger.info("Exported %s and %s", png_path, csv_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description="Export EBSD comparison maps")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ebsd_compare_config.yml"),
        help="Path to EBSD compare config.",
    )
    parser.add_argument(
        "--scan-a",
        type=Path,
        required=False,
        help="Path to scan A OH5 file.",
    )
    parser.add_argument(
        "--scan-b",
        type=Path,
        required=False,
        help="Path to scan B OH5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/compare_exports"),
        help="Directory for PNG/CSV exports.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and use simulated scans if paths are omitted.",
    )
    return parser


def main() -> None:
    """Run the compare exporter."""

    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config).get("ebsd_compare", {})
    configure_logging(args.debug, config.get("logging"))
    logger = logging.getLogger(__name__)
    exporter = CompareExporter(config, logger)
    if args.scan_a and args.scan_b:
        exporter.export(args.scan_a, args.scan_b, args.output_dir)
        return
    if args.debug:
        factory = SimulatedScanFactory.from_config(config.get("debug", {}), logger)
        scan_a, scan_b = factory.create_pair()
        exporter.export_datasets(scan_a, scan_b, args.output_dir)
        return
    raise SystemExit("Provide --scan-a/--scan-b or use --debug for simulated data.")


if __name__ == "__main__":
    main()
