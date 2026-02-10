"""Command-line interface for OH5-to-ANG export workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging, load_yaml_config
from kikuchiBandAnalyzer.oh5_to_ang_exporter.workflow import (
    AngExportResult,
    ColumnMapping,
    export_oh5_to_ang,
)



def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the CLI.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description="Export ANG from OH5 with column mappings.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file path for OH5-to-ANG export.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser



def _parse_mappings(raw: Any) -> list[ColumnMapping]:
    """Parse mapping entries from YAML config payload.

    Parameters:
        raw: Raw ``mappings`` value from YAML.

    Returns:
        List of ColumnMapping entries.

    Raises:
        ValueError: If the mapping format is invalid.
    """

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Config key 'mappings' must be a list.")

    items: list[ColumnMapping] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"Mapping entries must be dictionaries, got: {entry!r}")
        source = entry.get("source")
        target = entry.get("target")
        if source is None or target is None:
            raise ValueError(f"Mapping entry must include 'source' and 'target': {entry!r}")
        items.append(ColumnMapping(source_field=str(source), target_column=str(target), locked=False))
    return items



def _run_from_config(config_path: Path) -> AngExportResult:
    """Run one export from a YAML config file.

    Parameters:
        config_path: YAML configuration path.

    Returns:
        Export summary dataclass.

    Raises:
        KeyError: If required config keys are missing.
    """

    config = load_yaml_config(config_path)
    required = ["oh5_path", "ang_path"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    mappings = _parse_mappings(config.get("mappings"))
    return export_oh5_to_ang(
        oh5_path=Path(config["oh5_path"]),
        ang_path=Path(config["ang_path"]),
        user_mappings=mappings,
        output_ang_path=Path(config["output_ang_path"]) if config.get("output_ang_path") else None,
        include_mapping_note=bool(config.get("include_mapping_note", False)),
        logger=logging.getLogger(__name__),
    )



def main() -> None:
    """CLI entrypoint."""

    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(debug=bool(args.debug), log_config=None)

    result = _run_from_config(args.config)
    logging.getLogger(__name__).info("Export completed: %s", result.output_path)


if __name__ == "__main__":
    main()
