"""Shared utilities for EBSD compare workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def configure_logging(debug: bool = False) -> None:
    """Configure application logging.

    Parameters:
        debug: Whether to enable DEBUG logging.
    """

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
