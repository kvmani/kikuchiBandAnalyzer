"""Shared utilities for EBSD compare workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def configure_logging(debug: bool = False, log_config: Optional[Dict[str, Any]] = None) -> None:
    """Configure application logging.

    Parameters:
        debug: Whether to enable DEBUG logging.
        log_config: Optional logging configuration dictionary.

    Returns:
        None.
    """

    log_config = log_config or {}
    level_name = str(log_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    if debug:
        level = logging.DEBUG
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handlers = [logging.StreamHandler()]
    file_path = log_config.get("file_path")
    if file_path:
        handlers.append(logging.FileHandler(file_path))
    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
