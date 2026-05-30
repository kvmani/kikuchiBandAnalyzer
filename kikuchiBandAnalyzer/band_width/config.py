"""Configuration loading and validation for band-width workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class BandWidthConfigError(ValueError):
    """Raised when a band-width workflow configuration is invalid."""


_REQUIRED_KEYS = (
    "phase_list",
    "hkl_list",
    "desired_hkl",
    "desired_hkl_ref_width",
    "elastic_modulus",
    "rectWidth",
    "min_psnr",
)


def load_band_width_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a band-width YAML configuration file.

    Parameters:
        config_path: Path to the YAML file.

    Returns:
        Validated configuration dictionary.
    """

    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise BandWidthConfigError(f"Unable to read configuration file: {path}") from exc
    validate_band_width_config(config, base_dir=path.parent)
    return dict(config)


def validate_band_width_config(
    config: Mapping[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
    require_existing_input: bool = True,
) -> None:
    """Validate the minimum contract for a band-width workflow configuration.

    Parameters:
        config: Parsed YAML mapping.
        base_dir: Directory used to resolve relative input paths.
        require_existing_input: Whether ``h5_file_path`` must exist on disk.

    Returns:
        None.
    """

    if not isinstance(config, Mapping):
        raise BandWidthConfigError("Configuration must be a YAML mapping.")

    missing = [key for key in _REQUIRED_KEYS if key not in config]
    if missing:
        raise BandWidthConfigError(
            "Configuration missing required keys: " + ", ".join(missing)
        )

    _validate_source_paths(config, base_dir, require_existing_input)
    _validate_phase(config["phase_list"])
    _validate_hkl_list(config["hkl_list"])
    _validate_numeric(config, "desired_hkl_ref_width", positive=True)
    _validate_numeric(config, "elastic_modulus", positive=True)
    _validate_numeric(config, "rectWidth", positive=True, integer=True)
    _validate_numeric(config, "min_psnr", positive=False)

    if "crop_start" in config or "crop_end" in config:
        if "crop_start" not in config or "crop_end" not in config:
            raise BandWidthConfigError("crop_start and crop_end must be provided together.")
        _validate_numeric(config, "crop_start", positive=False, integer=True)
        _validate_numeric(config, "crop_end", positive=False, integer=True)
        if int(config["crop_end"]) <= int(config["crop_start"]):
            raise BandWidthConfigError("crop_end must be greater than crop_start.")


def _validate_source_paths(
    config: Mapping[str, Any],
    base_dir: str | Path | None,
    require_existing_input: bool,
) -> None:
    """Validate the configured EBSD source paths.

    Parameters:
        config: Configuration mapping.
        base_dir: Directory used to resolve relative paths.
        require_existing_input: Whether input paths must exist.

    Returns:
        None.
    """

    has_h5 = bool(config.get("h5_file_path"))
    has_ctf = bool(config.get("ctf_file_path"))
    if has_h5 == has_ctf:
        raise BandWidthConfigError(
            "Configure exactly one EBSD source: h5_file_path or ctf_file_path."
        )
    if has_h5:
        _validate_input_path(
            config["h5_file_path"],
            base_dir,
            require_existing_input,
            suffixes={".h5", ".hdf5", ".oh5"},
            key="h5_file_path",
        )
        return
    _validate_input_path(
        config["ctf_file_path"],
        base_dir,
        require_existing_input,
        suffixes={".ctf"},
        key="ctf_file_path",
    )
    pattern_folder = config.get("pattern_folder")
    if not isinstance(pattern_folder, str) or not pattern_folder.strip():
        raise BandWidthConfigError("pattern_folder is required when using ctf_file_path.")
    pattern_path = Path(pattern_folder)
    resolved = pattern_path if pattern_path.is_absolute() else Path(base_dir or ".") / pattern_path
    if require_existing_input and not resolved.is_dir():
        raise BandWidthConfigError(f"pattern_folder does not exist: {resolved}")
    _validate_ctf_detector(config.get("ctf_detector", {}))


def _validate_ctf_detector(value: Any) -> None:
    """Validate optional CTF detector geometry configuration.

    Parameters:
        value: Detector geometry mapping.

    Returns:
        None.
    """

    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise BandWidthConfigError("ctf_detector must be a mapping when provided.")
    if "pc" in value:
        pc = value["pc"]
        if not isinstance(pc, (list, tuple)) or len(pc) != 3:
            raise BandWidthConfigError(
                "ctf_detector.pc must contain three values [x*, y*, z*]."
            )
        try:
            [float(component) for component in pc]
        except (TypeError, ValueError) as exc:
            raise BandWidthConfigError("ctf_detector.pc values must be numeric.") from exc
    for key in ("sample_tilt", "tilt", "azimuthal", "px_size"):
        if key not in value:
            continue
        try:
            float(value[key])
        except (TypeError, ValueError) as exc:
            raise BandWidthConfigError(f"ctf_detector.{key} must be numeric.") from exc
    if "binning" in value and not isinstance(value["binning"], int):
        raise BandWidthConfigError("ctf_detector.binning must be an integer.")


def _validate_input_path(
    value: Any,
    base_dir: str | Path | None,
    require_existing_input: bool,
    *,
    suffixes: set[str],
    key: str,
) -> None:
    """Validate a configured input file path."""

    if not isinstance(value, str) or not value.strip():
        raise BandWidthConfigError(f"{key} must be a non-empty string.")
    path = Path(value)
    if path.suffix.lower() not in suffixes:
        suffix_text = ", ".join(sorted(suffixes))
        raise BandWidthConfigError(f"{key} must point to one of: {suffix_text}.")
    resolved = path if path.is_absolute() else Path(base_dir or ".") / path
    if require_existing_input and not resolved.exists():
        raise BandWidthConfigError(f"{key} does not exist: {resolved}")


def _validate_phase(value: Any) -> None:
    """Validate the crystal phase mapping."""

    if not isinstance(value, Mapping):
        raise BandWidthConfigError("phase_list must be a mapping.")
    for key in ("name", "space_group", "lattice"):
        if key not in value:
            raise BandWidthConfigError(f"phase_list missing required key: {key}")
    lattice = value["lattice"]
    if not isinstance(lattice, (list, tuple)) or len(lattice) != 6:
        raise BandWidthConfigError("phase_list.lattice must contain six values.")
    if "atoms" in value and not isinstance(value["atoms"], list):
        raise BandWidthConfigError("phase_list.atoms must be a list when provided.")


def _validate_hkl_list(value: Any) -> None:
    """Validate the configured HKL list."""

    if not isinstance(value, list) or not value:
        raise BandWidthConfigError("hkl_list must be a non-empty list.")
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise BandWidthConfigError("Each hkl_list entry must contain three values.")


def _validate_numeric(
    config: Mapping[str, Any],
    key: str,
    *,
    positive: bool,
    integer: bool = False,
) -> None:
    """Validate a numeric configuration value."""

    value = config[key]
    if integer and not isinstance(value, int):
        raise BandWidthConfigError(f"{key} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise BandWidthConfigError(f"{key} must be numeric.") from exc
    if positive and numeric <= 0:
        raise BandWidthConfigError(f"{key} must be positive.")
