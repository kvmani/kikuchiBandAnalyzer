"""Tests for modular package boundaries and quality gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kikuchiBandAnalyzer.band_width.config import (
    BandWidthConfigError,
    validate_band_width_config,
)


def _valid_config(tmp_path: Path) -> dict:
    """Build a minimal valid band-width configuration.

    Parameters:
        tmp_path: Temporary directory for the input file.

    Returns:
        Configuration dictionary.
    """

    h5_path = tmp_path / "scan.oh5"
    h5_path.write_bytes(b"placeholder")
    return {
        "h5_file_path": str(h5_path),
        "phase_list": {
            "name": "Ni",
            "space_group": 225,
            "lattice": [1, 1, 1, 90, 90, 90],
            "atoms": [],
        },
        "hkl_list": [[1, 1, 1]],
        "desired_hkl": "1,1,1",
        "desired_hkl_ref_width": 1.0,
        "elastic_modulus": 1.0,
        "rectWidth": 4,
        "min_psnr": 1.0,
        "crop_start": 0,
        "crop_end": 1,
    }


def test_band_width_config_accepts_minimal_valid_config(tmp_path) -> None:
    """Validate a complete minimal band-width configuration."""

    validate_band_width_config(_valid_config(tmp_path))


def test_band_width_config_rejects_missing_required_key(tmp_path) -> None:
    """Reject configs that cannot run the normal pipeline."""

    config = _valid_config(tmp_path)
    config.pop("phase_list")
    with pytest.raises(BandWidthConfigError, match="phase_list"):
        validate_band_width_config(config)


def test_band_width_config_accepts_ctf_source(tmp_path) -> None:
    """Accept HKL/Oxford CTF plus external pattern-folder source configs."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_folder = tmp_path / "patterns"
    ctf_path.write_text("Channel Text File\n", encoding="utf-8")
    pattern_folder.mkdir()
    config = _valid_config(tmp_path)
    config.pop("h5_file_path")
    config["ctf_file_path"] = str(ctf_path)
    config["pattern_folder"] = str(pattern_folder)
    validate_band_width_config(config)


def test_band_width_config_rejects_ambiguous_sources(tmp_path) -> None:
    """Reject configs that specify both HDF5 and CTF scan sources."""

    config = _valid_config(tmp_path)
    config["ctf_file_path"] = str(tmp_path / "scan.ctf")
    config["pattern_folder"] = str(tmp_path)
    with pytest.raises(BandWidthConfigError, match="exactly one"):
        validate_band_width_config(config, require_existing_input=False)


def test_band_width_config_rejects_invalid_ctf_detector_pc(tmp_path) -> None:
    """Reject CTF detector geometry that cannot define a pattern center."""

    config = _valid_config(tmp_path)
    config.pop("h5_file_path")
    config["ctf_file_path"] = str(tmp_path / "scan.ctf")
    config["pattern_folder"] = str(tmp_path)
    config["ctf_detector"] = {"pc": [0.5, 0.5]}
    with pytest.raises(BandWidthConfigError, match="ctf_detector.pc"):
        validate_band_width_config(config, require_existing_input=False)


def test_modular_imports_are_available() -> None:
    """Ensure new package boundaries expose compatibility APIs."""

    from kikuchiBandAnalyzer.band_width.detector import KikuchiBatchProcessor
    from kikuchiBandAnalyzer.band_width.pipeline import BandWidthAutomator
    from kikuchiBandAnalyzer.band_width.strategy import RectangularAreaBandDetector
    from kikuchiBandAnalyzer.fields import build_default_registry
    from kikuchiBandAnalyzer.io import extract_header_data

    assert KikuchiBatchProcessor is not None
    assert BandWidthAutomator is not None
    assert RectangularAreaBandDetector is not None
    assert build_default_registry is not None
    assert extract_header_data is not None


def test_package_code_has_no_print_or_input_calls() -> None:
    """Keep package modules non-interactive and logging-based."""

    package_root = Path(__file__).resolve().parents[1] / "kikuchiBandAnalyzer"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"print", "input"}:
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno}:{node.func.id}")
    assert offenders == []
