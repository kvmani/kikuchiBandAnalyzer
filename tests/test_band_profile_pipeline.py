"""Tests for band profile outputs and JSON handling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
distutils_module = types.ModuleType("distutils")
distutils_util = types.ModuleType("distutils.util")
distutils_util.strtobool = lambda value: value in ("1", "true", "True", True)
distutils_module.util = distutils_util
sys.modules.setdefault("distutils", distutils_module)
sys.modules.setdefault("distutils.util", distutils_util)

import utilities as ut
from KikuchiBandWidthAutomator import BandWidthAutomator
from kikuchiBandWidthDetector import prepare_json_input, KikuchiBatchProcessor


def _write_min_ang(path: Path, nrows: int, ncols_even: int, column_headers: list[str]) -> None:
    """
    Write a minimal .ang file compatible with modify_ang_file.

    Parameters:
        path: Output file path.
        nrows: Number of rows in the data section.
        ncols_even: Number of columns per row.
        column_headers: Column header names.

    Returns:
        None.
    """
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# HEADER: Start\n")
        handle.write("# COLUMN_HEADERS: " + ", ".join(column_headers) + "\n")
        handle.write(f"# NCOLS_EVEN: {ncols_even}\n")
        handle.write(f"# NROWS: {nrows}\n")
        handle.write("# HEADER: End\n")
        for _ in range(nrows):
            handle.write("  ".join(["0.00"] * ncols_even) + "\n")


def _create_min_h5(path: Path, scan_name: str, n_pixels: int) -> None:
    """
    Create a minimal HDF5 file with a CI dataset.

    Parameters:
        path: Output file path.
        scan_name: Scan group name.
        n_pixels: Number of pixels (flattened).

    Returns:
        None.
    """
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Manufacturer", data="Debug")
        handle.create_dataset("Version", data="1.0")
        scan = handle.create_group(scan_name)
        ebsd = scan.create_group("EBSD")
        ebsd.create_group("Header")
        data = ebsd.create_group("Data")
        data.create_dataset("CI", data=np.zeros(n_pixels, dtype=np.float32))


def test_save_results_to_json_includes_band_profile(tmp_path) -> None:
    """Ensure band_profile is serialized to JSON."""
    results = [
        {
            "x,y": [0, 0],
            "ind": 0,
            "bands": [
                {
                    "bandWidth": 1.0,
                    "psnr": 2.0,
                    "band_profile": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "central_line": [1.0, 2.0, 3.0, 4.0],
                    "band_valid": True,
                }
            ],
        }
    ]
    output_path = tmp_path / "results.json"
    ut.save_results_to_json(results, path=output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    band_profile = data[0]["bands"][0]["band_profile"]
    assert np.allclose(band_profile, [0.1, 0.2, 0.3])
    assert data[0]["bands"][0]["central_line"] == [1.0, 2.0, 3.0, 4.0]


def test_prepare_json_input_handles_optional_pattern_path(tmp_path) -> None:
    """Support JSON inputs with or without pattern_path."""
    payload = [
        {"points": [], "pattern_path": "patterns/pattern_0_0.png"},
        {"points": []},
    ]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    results = prepare_json_input(str(path), n_patterns=2, tile_from_single=False)
    assert results[0]["pattern_path"] == "patterns/pattern_0_0.png"
    assert "pattern_path" not in results[1]


def test_batch_processor_preserves_pattern_path(monkeypatch) -> None:
    """Ensure process_kikuchi_image_at_pixel passes through pattern_path."""
    ebsd_data = np.zeros((1, 1, 4, 4), dtype=np.float32)
    config = {
        "rectWidth": 2,
        "min_psnr": 1.0,
        "smoothing_sigma": 1.0,
        "phase_list": {"name": "Ni", "space_group": 225, "lattice": [1, 1, 1, 90, 90, 90]},
    }

    def _fake_detect_bands(self):
        return [
            {
                "bandWidth": 1.0,
                "psnr": 2.0,
                "band_valid": True,
                "band_profile": [0.0] * 8,
                "central_line": [0.0, 1.0, 2.0, 3.0],
            }
        ]

    monkeypatch.setattr("kikuchiBandWidthDetector.BandDetector.detect_bands", _fake_detect_bands)

    processor = KikuchiBatchProcessor(
        ebsd_data=ebsd_data,
        json_input=[{"points": [], "pattern_path": "(0,0)"}],
        config=config,
        desired_hkl="110",
        phase=None,
    )
    entry = processor.process_kikuchi_image_at_pixel(0, 0, processor.json_input[0])
    assert entry["pattern_path"] == "(0,0)"
    assert entry["bands"][0]["band_profile"] == [0.0] * 8


def test_export_results_writes_band_profile_dataset(tmp_path) -> None:
    """Write band_profile and central_line datasets to HDF5."""
    scan_name = "Scan"
    n_pixels = 2
    h5_path = tmp_path / "scan.oh5"
    _create_min_h5(h5_path, scan_name, n_pixels)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "h5_file_path: " + str(h5_path),
                "desired_hkl_ref_width: 1.0",
                "elastic_modulus: 1.0",
                "desired_hkl: 110",
                "rectWidth: 2",
                "phase_list:",
                "  name: Ni",
                "  space_group: 225",
                "  lattice: [1, 1, 1, 90, 90, 90]",
            ]
        ),
        encoding="utf-8",
    )

    automator = BandWidthAutomator(config_path=str(config_path))
    automator.modified_data_path = h5_path
    automator.output_dir = tmp_path
    automator.base_name = "scan"
    automator.in_ang_path = tmp_path / "scan.ang"
    column_headers = [
        "IQ",
        "Fit",
        "110_band_width",
        "110_eff_deff_ratio",
        "110_psnr",
        "110_defficientlineIntensity",
        "110_efficientlineIntensity",
        "110_efficientDefficientRatio",
        "110_Bandwidth_efficientDefficientRatio",
    ]
    _write_min_ang(automator.in_ang_path, nrows=1, ncols_even=2, column_headers=column_headers)

    processed = [
        {
            "ind": 0,
            "bands": [
                {
                    "band_valid": True,
                    "psnr": 2.0,
                    "bandWidth": 1.0,
                    "efficientlineIntensity": 1.0,
                    "defficientlineIntensity": 0.5,
                    "band_profile": [1.0] * 8,
                    "central_line": [1.0, 2.0, 3.0, 4.0],
                },
                {
                    "band_valid": True,
                    "psnr": 2.0,
                    "band_profile": [9.0] * 8,
                    "central_line": [9.0, 9.0, 9.0, 9.0],
                },
            ],
        },
        {"ind": 1, "bands": []},
    ]

    automator.export_results(processed)

    with h5py.File(h5_path, "r") as handle:
        profile = handle[f"/{scan_name}/EBSD/Data/band_profile"][()]
        central = handle[f"/{scan_name}/EBSD/Data/central_line"][()]

    assert profile.shape == (n_pixels, 8)
    assert profile.dtype == np.float32
    assert np.allclose(profile[0], np.array([1.0] * 8, dtype=np.float32))
    assert np.isnan(profile[1]).all()
    assert central.shape == (n_pixels, 4)
    assert np.allclose(central[0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
