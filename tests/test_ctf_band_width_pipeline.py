"""Tests for CTF-backed band-width pipeline preparation and export."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from KikuchiBandWidthAutomator import BandWidthAutomator


def _write_ctf(path: Path) -> None:
    """Write a tiny CTF file for automator tests.

    Parameters:
        path: Output CTF path.

    Returns:
        None.
    """

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Channel Text File\n")
        handle.write("XCells\t2\n")
        handle.write("YCells\t2\n")
        handle.write("Phase X Y Bands Error Euler1 Euler2 Euler3 MAD IQ BC\n")
        for index, (x, y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            handle.write(
                f"1 {x} {y} 6 0 {10 + index} {20 + index} {30 + index} "
                f"{0.1 + index} {100 + index} {200 + index}\n"
            )


def _write_patterns(pattern_dir: Path) -> None:
    """Write row-major pattern images.

    Parameters:
        pattern_dir: Directory to populate.

    Returns:
        None.
    """

    pattern_dir.mkdir()
    for index in range(4):
        Image.fromarray(np.full((5, 5), index + 1, dtype=np.uint8)).save(
            pattern_dir / f"pattern_{index:03d}.png"
        )


def _write_annotations(path: Path) -> None:
    """Write minimal per-pixel annotation entries.

    Parameters:
        path: Output JSON path.

    Returns:
        None.
    """

    payload = [{"points": [], "pattern_path": f"pattern_{index:03d}.png"} for index in range(4)]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_config(
    path: Path,
    ctf_path: Path,
    pattern_dir: Path,
    annotations: Path | None = None,
) -> None:
    """Write a CTF band-width config.

    Parameters:
        path: Output YAML path.
        ctf_path: CTF file path.
        pattern_dir: Pattern folder path.
        annotations: Optional band-line annotation JSON path.

    Returns:
        None.
    """

    lines = [
        f"ctf_file_path: {ctf_path}",
        f"pattern_folder: {pattern_dir}",
        "desired_hkl_ref_width: 1.0",
        "elastic_modulus: 2.0",
        "desired_hkl: 110",
        "rectWidth: 2",
        "min_psnr: 1.0",
        "orientation_source: acquisition",
        "hkl_list:",
        "  - [1, 1, 0]",
        "phase_list:",
        "  name: Ni",
        "  space_group: 225",
        "  lattice: [1, 1, 1, 90, 90, 90]",
        "ctf_detector:",
        "  pc: [0.5, 0.5, 0.5]",
        "  sample_tilt: 70.0",
        "  tilt: 0.0",
        "  azimuthal: 0.0",
        "debug: false",
    ]
    if annotations is not None:
        lines.insert(2, f"band_annotation_json_path: {annotations}")
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def test_ctf_automator_runs_with_precomputed_annotations(tmp_path, monkeypatch) -> None:
    """Run the CTF automator path through HDF5/CSV export."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_dir = tmp_path / "patterns"
    annotations = tmp_path / "annotations.json"
    config_path = tmp_path / "config.yml"
    _write_ctf(ctf_path)
    _write_patterns(pattern_dir)
    _write_annotations(annotations)
    _write_config(config_path, ctf_path, pattern_dir, annotations)

    def _fake_detect_bands(self):
        return [
            {
                "bandWidth": 1.5,
                "psnr": 3.0,
                "band_valid": True,
                "efficientlineIntensity": 4.0,
                "defficientlineIntensity": 2.0,
                "band_profile": [1.0] * 8,
                "central_line": [0.0, 1.0, 2.0, 3.0],
                "band_start_idx": 1,
                "central_peak_idx": 4,
                "band_end_idx": 6,
            }
        ]

    monkeypatch.setattr("kikuchiBandWidthDetector.BandDetector.detect_bands", _fake_detect_bands)

    automator = BandWidthAutomator(config_path=str(config_path))
    automator.run()

    output_path = tmp_path / "scan_modified.h5"
    assert output_path.exists()
    assert (tmp_path / "scan_bandOutputData.csv").exists()
    assert (tmp_path / "scan_filtered_band_data.csv").exists()
    assert (tmp_path / "scan_modified.ang").exists()
    assert (tmp_path / "scan_modified.oh5").exists()

    with h5py.File(output_path, "r") as handle:
        data_root = "/scan/EBSD/Data"
        assert handle[f"{data_root}/Pattern"].shape == (4, 5, 5)
        assert handle[f"{data_root}/CI"].shape == (4,)
        assert handle[f"{data_root}/IQ"].shape == (4,)
        assert np.allclose(handle[f"{data_root}/Band_Width"][()], 1.5)
        assert np.allclose(handle[f"{data_root}/psnr"][()], 3.0)
        assert handle[f"{data_root}/band_profile"].shape == (4, 8)
        assert handle[f"{data_root}/band_valid"][()].tolist() == [1, 1, 1, 1]
    ang_text = (tmp_path / "scan_modified.ang").read_text(encoding="utf-8")
    assert "PRIAS Bottom Strip" in ang_text
    assert "HKL/Oxford CTF" in ang_text


def test_ctf_automator_simulates_annotations_from_euler_angles(tmp_path, monkeypatch) -> None:
    """Run the direct CTF Euler-angle simulation path without annotation JSON."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_dir = tmp_path / "patterns"
    config_path = tmp_path / "config.yml"
    _write_ctf(ctf_path)
    _write_patterns(pattern_dir)
    _write_config(config_path, ctf_path, pattern_dir, annotations=None)

    class _FakeSimulation:
        """Fake geometrical simulation that returns one annotation per pixel."""

        phase = None

        def as_markers(self, kikuchi_line_labels: bool, desired_hkl: str):
            """Return deterministic grouped annotations.

            Parameters:
                kikuchi_line_labels: Whether line labels were requested.
                desired_hkl: Desired HKL string.

            Returns:
                Marker list and grouped annotations.
            """

            assert kikuchi_line_labels is True
            assert desired_hkl == "110"
            grouped = []
            for idx in range(4):
                row, col = divmod(idx, 2)
                grouped.append(
                    {
                        "x,y": [row, col],
                        "ind": idx,
                        "points": [
                            {
                                "hkl": "1 1 0",
                                "hkl_group": "(1, 1, 0)",
                                "central_line": [0.0, 1.0, 4.0, 1.0],
                                "line_mid_xy": [2.0, 1.0],
                                "line_dist": 0.0,
                            }
                        ],
                    }
                )
            return [], grouped

    class _FakeSimulator:
        """Fake Kikuchi simulator that validates detector/rotation inputs."""

        def __init__(self, reflectors):
            """Store reflectors for API compatibility.

            Parameters:
                reflectors: Reciprocal lattice reflectors.
            """

            self.reflectors = reflectors

        def on_detector(self, detector, rotations):
            """Validate direct CTF simulation inputs and return fake output.

            Parameters:
                detector: EBSDDetector instance.
                rotations: Orix rotation grid.

            Returns:
                Fake simulation object.
            """

            assert detector.shape == (5, 5)
            assert rotations.shape == (2, 2)
            return _FakeSimulation()

    def _fake_detect_bands(self):
        return [
            {
                "bandWidth": 2.0,
                "psnr": 4.0,
                "band_valid": True,
                "efficientlineIntensity": 8.0,
                "defficientlineIntensity": 2.0,
                "band_profile": [2.0] * 8,
                "central_line": [0.0, 1.0, 4.0, 1.0],
                "band_start_idx": 1,
                "central_peak_idx": 4,
                "band_end_idx": 6,
            }
        ]

    monkeypatch.setattr("KikuchiBandWidthAutomator.CustomKikuchiPatternSimulator", _FakeSimulator)
    monkeypatch.setattr("kikuchiBandWidthDetector.BandDetector.detect_bands", _fake_detect_bands)

    automator = BandWidthAutomator(config_path=str(config_path))
    automator.run()

    output_path = tmp_path / "scan_modified.h5"
    with h5py.File(output_path, "r") as handle:
        data_root = "/scan/EBSD/Data"
        assert np.allclose(handle[f"{data_root}/Band_Width"][()], 2.0)
        assert np.allclose(handle[f"{data_root}/psnr"][()], 4.0)
