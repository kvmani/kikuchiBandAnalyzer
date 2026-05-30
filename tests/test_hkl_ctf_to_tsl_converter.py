"""Tests for the limited FCC HKL CTF to TSL conversion utilities."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from kikuchiBandAnalyzer.io.hkl_ctf_to_tsl import (
    build_tsl_fields,
    load_mapped_patterns,
    parse_ctf_file,
)


def _write_ctf(path: Path) -> None:
    """Write a tiny HKL/Oxford-style CTF file.

    Parameters:
        path: Destination CTF path.

    Returns:
        None.
    """

    path.write_text(
        "\n".join(
            [
                "Channel Text File",
                "XCells\t2",
                "YCells\t2",
                "XStep\t2",
                "YStep\t2",
                "Phases\t1",
                "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS",
                "2 0 0 8 4 10 20 30 0.10 100 120",
                "2 2 0 8 4 11 21 31 0.20 110 121",
                "0 0 2 0 0 0 0 0 0.00 80 90",
                "2 2 2 8 4 12 22 32 0.40 130 122",
            ]
        ),
        encoding="utf-8",
    )


def _write_patterns(path: Path) -> None:
    """Write coordinate-named test pattern images.

    Parameters:
        path: Destination pattern folder.

    Returns:
        None.
    """

    path.mkdir()
    for y in range(2):
        for x in range(2):
            data = np.full((4, 5), y * 2 + x + 1, dtype=np.uint8)
            Image.fromarray(data).save(path / f"{x}_{y}.tiff")
    Image.fromarray(np.full((4, 5), 99, dtype=np.uint8)).save(path / "2_2.tiff")


def test_ctf_parser_and_pattern_mapping_use_xy_filename_order(tmp_path: Path) -> None:
    """Parse a CTF file and map patterns with the ``{x}_{y}.tiff`` convention."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_dir = tmp_path / "patterns"
    _write_ctf(ctf_path)
    _write_patterns(pattern_dir)

    ctf = parse_ctf_file(ctf_path)
    patterns = load_mapped_patterns(ctf, pattern_dir)

    assert ctf.nx == 2
    assert ctf.ny == 2
    assert ctf.x_step == 2.0
    assert ctf.y_step == 2.0
    assert patterns.shape == (4, 4, 5)
    assert int(patterns[0, 0, 0]) == 257
    assert int(patterns[3, 0, 0]) == 1028


def test_build_tsl_fields_populates_da_compatible_names(tmp_path: Path) -> None:
    """Create DA-compatible scalar fields from parsed CTF values and patterns."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_dir = tmp_path / "patterns"
    _write_ctf(ctf_path)
    _write_patterns(pattern_dir)

    ctf = parse_ctf_file(ctf_path)
    patterns = load_mapped_patterns(ctf, pattern_dir)
    fields = build_tsl_fields(ctf, patterns)

    assert set(fields).issuperset(
        {
            "Phi1",
            "Phi",
            "Phi2",
            "X Position",
            "Y Position",
            "IQ",
            "CI",
            "Phase",
            "SEM Signal",
            "Fit",
            "Pattern",
        }
    )
    assert np.allclose(fields["CI"], [0.75, 0.5, 1.0, 0.0])
    assert fields["Phase"].tolist() == [0, 0, -1, 0]
    assert fields["Valid"].tolist() == [0, 0, 2, 0]
    assert np.allclose(fields["Phi1"][:2], np.deg2rad([10, 11]))


def test_generated_fixture_h5_has_expected_shapes_when_present() -> None:
    """Verify the generated HKL fixture HDF5 file when local fixture outputs exist."""

    h5_path = Path("testData/hkl_ctf_test_data/converted_tsl/Subset_HKL_TSL.h5")
    if not h5_path.exists():
        return

    with h5py.File(h5_path, "r") as handle:
        scan = next(key for key in handle.keys() if key not in {"Manufacturer", "Version"})
        header = handle[f"{scan}/EBSD/Header"]
        data = handle[f"{scan}/EBSD/Data"]
        assert int(header["nColumns"][()][0]) == 15
        assert int(header["nRows"][()][0]) == 21
        assert data["Pattern"].shape == (315, 240, 320)
        assert data["Phi1"].shape == (315,)
        assert data["CI"].shape == (315,)
