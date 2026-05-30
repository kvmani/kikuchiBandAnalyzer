"""Tests for HKL/Oxford CTF plus pattern-folder reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.readers.ctf_reader import CtfPatternScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.readers.factory import open_scan_dataset


def _write_ctf(path: Path, iq_offset: float = 0.0) -> None:
    """Write a tiny CTF file for tests.

    Parameters:
        path: Output CTF path.
        iq_offset: Offset added to IQ values.

    Returns:
        None.
    """

    rows = [
        (1, 0.0, 0.0, 6, 0, 10, 20, 30, 0.1, 100 + iq_offset, 20),
        (1, 1.0, 0.0, 6, 0, 11, 21, 31, 0.2, 110 + iq_offset, 21),
        (1, 0.0, 1.0, 6, 0, 12, 22, 32, 0.3, 120 + iq_offset, 22),
        (1, 1.0, 1.0, 6, 0, 13, 23, 33, 0.4, 130 + iq_offset, 23),
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Channel Text File\n")
        handle.write("JobMode Grid\n")
        handle.write("XCells\t2\n")
        handle.write("YCells\t2\n")
        handle.write("XStep\t1.0\n")
        handle.write("YStep\t1.0\n")
        handle.write("Phases\t1\n")
        handle.write("Phase X Y Bands Error Euler1 Euler2 Euler3 MAD IQ BC\n")
        for row in rows:
            handle.write(" ".join(str(value) for value in row) + "\n")


def _write_patterns(pattern_dir: Path) -> None:
    """Write row-major grayscale pattern images.

    Parameters:
        pattern_dir: Directory to populate.

    Returns:
        None.
    """

    pattern_dir.mkdir()
    for index in range(4):
        data = np.full((3, 3), index + 1, dtype=np.uint8)
        Image.fromarray(data).save(pattern_dir / f"pattern_{index:03d}.png")


def test_ctf_reader_discovers_scalars_and_patterns(tmp_path) -> None:
    """Read CTF scalar maps and row-major external pattern images."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_dir = tmp_path / "patterns"
    _write_ctf(ctf_path)
    _write_patterns(pattern_dir)

    dataset = CtfPatternScanFileReader.from_path(ctf_path, pattern_dir=pattern_dir)
    try:
        assert dataset.nx == 2
        assert dataset.ny == 2
        assert "IQ" in dataset.catalog.scalars
        assert "Pattern" in dataset.catalog.patterns
        iq = dataset.get_map("IQ")
        assert np.allclose(iq, [[100, 110], [120, 130]])
        assert dataset.get_scalar("MAD", 1, 1) == 0.4
        pattern = dataset.get_pattern("Pattern", 1, 0)
        assert pattern.shape == (3, 3)
        assert np.all(pattern == 2)
    finally:
        dataset.close()


def test_ctf_reader_supports_template_pattern_names(tmp_path) -> None:
    """Resolve CTF pattern files using a coordinate filename template."""

    ctf_path = tmp_path / "scan.ctf"
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()
    _write_ctf(ctf_path)
    for y in range(2):
        for x in range(2):
            data = np.full((2, 2), y * 2 + x, dtype=np.uint8)
            Image.fromarray(data).save(pattern_dir / f"p_x{x}_y{y}.png")

    dataset = CtfPatternScanFileReader.from_path(
        ctf_path,
        pattern_dir=pattern_dir,
        pattern_template="p_x{x}_y{y}.png",
    )
    try:
        pattern = dataset.get_pattern("Pattern", 0, 1)
        assert np.all(pattern == 2)
    finally:
        dataset.close()


def test_factory_opens_ctf_and_engine_compares_maps(tmp_path) -> None:
    """Open CTF scans via the format factory and compare scalar maps."""

    ctf_a = tmp_path / "scan_a.ctf"
    ctf_b = tmp_path / "scan_b.ctf"
    _write_ctf(ctf_a, iq_offset=0.0)
    _write_ctf(ctf_b, iq_offset=10.0)

    dataset_a = open_scan_dataset(ctf_a)
    dataset_b = open_scan_dataset(ctf_b)
    try:
        engine = ComparisonEngine(dataset_a, dataset_b, config={})
        maps = engine.map_triplet("IQ", "delta")
        assert np.allclose(maps["D"], -10.0)
    finally:
        dataset_a.close()
        dataset_b.close()
