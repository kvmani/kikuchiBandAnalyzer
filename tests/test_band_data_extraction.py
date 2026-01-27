"""Tests for band profile extraction helpers used by downstream GUIs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.band_data import (
    extract_band_profile_payload,
    normalize_profile,
)
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader


def _write_band_oh5(path: Path) -> None:
    """Create a minimal OH5 file with band-profile datasets for testing."""

    nx = 2
    ny = 2
    n_pixels = nx * ny
    profile_len = 8
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Manufacturer", data="Test")
        handle.create_dataset("Version", data="1.0")
        scan = handle.create_group("Scan")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        header.create_dataset("nColumns", data=np.array([nx]))
        header.create_dataset("nRows", data=np.array([ny]))
        data = ebsd.create_group("Data")
        data.create_dataset("CI", data=np.zeros(n_pixels, dtype=np.float32))
        data.create_dataset("Pattern", data=np.zeros((n_pixels, 4, 4), dtype=np.float32))
        data.create_dataset("band_profile", data=np.tile(np.arange(profile_len, dtype=np.float32)[None, :], (n_pixels, 1)))
        data.create_dataset("central_line", data=np.tile(np.array([1, 2, 3, 4], dtype=np.float32)[None, :], (n_pixels, 1)))
        data.create_dataset("band_start_idx", data=np.full(n_pixels, 1, dtype=np.int32))
        data.create_dataset("central_peak_idx", data=np.full(n_pixels, 3, dtype=np.int32))
        data.create_dataset("band_end_idx", data=np.full(n_pixels, 6, dtype=np.int32))
        data.create_dataset("profile_length", data=np.full(n_pixels, profile_len, dtype=np.int32))
        valid = np.ones(n_pixels, dtype=np.int8)
        valid[0] = 0
        data.create_dataset("band_valid", data=valid)


def test_extract_band_profile_payload_reads_vectors(tmp_path) -> None:
    """Extract band profile vectors and indices for a pixel."""

    path = tmp_path / "band.oh5"
    _write_band_oh5(path)
    dataset = OH5ScanFileReader.from_path(path)
    try:
        assert "band_profile" in dataset.catalog.vectors
        payload = extract_band_profile_payload(dataset, x=1, y=1)
        assert payload.profile is not None
        assert payload.profile.shape == (8,)
        assert payload.band_start_idx == 1
        assert payload.central_peak_idx == 3
        assert payload.band_end_idx == 6
        assert payload.profile_length == 8
        assert payload.band_valid is True
    finally:
        dataset.close()


def test_extract_band_profile_payload_respects_band_valid_flag(tmp_path) -> None:
    """band_valid scalar overrides derived validity when present."""

    path = tmp_path / "band.oh5"
    _write_band_oh5(path)
    dataset = OH5ScanFileReader.from_path(path)
    try:
        payload = extract_band_profile_payload(dataset, x=0, y=0)
        assert payload.profile is not None
        assert payload.band_valid is False
    finally:
        dataset.close()


def test_normalize_profile_handles_zeros() -> None:
    """Normalization should avoid divide-by-zero for all-zero profiles."""

    profile = np.zeros(5, dtype=np.float32)
    normalized = normalize_profile(profile)
    assert np.allclose(normalized, profile)

