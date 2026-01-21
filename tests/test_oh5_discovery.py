"""Tests for OH5 field discovery."""

from __future__ import annotations

import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader


def test_oh5_discovery_scalar_maps() -> None:
    """Verify scalar map discovery and scalar sampling."""

    dataset = OH5ScanFileReader.from_path("testData/Test_Ti.oh5")
    try:
        assert "IQ" in dataset.catalog.scalars
        assert "CI" in dataset.catalog.scalars
        iq_map = dataset.get_map("IQ")
        ci_map = dataset.get_map("CI")
        assert iq_map.shape == (dataset.ny, dataset.nx)
        assert ci_map.shape == (dataset.ny, dataset.nx)
        x = dataset.nx // 2
        y = dataset.ny // 2
        iq_value = dataset.get_scalar("IQ", x, y)
        ci_value = dataset.get_scalar("CI", x, y)
        assert np.isfinite(iq_value)
        assert np.isfinite(ci_value)
    finally:
        dataset.close()
