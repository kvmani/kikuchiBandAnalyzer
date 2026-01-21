"""Tests for noisy OH5 generation."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.noise import NoisyOh5Generator


def test_noisy_generator_changes_iq_ci(tmp_path: Path) -> None:
    """Ensure IQ and CI arrays are modified in the noisy output."""

    input_path = Path("testData/Test_Ti.oh5")
    output_path = tmp_path / "Test_Ti_noisy.oh5"
    logger = logging.getLogger("test_noisy_generator")
    generator = NoisyOh5Generator(
        input_path=input_path,
        output_path=output_path,
        sigma_map={"IQ": 0.05, "CI": 0.02},
        seed=123,
        logger=logger,
    )
    generator.run()
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "r") as target:
        source_scan = source["Test3/EBSD/Data"]
        target_scan = target["Test3/EBSD/Data"]
        for field in ("IQ", "CI"):
            source_data = source_scan[field][()]
            target_data = target_scan[field][()]
            assert source_data.shape == target_data.shape
            assert not np.array_equal(source_data, target_data)
