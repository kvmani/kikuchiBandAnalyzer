"""Tests for pattern extraction and comparison."""

from __future__ import annotations

import logging

import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.simulated import SimulatedScanFactory


def test_probe_patterns_nonzero() -> None:
    """Ensure extracted patterns are non-empty and non-zero."""

    config = {
        "default_map_field": "IQ",
        "compare_fields": {"scalars": ["IQ"], "patterns": ["Pattern"]},
        "display": {"map_diff_mode": "delta", "pattern_diff_mode": "abs_delta"},
        "alignment": {"enabled": False},
    }
    factory = SimulatedScanFactory.from_config(
        {
            "nx": 6,
            "ny": 5,
            "nx_b": 6,
            "ny_b": 5,
            "include_patterns": True,
            "pattern_height": 12,
            "pattern_width": 12,
            "seed": 123,
        },
        logger=logging.getLogger(__name__),
    )
    scan_a, scan_b = factory.create_pair()
    try:
        engine = ComparisonEngine(scan_a, scan_b, config)
        patterns = engine.probe_patterns(2, 3, ["Pattern"], "abs_delta")
        triplet = patterns["Pattern"]
        assert triplet["A"] is not None
        assert triplet["B"] is not None
        assert triplet["D"] is not None
        assert triplet["A"].shape == (12, 12)
        assert not np.allclose(triplet["A"], 0)
        assert not np.allclose(triplet["B"], 0)
        assert not np.allclose(triplet["D"], 0)
    finally:
        scan_a.close()
        scan_b.close()
