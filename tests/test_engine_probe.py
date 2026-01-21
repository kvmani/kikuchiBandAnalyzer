"""Tests for the comparison engine probe logic."""

from __future__ import annotations

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader


def test_engine_probe_fields() -> None:
    """Ensure probe results include IQ and CI at the default pixel."""

    dataset_a = OH5ScanFileReader.from_path("testData/Test_Ti.oh5")
    dataset_b = OH5ScanFileReader.from_path("testData/Test_Ti.oh5")
    config = {
        "default_map_field": "IQ",
        "compare_fields": {"scalars": ["IQ", "CI"], "patterns": []},
        "display": {"map_diff_mode": "delta", "pattern_diff_mode": "abs_delta"},
    }
    engine = ComparisonEngine(dataset_a, dataset_b, config)
    try:
        x, y = engine.default_probe_xy()
        probe = engine.probe_scalars(x, y, ["IQ", "CI"])
        assert probe.x == x
        assert probe.y == y
        assert "IQ" in probe.fields
        assert "CI" in probe.fields
        assert "A" in probe.fields["IQ"]
        assert "B" in probe.fields["IQ"]
        assert "Delta" in probe.fields["IQ"]
    finally:
        dataset_a.close()
        dataset_b.close()
