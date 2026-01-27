"""Tests for band profile index metadata emitted by detect_edges."""

from __future__ import annotations

import numpy as np

from strategies import RectangularAreaBandDetector


def test_detect_edges_emits_index_metadata() -> None:
    """detect_edges should emit snake_case index metadata and profile_length."""

    detector = RectangularAreaBandDetector(
        image=np.zeros((8, 8), dtype=np.float32),
        central_line=[0.0, 0.0, 1.0, 1.0],
        config={"smoothing_sigma": 1.0, "min_psnr": 1.0, "rectWidth": 20},
        hkl="110",
    )
    profile = np.full(50, 5.0, dtype=np.float32)
    profile[10] = 1.0
    profile[25] = 20.0
    profile[40] = 2.0
    result = detector.detect_edges(profile)
    assert "band_start_idx" in result
    assert "band_end_idx" in result
    assert "central_peak_idx" in result
    assert "profile_length" in result
    assert result["profile_length"] == 50
    assert result["band_start_idx"] != -1
    assert result["band_end_idx"] != -1
    assert result["central_peak_idx"] != -1
    assert 0 <= result["band_start_idx"] < 50
    assert 0 <= result["central_peak_idx"] < 50
    assert 0 <= result["band_end_idx"] < 50

