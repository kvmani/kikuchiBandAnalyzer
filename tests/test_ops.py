"""Tests for comparison ops."""

from __future__ import annotations

import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.compare import ops


def test_ratio_handles_zeros() -> None:
    """Ensure ratio handles zeros and infinities safely."""

    a = np.array([1.0, 0.0, np.inf])
    b = np.array([0.0, 0.0, 2.0])
    result = ops.ratio(a, b)
    assert result.shape == a.shape
    assert np.isnan(result[1]) or np.isfinite(result[1])
    assert np.isnan(result[2]) or np.isfinite(result[2])
