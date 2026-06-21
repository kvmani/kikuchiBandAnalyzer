"""Tests for scientific scalar-map rendering settings."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from kikuchiBandAnalyzer.ebsd_compare.map_display import MapDisplaySettings


def test_linear_auto_limits_and_yaml_round_trip() -> None:
    """Compute percentile limits and preserve settings through mappings."""

    settings = MapDisplaySettings(percentile_low=0.0, percentile_high=100.0)
    norm, cmap = settings.render_parameters(np.array([0.0, 1.0, 2.0]))
    assert isinstance(norm, Normalize)
    assert norm.vmin == 0.0
    assert norm.vmax == 2.0
    assert cmap.name == "viridis"
    restored = MapDisplaySettings.from_mapping(settings.to_mapping())
    assert restored.to_mapping() == settings.to_mapping()


def test_log_requires_positive_values() -> None:
    """Reject logarithmic scaling when no positive values exist."""

    settings = MapDisplaySettings(scale="log")
    with pytest.raises(ValueError, match="positive"):
        settings.render_parameters(np.array([-2.0, 0.0, np.nan]))
    norm, _ = settings.render_parameters(np.array([0.1, 1.0, 10.0]))
    assert isinstance(norm, LogNorm)


def test_symlog_symmetric_limits() -> None:
    """Use equal signed bounds for strain/stress style maps."""

    settings = MapDisplaySettings(
        scale="symlog",
        symmetric=True,
        percentile_low=0.0,
        percentile_high=100.0,
        linthresh=0.01,
        colormap="coolwarm",
    )
    norm, _ = settings.render_parameters(np.array([-2.0, 0.5, 1.0]))
    assert isinstance(norm, SymLogNorm)
    assert norm.vmin == -2.0
    assert norm.vmax == 2.0
