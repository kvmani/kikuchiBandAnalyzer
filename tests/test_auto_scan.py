"""Tests for auto-scan raster stepping."""

from __future__ import annotations

from kikuchiBandAnalyzer.ebsd_compare.gui.auto_scan import RasterStepper


def test_raster_stepper_order() -> None:
    """Ensure raster stepping follows row-major order."""

    stepper = RasterStepper(width=3, height=2)
    coords = []
    while True:
        coord = stepper.next_coordinate()
        if coord is None:
            break
        coords.append(coord)
    assert coords == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
