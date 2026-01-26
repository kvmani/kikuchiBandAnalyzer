"""Tests for map contrast updates."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.gui.main_window import MapCanvas


def test_map_canvas_updates_contrast_limits() -> None:
    """Ensure contrast limits update when data is refreshed."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = MapCanvas("Contrast Test")
    data = np.arange(4, dtype=np.float32).reshape(2, 2)
    canvas.update_data(data, vmin=0.0, vmax=3.0, reset_view=True)
    assert canvas._image is not None
    assert canvas._image.get_clim() == (0.0, 3.0)
    canvas.update_data(data, vmin=1.0, vmax=2.0, reset_view=False)
    assert canvas._image.get_clim() == (1.0, 2.0)
    app.quit()
