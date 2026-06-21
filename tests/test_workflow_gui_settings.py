"""Tests for unified workflow GUI orientation and map defaults."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from kikuchiBandAnalyzer.workflow_gui.main_window import WorkflowGuiMainWindow


def test_workflow_gui_defaults_to_indexed_and_signed_symlog() -> None:
    """Use live indexing by default and symlog for signed scientific maps."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = WorkflowGuiMainWindow()
    assert window._orientation_source_combo.currentData() == "indexed"
    assert window._map_display_settings["strain"].scale == "symlog"
    assert window._map_display_settings["strain"].symmetric is True
    assert window._map_display_settings["stress"].scale == "symlog"
    assert window._map_display_settings["Band_Width"].scale == "linear"
    window.close()
    app.processEvents()
