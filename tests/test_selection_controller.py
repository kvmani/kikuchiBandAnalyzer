"""Tests for selection coordination logic."""

from __future__ import annotations

import logging

from kikuchiBandAnalyzer.ebsd_compare.gui.selection import SelectionController


def test_selection_controller_updates_state() -> None:
    """Ensure valid selections update state and callbacks."""

    selections = []
    errors = []

    def on_selection(state) -> None:
        selections.append(state)

    def on_error(message: str) -> None:
        errors.append(message)

    controller = SelectionController(on_selection, on_error, logger=logging.getLogger(__name__))
    controller.set_bounds(3, 2)
    assert controller.set_selected_pixel(1, 1, source="test")
    assert controller.selected() is not None
    assert selections[0].x == 1
    assert selections[0].y == 1
    assert selections[0].source == "test"
    assert not controller.set_selected_pixel(5, 0, source="test")
    assert errors
