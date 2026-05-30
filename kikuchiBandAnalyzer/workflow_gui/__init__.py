"""Workflow GUI for end-to-end Kikuchi band-width analysis."""

from __future__ import annotations

__all__ = ["WorkflowGuiMainWindow"]


def __getattr__(name: str) -> object:
    """Lazily expose the main window class.

    Parameters:
        name: Requested module attribute.

    Returns:
        Requested object.
    """

    if name == "WorkflowGuiMainWindow":
        from kikuchiBandAnalyzer.workflow_gui.main_window import WorkflowGuiMainWindow

        return WorkflowGuiMainWindow
    raise AttributeError(name)
