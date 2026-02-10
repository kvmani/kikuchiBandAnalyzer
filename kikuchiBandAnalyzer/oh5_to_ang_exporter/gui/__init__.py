"""GUI package for OH5-to-ANG exporter.

This module intentionally avoids importing ``main_window`` at import time so
``python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.gui.main_window`` does not
trigger duplicate-module warnings from ``runpy``.
"""

__all__ = ["Oh5ToAngExporterMainWindow"]


def __getattr__(name: str):
    """Lazily resolve public GUI exports.

    Parameters:
        name: Attribute name requested by importers.

    Returns:
        Exported object for supported names.

    Raises:
        AttributeError: If the name is not provided by this package.
    """

    if name == "Oh5ToAngExporterMainWindow":
        from .main_window import Oh5ToAngExporterMainWindow

        return Oh5ToAngExporterMainWindow
    raise AttributeError(name)
