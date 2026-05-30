"""Capture screenshots of the single-pattern solver GUI for example configs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kikuchiBandAnalyzer.single_pattern_solver.gui import SinglePatternSolverWindow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the screenshot parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description="Capture single-pattern solver GUI screenshots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_pattern_solver"),
        help="Directory for screenshots.",
    )
    return parser


def main() -> None:
    """Capture CTF and DA single-pattern GUI screenshots."""

    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Arial", 9))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _capture(app, Path("configs/single_pattern_ctf.yml"), args.output_dir / "single_pattern_gui_ctf.png")
    _capture(app, Path("configs/single_pattern_da.yml"), args.output_dir / "single_pattern_gui_da.png")


def _capture(app: QtWidgets.QApplication, config_path: Path, output_path: Path) -> None:
    """Capture one GUI screenshot.

    Parameters:
        app: Qt application.
        config_path: Config path.
        output_path: Destination screenshot path.

    Returns:
        None.
    """

    window = SinglePatternSolverWindow(config_path)
    window.show()
    for _ in range(10):
        app.processEvents(QtCore.QEventLoop.AllEvents, 100)
    window.snapshot(output_path)
    window.close()


if __name__ == "__main__":
    main()
