"""Capture workflow GUI snapshots for bundled CTF and DA reference inputs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kikuchiBandAnalyzer.workflow_gui.main_window import WorkflowGuiMainWindow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the snapshot command-line parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Capture workflow GUI snapshots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/workflow_gui_snapshots"),
        help="Directory for GUI screenshots and prepared working files.",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Use Qt offscreen rendering. This can omit system fonts on some Windows setups.",
    )
    return parser


def main() -> None:
    """Capture snapshots for the CTF fixture and DA reference dataset.

    Returns:
        None.
    """

    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Arial", 9))
    app.setStyleSheet("* { font-family: Arial; }")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    capture_ctf(root, app)
    capture_da(root, app)


def capture_ctf(root: Path, app: QtWidgets.QApplication) -> None:
    """Prepare the CTF fixture in the GUI and save a snapshot.

    Parameters:
        root: Snapshot root directory.
        app: Active Qt application.

    Returns:
        None.
    """

    window = WorkflowGuiMainWindow(
        input_mode="ctf",
        source_path=Path("testData/hkl_ctf_test_data/Subset.ctf"),
        pattern_dir=Path("testData/hkl_ctf_test_data/Binned_2x2"),
        output_dir=root / "ctf",
    )
    window.show()
    window.prepare_inputs()
    _process_events(app)
    window.snapshot(root / "workflow_gui_ctf_prepared.png")
    window.close()


def capture_da(root: Path, app: QtWidgets.QApplication) -> None:
    """Prepare the DA reference dataset in the GUI and save a snapshot.

    Parameters:
        root: Snapshot root directory.
        app: Active Qt application.

    Returns:
        None.
    """

    window = WorkflowGuiMainWindow(
        input_mode="tsl",
        source_path=Path("testData/DA.oh5"),
        ang_path=Path("testData/DA.ang"),
        output_dir=root / "da",
    )
    window.show()
    window.prepare_inputs()
    _process_events(app)
    window.snapshot(root / "workflow_gui_DA_prepared.png")
    window.close()


def _process_events(app: QtWidgets.QApplication) -> None:
    """Process Qt events long enough for canvases to render.

    Parameters:
        app: Active Qt application.

    Returns:
        None.
    """

    for _ in range(10):
        app.processEvents(QtCore.QEventLoop.AllEvents, 100)


if __name__ == "__main__":
    main()
