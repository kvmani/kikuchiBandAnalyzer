"""Main window for the EBSD scan comparator GUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging, load_yaml_config

matplotlib.use("QtAgg")


class MapCanvas(FigureCanvas):
    """Matplotlib canvas wrapper for displaying 2D maps.

    Parameters:
        title: Title for the plot.
    """

    def __init__(self, title: str) -> None:
        """Initialize the canvas.

        Parameters:
            title: Title for the plot.
        """

        self._figure = Figure(figsize=(4, 4))
        self._axes = self._figure.add_subplot(111)
        super().__init__(self._figure)
        self._title = title
        self._image = None
        self._axes.set_title(title)
        self._axes.set_xticks([])
        self._axes.set_yticks([])

    def update_data(self, data: np.ndarray, cmap: str = "viridis") -> None:
        """Update the displayed data.

        Parameters:
            data: 2D array to display.
            cmap: Matplotlib colormap name.
        """

        self._axes.clear()
        self._axes.set_title(self._title)
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._image = self._axes.imshow(data, cmap=cmap)
        self._figure.tight_layout()
        self.draw_idle()

    def connect_click(self, handler: QtCore.Slot) -> None:
        """Connect a click handler to the canvas.

        Parameters:
            handler: Matplotlib event handler.
        """

        self.mpl_connect("button_press_event", handler)


class EbsdCompareMainWindow(QtWidgets.QMainWindow):
    """Main window for comparing two EBSD scans."""

    def __init__(self, config_path: Path) -> None:
        """Initialize the main window.

        Parameters:
            config_path: Path to the YAML configuration.
        """

        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._config_path = config_path
        self._config = load_yaml_config(config_path).get("ebsd_compare", {})
        self._scan_a = None
        self._scan_b = None
        self._engine: Optional[ComparisonEngine] = None
        self._pattern_field: Optional[str] = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the GUI layout and widgets."""

        self.setWindowTitle("EBSD Scan Comparator")
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)

        file_layout = QtWidgets.QGridLayout()
        self._file_a_edit = QtWidgets.QLineEdit()
        self._file_b_edit = QtWidgets.QLineEdit()
        self._file_a_button = QtWidgets.QPushButton("Browse A")
        self._file_b_button = QtWidgets.QPushButton("Browse B")
        self._load_button = QtWidgets.QPushButton("Load Scans")
        self._file_a_button.clicked.connect(self._browse_file_a)
        self._file_b_button.clicked.connect(self._browse_file_b)
        self._load_button.clicked.connect(self._load_from_inputs)
        file_layout.addWidget(QtWidgets.QLabel("Scan A"), 0, 0)
        file_layout.addWidget(self._file_a_edit, 0, 1)
        file_layout.addWidget(self._file_a_button, 0, 2)
        file_layout.addWidget(QtWidgets.QLabel("Scan B"), 1, 0)
        file_layout.addWidget(self._file_b_edit, 1, 1)
        file_layout.addWidget(self._file_b_button, 1, 2)
        file_layout.addWidget(self._load_button, 2, 1)

        layout.addLayout(file_layout)

        map_control_layout = QtWidgets.QHBoxLayout()
        map_control_layout.addWidget(QtWidgets.QLabel("Map Field"))
        self._map_field_combo = QtWidgets.QComboBox()
        self._map_field_combo.currentTextChanged.connect(self._update_maps)
        map_control_layout.addWidget(self._map_field_combo)
        self._screenshot_button = QtWidgets.QPushButton("Save Screenshot")
        self._screenshot_button.clicked.connect(self._on_save_screenshot)
        map_control_layout.addWidget(self._screenshot_button)
        layout.addLayout(map_control_layout)

        map_layout = QtWidgets.QHBoxLayout()
        self._map_canvas_a = MapCanvas("Scan A")
        self._map_canvas_b = MapCanvas("Scan B")
        self._map_canvas_d = MapCanvas("Δ/Ratio")
        for canvas in (self._map_canvas_a, self._map_canvas_b, self._map_canvas_d):
            canvas.connect_click(self._on_map_click)
        map_layout.addWidget(self._map_canvas_a)
        map_layout.addWidget(self._map_canvas_b)
        map_layout.addWidget(self._map_canvas_d)
        layout.addLayout(map_layout)

        probe_layout = QtWidgets.QHBoxLayout()
        self._probe_table = QtWidgets.QTableWidget()
        self._probe_table.setColumnCount(5)
        self._probe_table.setHorizontalHeaderLabels([
            "Field",
            "A",
            "B",
            "Δ",
            "Ratio",
        ])
        self._probe_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        probe_layout.addWidget(self._probe_table)

        pattern_group = QtWidgets.QGroupBox("Pattern Comparison")
        pattern_layout = QtWidgets.QHBoxLayout(pattern_group)
        self._pattern_canvas_a = MapCanvas("Pattern A")
        self._pattern_canvas_b = MapCanvas("Pattern B")
        self._pattern_canvas_d = MapCanvas("Pattern Δ/Ratio")
        pattern_layout.addWidget(self._pattern_canvas_a)
        pattern_layout.addWidget(self._pattern_canvas_b)
        pattern_layout.addWidget(self._pattern_canvas_d)
        probe_layout.addWidget(pattern_group)
        self._pattern_group = pattern_group
        layout.addLayout(probe_layout)

        self.setCentralWidget(central_widget)

    def load_scans(self, path_a: Path, path_b: Path) -> None:
        """Load scan datasets and update the GUI.

        Parameters:
            path_a: Path to scan A.
            path_b: Path to scan B.
        """

        self._logger.info("Loading scans: %s and %s", path_a, path_b)
        field_aliases = self._config.get("field_aliases", {})
        self._scan_a = OH5ScanFileReader.from_path(path_a, field_aliases=field_aliases)
        self._scan_b = OH5ScanFileReader.from_path(path_b, field_aliases=field_aliases)
        self._engine = ComparisonEngine(self._scan_a, self._scan_b, self._config)
        self._populate_map_fields()
        self._select_pattern_field()
        self._update_maps()
        x, y = self._engine.default_probe_xy()
        self._update_probe(x, y)

    def save_screenshot(self, output_path: Path) -> None:
        """Save a screenshot of the current window.

        Parameters:
            output_path: Path for the PNG screenshot.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self.grab()
        pixmap.save(str(output_path), "PNG")
        self._logger.info("Saved screenshot to %s", output_path)

    def _browse_file_a(self) -> None:
        """Open a file dialog to choose scan A."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Scan A")
        if path:
            self._file_a_edit.setText(path)

    def _browse_file_b(self) -> None:
        """Open a file dialog to choose scan B."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Scan B")
        if path:
            self._file_b_edit.setText(path)

    def _load_from_inputs(self) -> None:
        """Load scans based on current input paths."""

        if not self._file_a_edit.text() or not self._file_b_edit.text():
            QtWidgets.QMessageBox.warning(self, "Missing files", "Select both scans.")
            return
        self.load_scans(Path(self._file_a_edit.text()), Path(self._file_b_edit.text()))

    def _populate_map_fields(self) -> None:
        """Populate the map field dropdown."""

        if self._engine is None:
            return
        self._map_field_combo.blockSignals(True)
        self._map_field_combo.clear()
        fields = self._engine.available_scalar_fields()
        self._map_field_combo.addItems(fields)
        default_field = self._engine.default_map_field()
        index = self._map_field_combo.findText(default_field)
        if index >= 0:
            self._map_field_combo.setCurrentIndex(index)
        self._map_field_combo.blockSignals(False)

    def _select_pattern_field(self) -> None:
        """Select the first available pattern field from config."""

        self._pattern_field = None
        pattern_config: List[str] = (
            self._config.get("compare_fields", {}).get("patterns", [])
        )
        if not pattern_config or self._engine is None:
            self._pattern_group.setVisible(False)
            return
        for field in pattern_config:
            if (
                field in self._scan_a.catalog.patterns
                and field in self._scan_b.catalog.patterns
            ):
                self._pattern_field = field
                self._pattern_group.setVisible(True)
                return
        self._pattern_group.setVisible(False)

    def _update_maps(self) -> None:
        """Update the map panels based on current selection."""

        if self._engine is None:
            return
        field_name = self._map_field_combo.currentText()
        if not field_name:
            return
        mode = self._config.get("display", {}).get("map_diff_mode", "delta")
        maps = self._engine.map_triplet(field_name, mode)
        self._map_canvas_a.update_data(maps["A"])
        self._map_canvas_b.update_data(maps["B"])
        self._map_canvas_d.update_data(maps["D"], cmap="coolwarm")

    def _on_map_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Handle map clicks to update the probe.

        Parameters:
            event: Matplotlib mouse event.
        """

        if event.xdata is None or event.ydata is None or self._engine is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or y < 0 or x >= self._scan_a.nx or y >= self._scan_a.ny:
            return
        self._update_probe(x, y)

    def _update_probe(self, x: int, y: int) -> None:
        """Update the probe table and pattern panels.

        Parameters:
            x: Column index.
            y: Row index.
        """

        if self._engine is None:
            return
        scalar_fields = (
            self._config.get("compare_fields", {}).get("scalars", [])
        )
        if not scalar_fields:
            scalar_fields = self._engine.available_scalar_fields()
        probe = self._engine.probe_scalars(x, y, scalar_fields)
        self._probe_table.setRowCount(len(probe.fields))
        for row, (field, values) in enumerate(probe.fields.items()):
            self._probe_table.setItem(row, 0, QtWidgets.QTableWidgetItem(field))
            self._probe_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{values['A']:.4g}")
            )
            self._probe_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(f"{values['B']:.4g}")
            )
            self._probe_table.setItem(
                row, 3, QtWidgets.QTableWidgetItem(f"{values['Delta']:.4g}")
            )
            ratio_value = values["Ratio"]
            ratio_text = "nan" if np.isnan(ratio_value) else f"{ratio_value:.4g}"
            self._probe_table.setItem(
                row, 4, QtWidgets.QTableWidgetItem(ratio_text)
            )
        self._probe_table.resizeRowsToContents()
        self._update_patterns(x, y)

    def _update_patterns(self, x: int, y: int) -> None:
        """Update the pattern panels for the given coordinate.

        Parameters:
            x: Column index.
            y: Row index.
        """

        if self._engine is None or self._pattern_field is None:
            return
        mode = self._config.get("display", {}).get("pattern_diff_mode", "abs_delta")
        patterns = self._engine.probe_patterns(x, y, [self._pattern_field], mode)
        pattern_triplet = patterns.get(self._pattern_field)
        if not pattern_triplet or pattern_triplet["A"] is None:
            return
        self._pattern_canvas_a.update_data(pattern_triplet["A"], cmap="gray")
        self._pattern_canvas_b.update_data(pattern_triplet["B"], cmap="gray")
        self._pattern_canvas_d.update_data(pattern_triplet["D"], cmap="gray")

    def _on_save_screenshot(self) -> None:
        """Prompt for screenshot output and save the current window."""

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Screenshot", filter="PNG Files (*.png)"
        )
        if path:
            self.save_screenshot(Path(path))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description="EBSD Scan Comparator GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ebsd_compare_config.yml"),
        help="Path to EBSD compare YAML config.",
    )
    parser.add_argument(
        "--scan-a",
        type=Path,
        default=None,
        help="Path to scan A OH5 file.",
    )
    parser.add_argument(
        "--scan-b",
        type=Path,
        default=None,
        help="Path to scan B OH5 file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    """Run the EBSD compare GUI."""

    args = build_arg_parser().parse_args()
    configure_logging(args.debug)
    app = QtWidgets.QApplication([])
    window = EbsdCompareMainWindow(args.config)
    window.show()
    if args.scan_a and args.scan_b:
        window.load_scans(args.scan_a, args.scan_b)
    app.exec()


if __name__ == "__main__":
    main()
