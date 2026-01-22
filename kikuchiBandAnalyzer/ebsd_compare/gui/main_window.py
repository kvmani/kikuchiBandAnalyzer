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
from kikuchiBandAnalyzer.ebsd_compare.gui.logging_widget import (
    GuiLogHandler,
    LogEmitter,
    LogViewer,
)
from kikuchiBandAnalyzer.ebsd_compare.gui.registration_window import RegistrationDialog
from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    AlignmentResult,
    alignment_from_config,
)
from kikuchiBandAnalyzer.ebsd_compare.simulated import SimulatedScanFactory
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
        self._alignment_result: Optional[AlignmentResult] = None
        self._log_emitter: Optional[LogEmitter] = None
        self._log_handler: Optional[GuiLogHandler] = None
        self._log_viewer: Optional[LogViewer] = None
        self._init_ui()
        self._attach_log_handler()

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
        self._registration_button = QtWidgets.QPushButton("Registration")
        self._registration_button.clicked.connect(self._open_registration_from_button)
        map_control_layout.addWidget(self._registration_button)
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

        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_config = self._config.get("logging", {})
        max_lines = int(log_config.get("gui_max_lines", 1000))
        self._log_viewer = LogViewer(max_lines=max_lines)
        log_layout.addWidget(self._log_viewer)
        layout.addWidget(log_group)

        self.setCentralWidget(central_widget)

    def _attach_log_handler(self) -> None:
        """Attach the GUI log handler to the root logger.

        Returns:
            None.
        """

        if self._log_viewer is None:
            return
        log_config = self._config.get("logging", {})
        emitter = LogEmitter()
        emitter.message.connect(self._log_viewer.append_message)
        handler = GuiLogHandler(emitter)
        handler.setFormatter(
            logging.Formatter(
                log_config.get(
                    "format",
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
            )
        )
        gui_level = log_config.get("gui_level")
        if gui_level:
            handler.setLevel(getattr(logging, str(gui_level).upper(), logging.INFO))
        logging.getLogger().addHandler(handler)
        self._log_emitter = emitter
        self._log_handler = handler

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window close events.

        Parameters:
            event: Qt close event instance.

        Returns:
            None.
        """

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        self._close_scans()
        super().closeEvent(event)

    def load_scans(self, path_a: Path, path_b: Path) -> None:
        """Load scan datasets and update the GUI.

        Parameters:
            path_a: Path to scan A.
            path_b: Path to scan B.
        """

        self._logger.info("Loading scans: %s and %s", path_a, path_b)
        field_aliases = self._config.get("field_aliases", {})
        dataset_a = OH5ScanFileReader.from_path(path_a, field_aliases=field_aliases)
        dataset_b = OH5ScanFileReader.from_path(path_b, field_aliases=field_aliases)
        self._file_a_edit.setText(str(path_a))
        self._file_b_edit.setText(str(path_b))
        self.load_scan_datasets(dataset_a, dataset_b)

    def load_scan_datasets(self, dataset_a: ScanDataset, dataset_b: ScanDataset) -> None:
        """Load pre-initialized scan datasets into the GUI.

        Parameters:
            dataset_a: ScanDataset for scan A.
            dataset_b: ScanDataset for scan B.

        Returns:
            None.
        """

        self._close_scans()
        self._scan_a = dataset_a
        self._scan_b = dataset_b
        self._file_a_edit.setText(str(dataset_a.file_path))
        self._file_b_edit.setText(str(dataset_b.file_path))
        self._alignment_result = None
        alignment_config = self._config.get("alignment", {})
        try:
            precomputed = alignment_from_config(alignment_config, logger=self._logger)
        except Exception as exc:
            self._logger.exception("Failed to load alignment from config: %s", exc)
            precomputed = None
        mismatch = self._scans_mismatch()
        if mismatch and bool(alignment_config.get("auto_launch_on_mismatch", True)):
            self._logger.warning("Scan grids mismatch; launching registration tool.")
            self._alignment_result = self._open_registration(initial_alignment=precomputed)
            if self._alignment_result is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Alignment required",
                    "Alignment is required before comparison can proceed.",
                )
                self._reset_display()
                return
        else:
            self._alignment_result = precomputed
            if mismatch and self._alignment_result is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Alignment required",
                    "Alignment is required before comparison can proceed.",
                )
                self._reset_display()
                return
        self._engine = ComparisonEngine(
            self._scan_a,
            self._scan_b,
            self._config,
            logger=self._logger,
            alignment=self._alignment_result,
        )
        if self._alignment_result is not None:
            self._logger.info(
                "Alignment active: rotation=%.3f deg, translation=(%.3f, %.3f)",
                self._alignment_result.rotation_deg,
                self._alignment_result.translation[0],
                self._alignment_result.translation[1],
            )
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

    def _close_scans(self) -> None:
        """Close any open scan readers.

        Returns:
            None.
        """

        if self._scan_a is not None:
            self._scan_a.close()
        if self._scan_b is not None:
            self._scan_b.close()
        self._scan_a = None
        self._scan_b = None
        self._engine = None
        self._reset_display()

    def _reset_display(self) -> None:
        """Reset GUI panels to an empty state.

        Returns:
            None.
        """

        blank = np.zeros((1, 1))
        self._map_canvas_a.update_data(blank)
        self._map_canvas_b.update_data(blank)
        self._map_canvas_d.update_data(blank)
        self._pattern_canvas_a.update_data(blank, cmap="gray")
        self._pattern_canvas_b.update_data(blank, cmap="gray")
        self._pattern_canvas_d.update_data(blank, cmap="gray")
        self._map_field_combo.clear()
        self._probe_table.setRowCount(0)
        self._pattern_group.setVisible(False)

    def _scans_mismatch(self) -> bool:
        """Return True if scan grids differ.

        Returns:
            True if grid shapes differ, otherwise False.
        """

        if self._scan_a is None or self._scan_b is None:
            return False
        return self._scan_a.nx != self._scan_b.nx or self._scan_a.ny != self._scan_b.ny

    def _open_registration(
        self, initial_alignment: Optional[AlignmentResult] = None
    ) -> Optional[AlignmentResult]:
        """Open the registration dialog and return the alignment result.

        Parameters:
            initial_alignment: Optional alignment to preload.

        Returns:
            AlignmentResult or None if cancelled.
        """

        if self._scan_a is None or self._scan_b is None:
            return None
        dialog = RegistrationDialog(
            self._scan_a,
            self._scan_b,
            self._config,
            logger=self._logger,
            parent=self,
            initial_alignment=initial_alignment,
        )
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            return dialog.alignment_result()
        return None

    def _open_registration_from_button(self) -> None:
        """Open the registration dialog and apply the result.

        Returns:
            None.
        """

        if self._scan_a is None or self._scan_b is None:
            QtWidgets.QMessageBox.warning(
                self, "No scans", "Load scans before opening registration."
            )
            return
        alignment = self._open_registration(initial_alignment=self._alignment_result)
        if alignment is None:
            return
        self._alignment_result = alignment
        if self._engine is None:
            self._engine = ComparisonEngine(
                self._scan_a,
                self._scan_b,
                self._config,
                logger=self._logger,
                alignment=alignment,
            )
            self._populate_map_fields()
            self._select_pattern_field()
        else:
            self._engine.set_alignment(alignment)
        self._update_maps()
        x, y = self._engine.default_probe_xy()
        self._update_probe(x, y)
        self._logger.info("Alignment applied via registration dialog.")
        self.statusBar().showMessage("Alignment applied.", 5000)

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
        if pattern_triplet["B"] is None or pattern_triplet["D"] is None:
            placeholder = np.zeros_like(pattern_triplet["A"])
            self._pattern_canvas_b.update_data(placeholder, cmap="gray")
            self._pattern_canvas_d.update_data(placeholder, cmap="gray")
            return
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
        help="Enable debug logging and load simulated scans if paths are omitted.",
    )
    return parser


def main() -> None:
    """Run the EBSD compare GUI."""

    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config).get("ebsd_compare", {})
    configure_logging(args.debug, config.get("logging"))
    logger = logging.getLogger(__name__)
    app = QtWidgets.QApplication([])
    window = EbsdCompareMainWindow(args.config)
    window.show()
    if args.scan_a and args.scan_b:
        window.load_scans(args.scan_a, args.scan_b)
    elif args.debug:
        factory = SimulatedScanFactory.from_config(config.get("debug", {}), logger)
        scan_a, scan_b = factory.create_pair()
        window.load_scan_datasets(scan_a, scan_b)
    app.exec()


if __name__ == "__main__":
    main()
