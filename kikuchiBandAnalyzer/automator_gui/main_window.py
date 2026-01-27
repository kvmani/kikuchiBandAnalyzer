"""Visualization-first GUI for running the Kikuchi band-width automator.

Users load an existing YAML configuration file (no in-GUI parameter editing)
and run the same analysis pipeline as :mod:`KikuchiBandWidthAutomator`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import matplotlib
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from configLoader import load_config
from kikuchiBandAnalyzer.automator_gui.worker import AutomatorWorker
from kikuchiBandAnalyzer.ebsd_compare.band_data import extract_band_profile_payload
from kikuchiBandAnalyzer.ebsd_compare.gui.band_profile_plot import BandProfilePlot
from kikuchiBandAnalyzer.ebsd_compare.gui.logging_widget import (
    GuiLogHandler,
    LogEmitter,
    LogViewer,
)
from kikuchiBandAnalyzer.ebsd_compare.gui.main_window import MapPanel
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging

matplotlib.use("QtAgg")


class AutomatorGuiMainWindow(QtWidgets.QMainWindow):
    """Main window for the Kikuchi BandWidth Automator GUI."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the GUI.

        Parameters:
            config_path: Optional YAML configuration file to load on startup.
        """

        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._config_path: Optional[Path] = None
        self._config: Optional[Dict[str, Any]] = None
        self._scan_dataset = None
        self._output_dataset = None
        self._pattern_field: Optional[str] = None
        self._selected_xy: Optional[Tuple[int, int]] = None
        self._worker: Optional[AutomatorWorker] = None
        self._log_handler: Optional[GuiLogHandler] = None
        self._log_viewer: Optional[LogViewer] = None
        self._log_dock: Optional[QtWidgets.QDockWidget] = None
        self._main_splitter: Optional[QtWidgets.QSplitter] = None
        self._left_splitter: Optional[QtWidgets.QSplitter] = None
        self._right_splitter: Optional[QtWidgets.QSplitter] = None

        self._init_ui()
        self._attach_log_handler()
        if config_path is not None:
            self.load_yaml(config_path)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window close events."""

        self._stop_worker()
        self._close_scan()
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)

    def _init_ui(self) -> None:
        """Build the GUI layout and widgets."""

        self.setWindowTitle("Kikuchi BandWidth Automator (GUI)")
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        config_group = QtWidgets.QGroupBox("YAML Configuration (read-only)")
        config_layout = QtWidgets.QVBoxLayout(config_group)
        path_row = QtWidgets.QHBoxLayout()
        self._config_path_edit = QtWidgets.QLineEdit()
        self._config_path_edit.setReadOnly(True)
        self._config_path_edit.setPlaceholderText("Select a bandDetectorOptions*.yml file…")
        browse_btn = QtWidgets.QPushButton("Browse…")
        load_btn = QtWidgets.QPushButton("Load")
        browse_btn.clicked.connect(self._browse_yaml)
        load_btn.clicked.connect(self._load_yaml_from_input)
        path_row.addWidget(QtWidgets.QLabel("YAML"))
        path_row.addWidget(self._config_path_edit, stretch=1)
        path_row.addWidget(browse_btn)
        path_row.addWidget(load_btn)
        self._config_summary = QtWidgets.QPlainTextEdit()
        self._config_summary.setReadOnly(True)
        self._config_summary.setMaximumBlockCount(2000)
        self._config_summary.setPlaceholderText("Config summary will appear here after loading.")
        config_layout.addLayout(path_row)
        config_layout.addWidget(self._config_summary)

        # LEFT (bottom): EBSD map viewer
        map_container = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(6)
        map_controls = QtWidgets.QHBoxLayout()
        self._map_field_combo = QtWidgets.QComboBox()
        self._map_field_combo.currentTextChanged.connect(self._refresh_map)
        self._map_field_combo.setToolTip("Select the scalar field displayed on the EBSD map.")
        self._cursor_label = QtWidgets.QLabel("Cursor: (—, —)")
        self._cursor_label.setStyleSheet("color: #606060;")
        map_controls.addWidget(QtWidgets.QLabel("Map field"))
        map_controls.addWidget(self._map_field_combo, stretch=1)
        map_controls.addWidget(self._cursor_label)
        map_layout.addLayout(map_controls)
        self._map_panel = MapPanel("EBSD Map", 2.0, 98.0)
        self._map_panel.canvas().connect_click(self._on_map_click)
        self._map_panel.connect_contrast_changed(self._on_map_contrast_changed)
        self._map_panel.canvas().mpl_connect("motion_notify_event", self._on_map_hover)
        map_layout.addWidget(self._map_panel, stretch=1)

        # RIGHT (top): pattern viewer
        pattern_container = QtWidgets.QWidget()
        pattern_layout = QtWidgets.QVBoxLayout(pattern_container)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.setSpacing(6)
        self._pattern_panel = MapPanel("Kikuchi Pattern", 1.0, 99.0)
        self._pattern_panel.connect_contrast_changed(self._on_pattern_contrast_changed)
        pattern_layout.addWidget(self._pattern_panel, stretch=1)
        self._overlay_checkbox = QtWidgets.QCheckBox("Show band overlay line")
        self._overlay_checkbox.setChecked(True)
        self._overlay_checkbox.stateChanged.connect(self._refresh_profile_and_overlay)
        pattern_layout.addWidget(self._overlay_checkbox)

        # RIGHT (bottom): profile + metrics
        profile_container = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(4)
        self._profile_plot = BandProfilePlot(
            title="Band Profile",
            label_a="Scan",
            label_b="(unused)",
            marker_labels_include_series=False,
            logger=self._logger,
        )
        self._profile_plot.setMinimumHeight(220)
        profile_layout.addWidget(self._profile_plot, stretch=1)
        self._metrics_label = QtWidgets.QLabel("Band metrics: N/A (run analysis to populate outputs)")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("color: #404040;")
        profile_layout.addWidget(self._metrics_label)

        # Assemble splitters: left column (YAML + map), right column (pattern + profile).
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.addWidget(config_group)
        left_splitter.addWidget(map_container)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 3)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(pattern_container)
        right_splitter.addWidget(profile_container)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 3)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        root_layout.addWidget(main_splitter, stretch=1)

        self._main_splitter = main_splitter
        self._left_splitter = left_splitter
        self._right_splitter = right_splitter
        QtCore.QTimer.singleShot(0, self._apply_default_splitter_sizes)

        run_group = QtWidgets.QGroupBox("Run")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        top_row = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run Analysis")
        self._run_button.clicked.connect(self._start_run)
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_run)
        self._stage_label = QtWidgets.QLabel("Stage: —")
        self._stage_label.setStyleSheet("color: #303030; font-weight: 600;")
        top_row.addWidget(self._run_button)
        top_row.addWidget(self._cancel_button)
        top_row.addStretch(1)
        top_row.addWidget(self._stage_label)
        run_layout.addLayout(top_row)
        bottom_row = QtWidgets.QHBoxLayout()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._eta_label = QtWidgets.QLabel("ETA: —")
        self._pixel_label = QtWidgets.QLabel("Pixel: —")
        self._eta_label.setStyleSheet("color: #606060;")
        self._pixel_label.setStyleSheet("color: #606060;")
        bottom_row.addWidget(self._progress, stretch=1)
        bottom_row.addWidget(self._pixel_label)
        bottom_row.addWidget(self._eta_label)
        run_layout.addLayout(bottom_row)
        output_row = QtWidgets.QHBoxLayout()
        self._output_path_label = QtWidgets.QLabel("Output: —")
        self._output_path_label.setStyleSheet("color: #606060;")
        self._open_output_button = QtWidgets.QPushButton("Open output folder")
        self._open_output_button.setEnabled(False)
        self._open_output_button.clicked.connect(self._open_output_folder)
        output_row.addWidget(self._output_path_label, stretch=1)
        output_row.addWidget(self._open_output_button)
        run_layout.addLayout(output_row)
        root_layout.addWidget(run_group)

        self.setCentralWidget(central)

        self._log_viewer = LogViewer(max_lines=2000)
        dock = QtWidgets.QDockWidget("Log Console", self)
        dock.setObjectName("log_console")
        dock.setWidget(self._log_viewer)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        dock.setMinimumHeight(200)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self._log_dock = dock

    def _apply_default_splitter_sizes(self) -> None:
        """Apply initial splitter ratios for a balanced layout."""

        if (
            self._main_splitter is None
            or self._left_splitter is None
            or self._right_splitter is None
        ):
            return
        total_width = max(int(self.width()), 900)
        left_width = int(total_width * 0.45)
        self._main_splitter.setSizes([left_width, total_width - left_width])

        total_height = max(int(self.height()), 700)
        self._left_splitter.setSizes([int(total_height * 0.35), int(total_height * 0.65)])
        self._right_splitter.setSizes([int(total_height * 0.42), int(total_height * 0.58)])

    def _attach_log_handler(self) -> None:
        """Attach the GUI log handler."""

        if self._log_viewer is None:
            return
        emitter = LogEmitter()
        emitter.message.connect(self._log_viewer.append_entry)
        handler = GuiLogHandler(emitter)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _browse_yaml(self) -> None:
        """Open a file dialog to choose a YAML config."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select YAML configuration", filter="YAML Files (*.yml *.yaml)"
        )
        if path:
            self._config_path_edit.setText(path)

    def _load_yaml_from_input(self) -> None:
        """Load YAML configuration from the current path edit."""

        text = self._config_path_edit.text().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "No config", "Select a YAML config file first.")
            return
        self.load_yaml(Path(text))

    def load_yaml(self, path: Path) -> None:
        """Load and validate a YAML configuration.

        Parameters:
            path: YAML config file path.
        """

        if not path.exists():
            QtWidgets.QMessageBox.critical(self, "Missing file", f"Config file not found:\n{path}")
            return
        try:
            config = load_config(str(path))
        except Exception as exc:
            self._logger.exception("Failed to load YAML: %s", exc)
            QtWidgets.QMessageBox.critical(self, "YAML error", f"Failed to parse YAML:\n{exc}")
            return

        errors = self._validate_config(config)
        if errors:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid config",
                "The YAML configuration is missing required keys:\n\n- "
                + "\n- ".join(errors),
            )
            return
        logging.getLogger().setLevel(
            logging.DEBUG if bool(config.get("debug", False)) else logging.INFO
        )
        self._config_path = path
        self._config = config
        self._config_path_edit.setText(str(path))
        self._config_summary.setPlainText(self._format_config_summary(config))
        self._logger.info("Loaded YAML config: %s", path)
        self._load_scan_preview(Path(str(config["h5_file_path"])))

    def _validate_config(self, config: Dict[str, Any]) -> list[str]:
        """Validate the automator YAML structure.

        Parameters:
            config: Parsed YAML dictionary.

        Returns:
            List of missing/invalid key descriptions.
        """

        required = ["h5_file_path", "phase_list", "hkl_list", "desired_hkl", "desired_hkl_ref_width", "elastic_modulus"]
        missing = [key for key in required if key not in config]
        if "phase_list" in config and not isinstance(config["phase_list"], dict):
            missing.append("phase_list must be a mapping")
        if "hkl_list" in config and not isinstance(config["hkl_list"], (list, tuple)):
            missing.append("hkl_list must be a list")
        return missing

    def _format_config_summary(self, config: Dict[str, Any]) -> str:
        """Create a compact read-only configuration summary string."""

        lines = []
        for key in ("h5_file_path", "desired_hkl", "rectWidth", "debug", "crop_start", "crop_end"):
            if key in config:
                lines.append(f"{key}: {config.get(key)}")
        phase = config.get("phase_list") or {}
        if isinstance(phase, dict):
            lines.append(f"phase_list.name: {phase.get('name')}")
            lines.append(f"phase_list.space_group: {phase.get('space_group')}")
        lines.append(f"hkl_list: {config.get('hkl_list')}")
        return "\n".join(lines)

    def _load_scan_preview(self, file_path: Path) -> None:
        """Load the input OH5/HDF5 file for visualization.

        Parameters:
            file_path: Path to the input `.oh5`/`.h5`.
        """

        errors = self._validate_input_file(file_path)
        if errors:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input file",
                f"Input file validation failed:\n{file_path}\n\n- " + "\n- ".join(errors),
            )
            return
        self._close_scan()
        self._scan_dataset = OH5ScanFileReader.from_path(file_path)
        scalar_fields = self._scan_dataset.catalog.list_scalar_fields()
        self._map_field_combo.blockSignals(True)
        self._map_field_combo.clear()
        self._map_field_combo.addItems(scalar_fields)
        self._map_field_combo.blockSignals(False)
        default_map = "IQ" if "IQ" in scalar_fields else (scalar_fields[0] if scalar_fields else "")
        if default_map:
            self._map_field_combo.setCurrentText(default_map)
        patterns = self._scan_dataset.catalog.list_pattern_fields()
        self._pattern_field = "Pattern" if "Pattern" in patterns else (patterns[0] if patterns else None)
        self._logger.info(
            "Loaded preview scan: %s (nx=%s ny=%s patterns=%s)",
            file_path,
            self._scan_dataset.nx,
            self._scan_dataset.ny,
            len(patterns),
        )
        x = max(0, self._scan_dataset.nx // 2)
        y = max(0, self._scan_dataset.ny // 2)
        self._set_selected_pixel(x, y)
        self._refresh_map(reset_view=True)

    def _validate_input_file(self, file_path: Path) -> list[str]:
        """Validate HDF5 structure needed for the GUI preview."""

        if not file_path.exists():
            return ["file does not exist"]
        errors = []
        try:
            with h5py.File(file_path, "r") as handle:
                scan_name = None
                for key in handle.keys():
                    if key not in {"Manufacturer", "Version"} and isinstance(handle[key], h5py.Group):
                        scan_name = key
                        break
                if scan_name is None:
                    return ["no scan group found (expected a top-level scan group)"]
                data = handle.get(f"{scan_name}/EBSD/Data")
                if data is None:
                    errors.append("missing group /<scan>/EBSD/Data")
                else:
                    if "Pattern" not in data:
                        errors.append("missing dataset /<scan>/EBSD/Data/Pattern")
        except Exception as exc:
            errors.append(f"failed to open HDF5: {exc}")
        return errors

    def _close_scan(self) -> None:
        """Close any open preview/output datasets."""

        for dataset in (self._scan_dataset, self._output_dataset):
            if dataset is not None:
                try:
                    dataset.close()
                except Exception:
                    pass
        self._scan_dataset = None
        self._output_dataset = None
        self._pattern_field = None
        self._open_output_button.setEnabled(False)
        self._output_path_label.setText("Output: —")

    def _set_selected_pixel(self, x: int, y: int) -> None:
        """Update the selected pixel and refresh dependent views."""

        self._selected_xy = (x, y)
        self._map_panel.canvas().set_marker(x, y)
        self._refresh_pattern(reset_view=True)
        self._refresh_profile_and_overlay()
        self.statusBar().showMessage(f"Selected X={x}, Y={y}", 4000)

    def _on_map_click(self, event) -> None:
        """Handle mouse clicks on the EBSD map."""

        if self._scan_dataset is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or y < 0 or x >= self._scan_dataset.nx or y >= self._scan_dataset.ny:
            return
        self._set_selected_pixel(x, y)

    def _on_map_hover(self, event) -> None:
        """Update cursor readout on map hover."""

        if event.xdata is None or event.ydata is None:
            self._cursor_label.setText("Cursor: (—, —)")
            return
        self._cursor_label.setText(f"Cursor: ({event.xdata:.1f}, {event.ydata:.1f})")

    def _calculate_limits(self, data: np.ndarray, low_pct: float, high_pct: float) -> Tuple[float, float]:
        """Compute contrast limits from percentiles."""

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(finite, low_pct))
        vmax = float(np.percentile(finite, high_pct))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _on_map_contrast_changed(self, *_: object) -> None:
        """Handle contrast changes for the map panel."""

        self._refresh_map(reset_view=False)

    def _on_pattern_contrast_changed(self, *_: object) -> None:
        """Handle contrast changes for the pattern panel."""

        self._refresh_pattern(reset_view=False)

    def _refresh_map(self, reset_view: bool = True) -> None:
        """Refresh the EBSD map display."""

        if self._scan_dataset is None:
            self._map_panel.canvas().update_data(np.zeros((2, 2), dtype=np.float32), reset_view=True)
            return
        field = self._map_field_combo.currentText()
        if not field:
            return
        data = self._scan_dataset.get_map(field)
        low, high = self._map_panel.contrast_values()
        vmin, vmax = self._calculate_limits(data, low, high)
        self._map_panel.canvas().update_data(data, cmap="gray", vmin=vmin, vmax=vmax, reset_view=reset_view)

    def _refresh_pattern(self, reset_view: bool = True) -> None:
        """Refresh the pattern display for the selected pixel."""

        placeholder = np.zeros((32, 32), dtype=np.float32)
        if self._scan_dataset is None or self._pattern_field is None or self._selected_xy is None:
            self._pattern_panel.canvas().update_data(placeholder, cmap="gray", reset_view=True)
            return
        x, y = self._selected_xy
        pattern = self._scan_dataset.get_pattern(self._pattern_field, x, y)
        if pattern is None:
            self._logger.warning("Pattern missing at X=%s Y=%s.", x, y)
            self._pattern_panel.canvas().update_data(placeholder, cmap="gray", reset_view=reset_view)
            return
        low, high = self._pattern_panel.contrast_values()
        vmin, vmax = self._calculate_limits(pattern, low, high)
        self._pattern_panel.canvas().update_data(pattern, cmap="gray", vmin=vmin, vmax=vmax, reset_view=reset_view)

    def _refresh_profile_and_overlay(self) -> None:
        """Refresh the profile plot and pattern overlay using output datasets (when available)."""

        if self._selected_xy is None:
            self._profile_plot.clear("Select a pixel.")
            self._pattern_panel.canvas().clear_overlay_line()
            return
        x, y = self._selected_xy
        dataset = self._output_dataset
        if dataset is None and self._scan_dataset is not None:
            if "band_profile" in self._scan_dataset.catalog.vectors:
                dataset = self._scan_dataset
        if dataset is None:
            self._profile_plot.clear("Run analysis to generate band_profile outputs.")
            self._pattern_panel.canvas().clear_overlay_line()
            self._metrics_label.setText("Band metrics: N/A (run analysis to populate outputs)")
            return

        payload = extract_band_profile_payload(dataset, x, y, logger=self._logger)
        if payload.profile is None or not payload.band_valid:
            self._profile_plot.clear("No valid band at this pixel.")
            self._pattern_panel.canvas().clear_overlay_line()
            self._metrics_label.setText("Band metrics: No valid band at this pixel.")
            return
        self._profile_plot.update_plot(payload, None, normalize=True, show_markers=True)
        canvas = self._pattern_panel.canvas()
        if not self._overlay_checkbox.isChecked() or payload.central_line is None:
            canvas.clear_overlay_line()
        else:
            line = np.asarray(payload.central_line, dtype=np.float32).ravel()
            if line.size < 4 or not np.isfinite(line[:4]).all():
                canvas.clear_overlay_line()
                self._logger.debug(
                    "central_line unavailable at X=%s Y=%s (value=%s).", x, y, line
                )
            else:
                canvas.set_overlay_line(
                    float(line[0]),
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                )

        metrics = self._read_metrics(dataset, x, y)
        self._metrics_label.setText(metrics)

    def _read_metrics(self, dataset, x: int, y: int) -> str:
        """Read a compact set of scalar metrics from the output OH5."""

        fields = [
            "Band_Width",
            "psnr",
            "band_valid",
            "band_start_idx",
            "central_peak_idx",
            "band_end_idx",
            "band_intensity_ratio",
            "band_intensity_diff_norm",
        ]
        parts = []
        for field in fields:
            try:
                value = dataset.get_scalar(field, x, y)
            except Exception:
                continue
            parts.append(f"{field}={value:.4g}")
        if not parts:
            return "Band metrics: (no scalar band outputs found in file)"
        return "Band metrics: " + ", ".join(parts)

    def _start_run(self) -> None:
        """Start the automator pipeline in a worker thread."""

        if self._config_path is None:
            QtWidgets.QMessageBox.warning(self, "No config", "Load a YAML configuration first.")
            return
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "Busy", "Analysis is already running.")
            return
        self._logger.info("Starting analysis run with %s", self._config_path)
        self._progress.setValue(0)
        self._stage_label.setText("Stage: starting…")
        self._eta_label.setText("ETA: —")
        self._pixel_label.setText("Pixel: —")
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._map_panel.canvas().clear_secondary_marker()
        worker = AutomatorWorker(self._config_path, parent=self)
        worker.stage_changed.connect(self._on_worker_stage)
        worker.progress_changed.connect(self._on_worker_progress)
        worker.pixel_changed.connect(self._on_worker_pixel)
        worker.finished_success.connect(self._on_worker_finished)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _cancel_run(self) -> None:
        """Request cancellation of the running job."""

        if self._worker is None:
            return
        self._logger.warning("Cancellation requested by user.")
        self._worker.request_cancel()
        self._cancel_button.setEnabled(False)

    def _stop_worker(self) -> None:
        """Stop and cleanup a running worker."""

        if self._worker is None:
            return
        if self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(2000)
        self._worker = None

    def _open_output_folder(self) -> None:
        """Open the output folder in the system file browser."""

        if self._output_dataset is None:
            return
        folder = self._output_dataset.file_path.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _on_worker_stage(self, stage: str, index: int, total: int) -> None:
        """Update stage label and progress bar when stage changes."""

        self._stage_label.setText(f"Stage: {stage} ({index}/{total})")
        stage_progress = int(round((index - 1) / max(1, total) * 100))
        self._progress.setValue(stage_progress)

    def _on_worker_progress(self, processed: int, total: int, eta_seconds: float) -> None:
        """Update progress/ETA based on worker callbacks."""

        if total > 0:
            pct = int(round(100.0 * processed / total))
        else:
            pct = 0
        self._progress.setValue(pct)
        self._eta_label.setText(f"ETA: {eta_seconds:.1f} s")

    def _on_worker_pixel(self, x: int, y: int, processed: int) -> None:
        """Update the running pixel indicator."""

        self._pixel_label.setText(f"Pixel: X={x}, Y={y}")
        self._map_panel.canvas().set_secondary_marker(x, y, color="#ff0000")

    def _on_worker_finished(self, output_path: str, summary: object) -> None:
        """Handle successful completion of the analysis."""

        self._logger.info("Analysis complete: %s", output_path)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._progress.setValue(100)
        self._stage_label.setText("Stage: complete")
        self._map_panel.canvas().clear_secondary_marker()
        self._worker = None
        self._output_path_label.setText(f"Output: {output_path}")

        try:
            self._output_dataset = OH5ScanFileReader.from_path(Path(output_path))
            self._open_output_button.setEnabled(True)
        except Exception as exc:
            self._logger.exception("Failed to open output OH5 for preview: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Output preview failed", f"Could not open output file:\n{exc}")
        self._refresh_profile_and_overlay()

        message = f"Outputs written to:\n{output_path}\n\nSummary:\n{summary}"
        QtWidgets.QMessageBox.information(self, "Analysis complete", message)

    def _on_worker_cancelled(self, message: str) -> None:
        """Handle worker cancellation."""

        self._logger.warning("Analysis cancelled: %s", message)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._stage_label.setText("Stage: cancelled")
        self._worker = None
        self._open_output_button.setEnabled(False)
        QtWidgets.QMessageBox.information(self, "Cancelled", message)

    def _on_worker_failed(self, message: str) -> None:
        """Handle worker failures."""

        self._logger.error("Analysis failed: %s", message)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._stage_label.setText("Stage: failed")
        self._worker = None
        self._open_output_button.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Analysis failed", message)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for launching the GUI."""

    parser = argparse.ArgumentParser(description="Kikuchi BandWidth Automator GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the automator YAML config (bandDetectorOptions*.yml).",
    )
    return parser


def main() -> None:
    """CLI entry point for running the GUI."""

    args = build_arg_parser().parse_args()
    configure_logging(False, None)
    app = QtWidgets.QApplication([])
    window = AutomatorGuiMainWindow(config_path=args.config)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
