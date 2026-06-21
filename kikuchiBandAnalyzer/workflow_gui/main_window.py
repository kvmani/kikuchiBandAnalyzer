"""End-to-end GUI for EBSD input preparation and Kikuchi band analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
from typing import Any, Optional

import kikuchipy as kp
import numpy as np
import yaml
from diffsims.crystallography import ReciprocalLatticeVector
from diffpy.structure import Atom, Lattice, Structure
from orix import plot
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d
from PySide6 import QtCore, QtGui, QtWidgets

from kikuchiBandAnalyzer.automator_gui.worker import AutomatorWorker
from kikuchiBandAnalyzer.ebsd_compare.band_data import BandProfilePayload, extract_band_profile_payload
from kikuchiBandAnalyzer.ebsd_compare.gui.band_profile_plot import BandProfilePlot
from kikuchiBandAnalyzer.ebsd_compare.gui.logging_widget import GuiLogHandler, LogEmitter, LogViewer
from kikuchiBandAnalyzer.ebsd_compare.gui.main_window import MapPanel
from kikuchiBandAnalyzer.ebsd_compare.map_display import MapDisplaySettings
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging
from kikuchiBandAnalyzer.io.hkl_ctf_to_tsl import convert_hkl_ctf_fixture_to_tsl, parse_ctf_file
from kikuchiBandAnalyzer.single_pattern_solver.gui import SinglePatternCanvas
from kikuchiBandAnalyzer.single_pattern_solver.solver import (
    SinglePatternConfig,
    SinglePatternSolution,
    solve_single_pattern,
)
from kikuchiBandAnalyzer.workflow_gui.map_settings_dialog import MapDisplaySettingsDialog
from simulators import CustomKikuchiPatternSimulator
import utilities as ut


DEFAULT_FCC_HKLS = [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]


class WorkflowGuiMainWindow(QtWidgets.QMainWindow):
    """Main window for input preparation, indexing, and band-width analysis."""

    def __init__(
        self,
        *,
        input_mode: str = "ctf",
        source_path: Optional[Path] = None,
        pattern_dir: Optional[Path] = None,
        ang_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        """Initialize the workflow GUI.

        Parameters:
            input_mode: Initial input mode, either ``ctf`` or ``tsl``.
            source_path: Optional CTF/OH5/H5 source path.
            pattern_dir: Optional CTF pattern directory.
            ang_path: Optional ANG path for TSL mode.
            output_dir: Optional output directory.
            config_path: Optional workflow or single-pattern YAML configuration.
        """

        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._scan_dataset = None
        self._output_dataset = None
        self._pattern_field: Optional[str] = None
        self._selected_xy: Optional[tuple[int, int]] = None
        self._prepared_h5_path: Optional[Path] = None
        self._prepared_oh5_path: Optional[Path] = None
        self._prepared_ang_path: Optional[Path] = None
        self._resolved_config_path: Optional[Path] = None
        self._worker: Optional[AutomatorWorker] = None
        self._log_handler: Optional[GuiLogHandler] = None
        self._simulated_lines_by_index: dict[int, list[dict[str, object]]] = {}
        self._single_solution: Optional[SinglePatternSolution] = None
        self._map_panels: dict[str, tuple[MapPanel, MapPanel]] = {}
        self._ipf_cache: dict[str, np.ndarray] = {}
        self._map_display_settings: dict[str, MapDisplaySettings] = {}
        self._orientation_diagnostics: Any = None
        self._live_render_clock = QtCore.QElapsedTimer()
        self._live_render_clock.start()
        self._auto_solve_timer = QtCore.QTimer(self)
        self._auto_solve_timer.setSingleShot(True)
        self._auto_solve_timer.setInterval(250)
        self._auto_solve_timer.timeout.connect(self._solve_selected_pattern)

        self._init_ui()
        self._attach_log_handler()
        self._apply_initial_values(input_mode, source_path, pattern_dir, ang_path, output_dir)
        if config_path is not None:
            self.load_config(config_path)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Clean up worker and open files when the window closes.

        Parameters:
            event: Qt close event.

        Returns:
            None.
        """

        self._stop_worker()
        self._close_datasets()
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)

    def _init_ui(self) -> None:
        """Build the GUI layout and connect widget signals."""

        self.setWindowTitle("Kikuchi EBSD Workflow")
        self.resize(1500, 950)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(self._build_control_panel())
        main_splitter.addWidget(self._build_visual_panel())
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        root.addWidget(main_splitter, stretch=1)
        root.addWidget(self._build_run_panel())
        self.setCentralWidget(central)

        self._log_viewer = LogViewer(max_lines=4000)
        dock = QtWidgets.QDockWidget("Log Console", self)
        dock.setWidget(self._log_viewer)
        dock.setObjectName("workflow_log_console")
        dock.setMinimumHeight(210)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _build_control_panel(self) -> QtWidgets.QWidget:
        """Create the left-side workflow controls.

        Returns:
            Configured control widget.
        """

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(470)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        input_group = QtWidgets.QGroupBox("Input")
        form = QtWidgets.QFormLayout(input_group)
        self._config_edit = self._path_edit()
        self._config_button = QtWidgets.QPushButton("Load")
        self._config_button.clicked.connect(self._browse_config)
        form.addRow("YAML", self._path_row(self._config_edit, self._config_button))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItem("HKL/Oxford CTF + patterns", "ctf")
        self._mode_combo.addItem("TSL/EDAX OH5/H5 + ANG", "tsl")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode", self._mode_combo)

        self._source_edit = self._path_edit()
        self._source_button = QtWidgets.QPushButton("Browse")
        self._source_button.clicked.connect(self._browse_source)
        form.addRow("Source", self._path_row(self._source_edit, self._source_button))

        self._pattern_edit = self._path_edit()
        self._pattern_button = QtWidgets.QPushButton("Browse")
        self._pattern_button.clicked.connect(self._browse_pattern_dir)
        form.addRow("Patterns", self._path_row(self._pattern_edit, self._pattern_button))

        self._ang_edit = self._path_edit()
        self._ang_button = QtWidgets.QPushButton("Browse")
        self._ang_button.clicked.connect(self._browse_ang)
        form.addRow("ANG", self._path_row(self._ang_edit, self._ang_button))

        self._output_edit = self._path_edit()
        self._output_button = QtWidgets.QPushButton("Browse")
        self._output_button.clicked.connect(self._browse_output_dir)
        form.addRow("Output dir", self._path_row(self._output_edit, self._output_button))

        self._template_edit = QtWidgets.QLineEdit("{x}_{y}.tiff")
        self._template_edit.setToolTip("CTF pattern filename template. Use {x}, {y}, {row}, {col}, or {index}.")
        form.addRow("CTF pattern map", self._template_edit)
        layout.addWidget(input_group)

        analysis_group = QtWidgets.QGroupBox("Analysis")
        analysis_form = QtWidgets.QFormLayout(analysis_group)
        self._phase_name_edit = QtWidgets.QLineEdit("Cr")
        self._space_group_spin = QtWidgets.QSpinBox()
        self._space_group_spin.setRange(1, 230)
        self._space_group_spin.setValue(225)
        self._lattice_edit = QtWidgets.QLineEdit("2.91, 2.91, 2.91, 90, 90, 90")
        self._desired_hkl_edit = QtWidgets.QLineEdit("1,1,1")
        self._hkl_list_edit = QtWidgets.QPlainTextEdit("[[1,1,1], [2,0,0], [2,2,0], [3,1,1]]")
        self._hkl_list_edit.setMaximumHeight(58)
        self._pc_edit = QtWidgets.QLineEdit("0.457, 0.584, 0.696374")
        self._detector_convention_combo = QtWidgets.QComboBox()
        self._detector_convention_combo.addItems(["oxford", "edax", "tsl"])
        self._sample_tilt_spin = self._double_spin(70.0, -180.0, 180.0, 2)
        self._camera_tilt_spin = self._double_spin(0.0, -180.0, 180.0, 2)
        self._azimuth_spin = self._double_spin(0.0, -180.0, 180.0, 2)
        self._ref_width_spin = self._double_spin(1.0, 0.000001, 1e9, 6)
        self._modulus_spin = self._double_spin(205e9, 0.0, 1e15, 3)
        self._rect_width_spin = QtWidgets.QSpinBox()
        self._rect_width_spin.setRange(1, 500)
        self._rect_width_spin.setValue(20)
        self._min_psnr_spin = self._double_spin(1.01, 0.0, 1000.0, 3)
        self._debug_checkbox = QtWidgets.QCheckBox("Debug")
        self._orientation_source_combo = QtWidgets.QComboBox()
        self._orientation_source_combo.addItem("Live Hough indexed (PyEBSDIndex)", "indexed")
        self._orientation_source_combo.addItem("Acquisition Euler angles", "acquisition")
        self._orientation_source_combo.currentIndexChanged.connect(
            self._on_orientation_source_changed
        )
        analysis_form.addRow("Phase", self._phase_name_edit)
        analysis_form.addRow("Space group", self._space_group_spin)
        analysis_form.addRow("Lattice", self._lattice_edit)
        analysis_form.addRow("Desired HKL", self._desired_hkl_edit)
        analysis_form.addRow("HKL list", self._hkl_list_edit)
        analysis_form.addRow("PC", self._pc_edit)
        analysis_form.addRow("PC convention", self._detector_convention_combo)
        analysis_form.addRow("Sample tilt", self._sample_tilt_spin)
        analysis_form.addRow("Detector tilt", self._camera_tilt_spin)
        analysis_form.addRow("Azimuthal", self._azimuth_spin)
        analysis_form.addRow("Reference width", self._ref_width_spin)
        analysis_form.addRow("Elastic modulus", self._modulus_spin)
        analysis_form.addRow("rectWidth", self._rect_width_spin)
        analysis_form.addRow("min_psnr", self._min_psnr_spin)
        analysis_form.addRow("Euler source", self._orientation_source_combo)
        analysis_form.addRow("", self._debug_checkbox)
        layout.addWidget(analysis_group)

        calibration_group = QtWidgets.QGroupBox("Selected Pattern")
        calibration_form = QtWidgets.QFormLayout(calibration_group)
        self._pixel_x_spin = QtWidgets.QSpinBox()
        self._pixel_y_spin = QtWidgets.QSpinBox()
        for spin in (self._pixel_x_spin, self._pixel_y_spin):
            spin.setRange(0, 0)
        self._pixel_x_spin.valueChanged.connect(self._on_pixel_controls_changed)
        self._pixel_y_spin.valueChanged.connect(self._on_pixel_controls_changed)
        pixel_row = QtWidgets.QWidget()
        pixel_layout = QtWidgets.QHBoxLayout(pixel_row)
        pixel_layout.setContentsMargins(0, 0, 0, 0)
        pixel_layout.addWidget(QtWidgets.QLabel("X"))
        pixel_layout.addWidget(self._pixel_x_spin)
        pixel_layout.addWidget(QtWidgets.QLabel("Y"))
        pixel_layout.addWidget(self._pixel_y_spin)
        calibration_form.addRow("Pixel", pixel_row)
        self._auto_resolve_checkbox = QtWidgets.QCheckBox("Re-solve after PC drag")
        self._auto_resolve_checkbox.setChecked(True)
        calibration_form.addRow("", self._auto_resolve_checkbox)
        self._solve_pixel_button = QtWidgets.QPushButton("Solve Selected Pattern")
        self._solve_pixel_button.clicked.connect(self._solve_selected_pattern)
        calibration_form.addRow("", self._solve_pixel_button)
        layout.addWidget(calibration_group)

        self._prepare_button = QtWidgets.QPushButton("Prepare / Preview")
        self._prepare_button.clicked.connect(self.prepare_inputs)
        layout.addWidget(self._prepare_button)
        self._summary = QtWidgets.QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumBlockCount(1000)
        self._summary.setPlaceholderText("Preparation summary will appear here.")
        layout.addWidget(self._summary, stretch=1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(410)
        scroll.setMaximumWidth(500)
        return scroll

    def _build_visual_panel(self) -> QtWidgets.QWidget:
        """Create linked map tabs and shared pattern inspection widgets.

        Returns:
            Configured visual widget.
        """

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._cursor_label = QtWidgets.QLabel("Cursor: --")
        layout.addWidget(self._cursor_label, alignment=QtCore.Qt.AlignRight)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._result_tabs = QtWidgets.QTabWidget()
        map_specs = [
            ("IPF-X", "IPF-X"),
            ("IPF-Y", "IPF-Y"),
            ("IPF-Z", "IPF-Z"),
            ("Band Width", "Band_Width"),
            ("PSNR", "psnr"),
            ("Validity", "band_valid"),
            ("Strain", "strain"),
            ("Stress", "stress"),
            ("Index/Fallback", "orientation_fallback"),
        ]
        gear_icon = QtGui.QIcon.fromTheme("preferences-system")
        if gear_icon.isNull():
            gear_icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        for tab_label, field_name in map_specs:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QHBoxLayout(tab)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            iq_panel = MapPanel("IQ", 2.0, 98.0)
            result_panel = MapPanel(tab_label, 2.0, 98.0)
            self._map_display_settings.setdefault("IQ", self._default_map_display_settings("IQ"))
            self._map_display_settings.setdefault(
                field_name, self._default_map_display_settings(field_name)
            )
            iq_panel.add_tool_button(
                gear_icon,
                "IQ display properties",
                lambda _checked=False: self._edit_map_display("IQ"),
            )
            if not field_name.startswith("IPF-"):
                result_panel.add_tool_button(
                    gear_icon,
                    f"{tab_label} display properties",
                    lambda _checked=False, field=field_name: self._edit_map_display(field),
                )
            for map_panel in (iq_panel, result_panel):
                map_panel.canvas().connect_click(self._on_map_click)
                map_panel.canvas().mpl_connect("motion_notify_event", self._on_map_hover)
                map_panel.connect_contrast_changed(lambda *_: self._refresh_map(False))
                tab_layout.addWidget(map_panel, stretch=1)
            self._map_panels[field_name] = (iq_panel, result_panel)
            self._result_tabs.addTab(tab, tab_label)
        self._result_tabs.currentChanged.connect(lambda _index: self._refresh_map(False))
        self._map_panel = self._map_panels["IPF-X"][1]
        self._map_field_combo = QtWidgets.QComboBox()
        self._map_field_combo.addItems([field for _, field in map_specs])
        self._map_field_combo.setVisible(False)
        splitter.addWidget(self._result_tabs)

        inspector_tabs = QtWidgets.QTabWidget()
        self._single_pattern_canvas = SinglePatternCanvas(self._on_pc_dragged)
        inspector_tabs.addTab(self._single_pattern_canvas, "Diagnostic Solver")

        batch_inspector = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        batch_inspector.setChildrenCollapsible(False)
        self._pattern_panel = MapPanel("Pattern + Band Overlay", 1.0, 99.0)
        self._pattern_panel.connect_contrast_changed(lambda *_: self._refresh_pattern(False))
        pattern_container = QtWidgets.QWidget()
        pattern_layout = QtWidgets.QVBoxLayout(pattern_container)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        self._overlay_checkbox = QtWidgets.QCheckBox("Show detected band")
        self._overlay_checkbox.setChecked(True)
        self._overlay_checkbox.stateChanged.connect(self._refresh_profile_and_overlay)
        pattern_layout.addWidget(self._pattern_panel, stretch=1)
        pattern_layout.addWidget(self._overlay_checkbox)
        batch_inspector.addWidget(pattern_container)
        self._profile_plot = BandProfilePlot(
            title="Band Profile",
            label_a="Selected pixel",
            label_b="",
            marker_labels_include_series=False,
            logger=self._logger,
        )
        self._metrics_label = QtWidgets.QLabel("Band metrics: not available")
        self._metrics_label.setWordWrap(True)
        profile_container = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.addWidget(self._profile_plot, stretch=1)
        profile_layout.addWidget(self._metrics_label)
        batch_inspector.addWidget(profile_container)
        batch_inspector.setStretchFactor(0, 2)
        batch_inspector.setStretchFactor(1, 1)
        inspector_tabs.addTab(batch_inspector, "Batch Result")
        self._inspector_tabs = inspector_tabs
        splitter.addWidget(inspector_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)
        return panel

    def _build_run_panel(self) -> QtWidgets.QWidget:
        """Create the run/progress panel.

        Returns:
            Configured run widget.
        """

        panel = QtWidgets.QGroupBox("Run")
        layout = QtWidgets.QVBoxLayout(panel)
        row = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run Full Band-Width Analysis")
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._start_run)
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_run)
        self._stage_label = QtWidgets.QLabel("Stage: idle")
        row.addWidget(self._run_button)
        row.addWidget(self._cancel_button)
        row.addWidget(self._stage_label, stretch=1)
        layout.addLayout(row)
        progress_row = QtWidgets.QHBoxLayout()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._pixel_label = QtWidgets.QLabel("Pixel: --")
        self._eta_label = QtWidgets.QLabel("ETA: --")
        progress_row.addWidget(self._progress, stretch=1)
        progress_row.addWidget(self._pixel_label)
        progress_row.addWidget(self._eta_label)
        layout.addLayout(progress_row)
        output_row = QtWidgets.QHBoxLayout()
        self._output_label = QtWidgets.QLabel("Output: --")
        self._open_output_button = QtWidgets.QPushButton("Open output")
        self._open_output_button.setEnabled(False)
        self._open_output_button.clicked.connect(self._open_output_folder)
        output_row.addWidget(self._output_label, stretch=1)
        output_row.addWidget(self._open_output_button)
        layout.addLayout(output_row)
        return panel

    def _path_edit(self) -> QtWidgets.QLineEdit:
        """Create a path line edit.

        Returns:
            Read-write path edit widget.
        """

        edit = QtWidgets.QLineEdit()
        edit.setMinimumWidth(230)
        return edit

    def _path_row(self, edit: QtWidgets.QLineEdit, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        """Create a compact path-edit row.

        Parameters:
            edit: Line edit containing the path.
            button: Browse button.

        Returns:
            Container widget.
        """

        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, stretch=1)
        layout.addWidget(button)
        return row

    def _double_spin(self, value: float, minimum: float, maximum: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        """Create a double spin box.

        Parameters:
            value: Initial value.
            minimum: Minimum accepted value.
            maximum: Maximum accepted value.
            decimals: Number of decimals to show.

        Returns:
            Configured spin box.
        """

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _attach_log_handler(self) -> None:
        """Attach a Qt log handler to the root logger."""

        emitter = LogEmitter()
        emitter.message.connect(self._log_viewer.append_entry)
        handler = GuiLogHandler(emitter)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _apply_initial_values(
        self,
        input_mode: str,
        source_path: Optional[Path],
        pattern_dir: Optional[Path],
        ang_path: Optional[Path],
        output_dir: Optional[Path],
    ) -> None:
        """Apply optional constructor paths to the controls.

        Parameters:
            input_mode: Initial mode.
            source_path: Optional source file.
            pattern_dir: Optional pattern directory.
            ang_path: Optional ANG file.
            output_dir: Optional output directory.

        Returns:
            None.
        """

        self._mode_combo.setCurrentIndex(0 if input_mode == "ctf" else 1)
        if source_path:
            self._source_edit.setText(str(source_path))
        if pattern_dir:
            self._pattern_edit.setText(str(pattern_dir))
        if ang_path:
            self._ang_edit.setText(str(ang_path))
        if output_dir:
            self._output_edit.setText(str(output_dir))
        self._on_mode_changed()

    def _on_mode_changed(self, _index: object = None, *, preserve_values: bool = False) -> None:
        """Update controls for the active input mode.

        Parameters:
            _index: Optional combo-box signal payload.
            preserve_values: Keep YAML-provided phase, lattice, and PC values.

        Returns:
            None.
        """

        mode = self._current_mode()
        is_ctf = mode == "ctf"
        self._pattern_edit.setEnabled(is_ctf)
        self._pattern_button.setEnabled(is_ctf)
        self._template_edit.setEnabled(is_ctf)
        self._ang_edit.setEnabled(not is_ctf)
        self._ang_button.setEnabled(not is_ctf)
        if preserve_values:
            return
        self._detector_convention_combo.setCurrentText("oxford" if is_ctf else "edax")
        if is_ctf:
            self._phase_name_edit.setText("Cr")
            self._lattice_edit.setText("2.91, 2.91, 2.91, 90, 90, 90")
            self._pc_edit.setText("0.457, 0.584, 0.696374")
        else:
            self._phase_name_edit.setText("Ni")
            self._lattice_edit.setText("3.5236, 3.5236, 3.5236, 90, 90, 90")
            self._pc_edit.setText("0.547204, 0.710997, 0.696374")

    def _current_mode(self) -> str:
        """Return the active input mode.

        Returns:
            ``ctf`` or ``tsl``.
        """

        return str(self._mode_combo.currentData())

    def _browse_source(self) -> None:
        """Browse for source input file."""

        if self._current_mode() == "ctf":
            filt = "CTF Files (*.ctf)"
        else:
            filt = "HDF5/OH5 Files (*.oh5 *.h5 *.hdf5)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select source", filter=filt)
        if path:
            self._source_edit.setText(path)

    def _browse_config(self) -> None:
        """Browse for and load a workflow YAML configuration."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Workflow Configuration",
            filter="YAML Files (*.yml *.yaml);;All Files (*)",
        )
        if path:
            self.load_config(Path(path))

    def load_config(self, path: Path) -> None:
        """Load supported workflow or single-pattern YAML values into controls.

        Parameters:
            path: YAML configuration path.

        Returns:
            None.
        """

        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Workflow configuration must be a mapping: {config_path}")
        self._config_edit.setText(str(config_path))
        input_cfg = dict(raw.get("input", {}) or {})
        detector_cfg = dict(raw.get("detector", raw.get("ctf_detector", {})) or {})
        phase_cfg = dict(raw.get("phase", raw.get("phase_list", {})) or {})
        simulation_cfg = dict(raw.get("simulation", {}) or {})
        profile_cfg = dict(raw.get("band_profile", {}) or {})

        is_ctf = bool(raw.get("ctf_file_path")) or str(input_cfg.get("type", "")).lower() == "ctf"
        self._mode_combo.setCurrentIndex(0 if is_ctf else 1)
        source = raw.get("ctf_file_path") or raw.get("h5_file_path")
        source = input_cfg.get("ctf_path" if is_ctf else "path", source)
        if source:
            self._source_edit.setText(str(source))
        patterns = input_cfg.get("pattern_dir", raw.get("pattern_folder"))
        if patterns:
            self._pattern_edit.setText(str(patterns))
        template = input_cfg.get("pattern_template", raw.get("pattern_template"))
        if template:
            self._template_edit.setText(str(template))
        ang = raw.get("ang_file_path") or input_cfg.get("ang_path")
        if ang:
            self._ang_edit.setText(str(ang))
        if raw.get("output_dir"):
            self._output_edit.setText(str(raw["output_dir"]))

        if phase_cfg:
            self._phase_name_edit.setText(str(phase_cfg.get("name", self._phase_name_edit.text())))
            self._space_group_spin.setValue(int(phase_cfg.get("space_group", self._space_group_spin.value())))
            if phase_cfg.get("lattice"):
                self._lattice_edit.setText(", ".join(str(value) for value in phase_cfg["lattice"]))
        hkls = simulation_cfg.get("hkl_list", raw.get("hkl_list"))
        if hkls:
            self._hkl_list_edit.setPlainText(yaml.safe_dump(hkls, default_flow_style=True).strip())
        desired_hkl = profile_cfg.get("desired_hkl", raw.get("desired_hkl"))
        if desired_hkl:
            self._desired_hkl_edit.setText(str(desired_hkl))
        pc = detector_cfg.get("pc", raw.get("pc"))
        if pc:
            self._pc_edit.setText(", ".join(str(value) for value in pc))
        self._detector_convention_combo.setCurrentText(
            str(detector_cfg.get("convention", raw.get("detector_convention", self._detector_convention_combo.currentText())))
        )
        self._sample_tilt_spin.setValue(float(detector_cfg.get("sample_tilt", self._sample_tilt_spin.value())))
        self._camera_tilt_spin.setValue(float(detector_cfg.get("tilt", self._camera_tilt_spin.value())))
        self._azimuth_spin.setValue(float(detector_cfg.get("azimuthal", self._azimuth_spin.value())))
        self._ref_width_spin.setValue(float(raw.get("desired_hkl_ref_width", self._ref_width_spin.value())))
        self._modulus_spin.setValue(float(raw.get("elastic_modulus", self._modulus_spin.value())))
        self._rect_width_spin.setValue(int(profile_cfg.get("rectWidth", raw.get("rectWidth", self._rect_width_spin.value()))))
        self._min_psnr_spin.setValue(float(profile_cfg.get("min_psnr", raw.get("min_psnr", self._min_psnr_spin.value()))))
        self._debug_checkbox.setChecked(bool(raw.get("debug", False)))
        orientation_source = str(raw.get("orientation_source", "indexed")).lower()
        if orientation_source == "original":
            orientation_source = "acquisition"
        source_index = self._orientation_source_combo.findData(orientation_source)
        self._orientation_source_combo.setCurrentIndex(max(0, source_index))
        if input_cfg.get("x") is not None:
            self._pixel_x_spin.setValue(int(input_cfg["x"]))
        if input_cfg.get("y") is not None:
            self._pixel_y_spin.setValue(int(input_cfg["y"]))
        hough_cfg = dict(raw.get("hough", {}) or {})
        if "orientation_source" not in raw and not bool(
            hough_cfg.get("use_indexed_orientation", True)
        ):
            self._orientation_source_combo.setCurrentIndex(
                self._orientation_source_combo.findData("acquisition")
            )
        for field_name, values in dict(raw.get("map_display", {}) or {}).items():
            defaults = self._default_map_display_settings(str(field_name)).to_mapping()
            self._map_display_settings[str(field_name)] = MapDisplaySettings.from_mapping(
                values, **defaults
            )
        for map_field, (iq_panel, result_panel) in self._map_panels.items():
            iq_settings = self._map_display_settings.get("IQ")
            if iq_settings is not None and iq_settings.range_mode == "auto":
                iq_panel.set_contrast_values(
                    iq_settings.percentile_low,
                    iq_settings.percentile_high,
                    block_signals=True,
                )
            result_settings = self._map_display_settings.get(map_field)
            if result_settings is not None and result_settings.range_mode == "auto":
                result_panel.set_contrast_values(
                    result_settings.percentile_low,
                    result_settings.percentile_high,
                    block_signals=True,
                )
        self._on_mode_changed(preserve_values=True)
        self._logger.info("Loaded workflow configuration: %s", config_path)

    def _browse_pattern_dir(self) -> None:
        """Browse for CTF pattern directory."""

        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select pattern folder")
        if path:
            self._pattern_edit.setText(path)

    def _browse_ang(self) -> None:
        """Browse for ANG file."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select ANG", filter="ANG Files (*.ang)")
        if path:
            self._ang_edit.setText(path)

    def _browse_output_dir(self) -> None:
        """Browse for output directory."""

        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_edit.setText(path)

    def prepare_inputs(self) -> None:
        """Validate selected inputs, create working files, and load preview."""

        try:
            if self._current_mode() == "ctf":
                self._prepare_ctf_inputs()
            else:
                self._prepare_tsl_inputs()
            self._write_resolved_config()
            self._load_preview(self._prepared_h5_path or self._prepared_oh5_path)
            self._simulate_preview_lines()
            self._run_button.setEnabled(True)
        except Exception as exc:
            self._logger.exception("Preparation failed: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Preparation failed", str(exc))

    def _prepare_ctf_inputs(self) -> None:
        """Prepare HKL/Oxford CTF input by translating it to TSL files."""

        ctf_path = Path(self._source_edit.text().strip())
        pattern_dir = Path(self._pattern_edit.text().strip())
        output_dir = self._resolved_output_dir()
        repo_root = Path(__file__).resolve().parents[2]
        template = self._template_edit.text().strip() or "{x}_{y}.tiff"
        ctf = parse_ctf_file(ctf_path)
        self._logger.info("CTF grid: %d x %d, rows=%d.", ctf.nx, ctf.ny, ctf.nx * ctf.ny)
        result = convert_hkl_ctf_fixture_to_tsl(
            ctf_path=ctf_path,
            pattern_dir=pattern_dir,
            reference_h5_path=repo_root / "testData" / "DA.h5",
            reference_ang_path=repo_root / "testData" / "DA.ang",
            output_dir=output_dir,
            scan_name=ctf_path.stem,
            phase_name=self._phase_name_edit.text().strip() or "Cr",
            phase_formula=self._phase_name_edit.text().strip() or "Cr",
            lattice_parameter=float(self._parse_lattice()[0]),
            pattern_template=template,
            logger=self._logger,
        )
        self._prepared_h5_path = result.h5_path
        self._prepared_oh5_path = result.oh5_path
        self._prepared_ang_path = result.ang_path
        self._summary.setPlainText(
            "\n".join(
                [
                    "Prepared HKL/Oxford CTF input.",
                    f"Mapping: {result.pattern_mapping}",
                    f"Ignored extra images: {result.ignored_pattern_count}",
                    f"H5: {result.h5_path}",
                    f"OH5: {result.oh5_path}",
                    f"ANG: {result.ang_path}",
                    "Detector convention for indexing: oxford",
                ]
            )
        )

    def _prepare_tsl_inputs(self) -> None:
        """Prepare TSL/EDAX OH5/H5 plus ANG input for analysis."""

        source = Path(self._source_edit.text().strip())
        ang = Path(self._ang_edit.text().strip())
        if not source.exists():
            raise FileNotFoundError(source)
        if not ang.exists():
            raise FileNotFoundError(ang)
        output_dir = self._resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        copied_source = output_dir / source.name
        copied_ang = output_dir / f"{source.stem}.ang"
        if source.resolve() != copied_source.resolve():
            shutil.copy2(source, copied_source)
        if ang.resolve() != copied_ang.resolve():
            shutil.copy2(ang, copied_ang)
        self._prepared_oh5_path = copied_source if copied_source.suffix.lower() == ".oh5" else copied_source.with_suffix(".oh5")
        self._prepared_h5_path = copied_source
        self._prepared_ang_path = copied_ang
        self._summary.setPlainText(
            "\n".join(
                [
                    "Prepared TSL/EDAX input.",
                    f"Source: {copied_source}",
                    f"ANG: {copied_ang}",
                    "Detector convention for indexing: edax",
                ]
            )
        )

    def _write_resolved_config(self) -> None:
        """Write the run YAML consumed by the background automator."""

        output_dir = self._resolved_output_dir()
        config: dict[str, Any] = {
            "output_dir": str(output_dir),
            "desired_hkl": self._desired_hkl_value(),
            "desired_hkl_ref_width": float(self._ref_width_spin.value()),
            "elastic_modulus": float(self._modulus_spin.value()),
            "rectWidth": int(self._rect_width_spin.value()),
            "min_psnr": float(self._min_psnr_spin.value()),
            "smoothing_sigma": 2.0,
            "strategy": "rectangular_area",
            "hkl_list": self._parse_hkl_list(),
            "phase_list": {
                "name": self._phase_name_edit.text().strip(),
                "space_group": int(self._space_group_spin.value()),
                "lattice": self._parse_lattice(),
                "atoms": [
                    {
                        "element": self._phase_name_edit.text().strip(),
                        "position": [0, 0, 0],
                    }
                ],
            },
            "debug": bool(self._debug_checkbox.isChecked()),
            "plot_band_detection": False,
            "plot_band_detection_condition": "False",
            "skip_display_EBSDmap": True,
            "orientation_source": str(self._orientation_source_combo.currentData()),
            "orientation_direction": "lab2crystal",
            "hough": {
                "enabled": True,
                "use_indexed_orientation": self._orientation_source_combo.currentData()
                == "indexed",
                "n_bands": 10,
                "t_sigma": 2.0,
                "r_sigma": 2.0,
            },
            "map_display": {
                field: settings.to_mapping()
                for field, settings in self._map_display_settings.items()
            },
        }
        pc = self._parse_float_list(self._pc_edit.text(), expected=3, label="PC")
        convention = self._detector_convention_combo.currentText()
        if self._current_mode() == "ctf":
            config.update(
                {
                    "ctf_file_path": self._source_edit.text().strip(),
                    "pattern_folder": self._pattern_edit.text().strip(),
                    "pattern_template": self._template_edit.text().strip() or None,
                    "ctf_euler_direction": "lab2crystal",
                    "prepared_h5_path": str(self._prepared_h5_path),
                    "ang_template_path": str(self._prepared_ang_path),
                    "ctf_detector": {
                        "convention": convention,
                        "pc": pc,
                        "sample_tilt": float(self._sample_tilt_spin.value()),
                        "tilt": float(self._camera_tilt_spin.value()),
                        "azimuthal": float(self._azimuth_spin.value()),
                        "px_size": 1.0,
                        "binning": 1,
                    },
                }
            )
        else:
            config.update(
                {
                    "h5_file_path": str(self._prepared_h5_path),
                    "pc": pc,
                    "detector_convention": convention,
                    "detector": {
                        "convention": convention,
                        "pc": pc,
                        "sample_tilt": float(self._sample_tilt_spin.value()),
                        "tilt": float(self._camera_tilt_spin.value()),
                        "azimuthal": float(self._azimuth_spin.value()),
                    },
                }
            )
        config_path = output_dir / "workflow_band_width_config.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self._resolved_config_path = config_path
        self._logger.info("Wrote resolved workflow config: %s", config_path)

    def _resolved_output_dir(self) -> Path:
        """Return the selected output directory.

        Returns:
            Output directory path.
        """

        text = self._output_edit.text().strip()
        if text:
            return Path(text).resolve()
        source = Path(self._source_edit.text().strip())
        return (source.parent / "workflow_outputs").resolve()

    def _parse_hkl_list(self) -> list[list[int]]:
        """Parse HKL list text.

        Returns:
            List of HKL triplets.
        """

        value = yaml.safe_load(self._hkl_list_edit.toPlainText())
        if not isinstance(value, list):
            raise ValueError("HKL list must be a list of triplets.")
        parsed = [[int(v) for v in row] for row in value]
        if any(len(row) != 3 for row in parsed):
            raise ValueError("Each HKL entry must contain exactly three integers.")
        return parsed

    def _parse_lattice(self) -> list[float]:
        """Parse lattice constants from the lattice edit.

        Returns:
            Six lattice constants.
        """

        values = self._parse_float_list(self._lattice_edit.text(), expected=6, label="lattice")
        return values

    def _desired_hkl_value(self) -> str:
        """Return the desired HKL in a parser-friendly comma-separated form.

        Returns:
            Desired HKL string.
        """

        text = self._desired_hkl_edit.text().strip()
        if text and "," not in text and " " not in text and len(text) == 3:
            return ",".join(text)
        return text

    def _parse_float_list(self, text: str, *, expected: int, label: str) -> list[float]:
        """Parse a comma-separated float list.

        Parameters:
            text: Source text.
            expected: Expected number of floats.
            label: Field label for errors.

        Returns:
            Parsed floats.
        """

        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if len(values) != expected:
            raise ValueError(f"{label} must contain {expected} values.")
        return values

    def _load_preview(self, path: Optional[Path]) -> None:
        """Load an HDF5/OH5 preview file into the visualization widgets.

        Parameters:
            path: Prepared HDF5/OH5 path.

        Returns:
            None.
        """

        if path is None:
            raise ValueError("No prepared HDF5/OH5 path is available.")
        self._close_datasets()
        self._scan_dataset = OH5ScanFileReader.from_path(path)
        patterns = self._scan_dataset.catalog.list_pattern_fields()
        self._pattern_field = "Pattern" if "Pattern" in patterns else (patterns[0] if patterns else None)
        self._logger.info(
            "Loaded preview: %s (nx=%d, ny=%d, pattern=%s).",
            path,
            self._scan_dataset.nx,
            self._scan_dataset.ny,
            self._pattern_field,
        )
        self._pixel_x_spin.setRange(0, max(0, self._scan_dataset.nx - 1))
        self._pixel_y_spin.setRange(0, max(0, self._scan_dataset.ny - 1))
        self._set_selected_pixel(self._scan_dataset.nx // 2, self._scan_dataset.ny // 2)
        self._ipf_cache.clear()
        self._refresh_map(True)

    def _close_datasets(self) -> None:
        """Close open scan/output datasets."""

        for dataset in (self._scan_dataset, self._output_dataset):
            if dataset is not None:
                try:
                    dataset.close()
                except Exception:
                    pass
        self._scan_dataset = None
        self._output_dataset = None
        self._pattern_field = None

    def _set_selected_pixel(self, x: int, y: int) -> None:
        """Select a pixel and refresh dependent views.

        Parameters:
            x: Column index.
            y: Row index.

        Returns:
            None.
        """

        self._selected_xy = (int(x), int(y))
        self._pixel_x_spin.blockSignals(True)
        self._pixel_y_spin.blockSignals(True)
        self._pixel_x_spin.setValue(int(x))
        self._pixel_y_spin.setValue(int(y))
        self._pixel_x_spin.blockSignals(False)
        self._pixel_y_spin.blockSignals(False)
        for iq_panel, result_panel in self._map_panels.values():
            iq_panel.canvas().set_marker(int(x), int(y))
            result_panel.canvas().set_marker(int(x), int(y))
        self._refresh_pattern(True)
        self._refresh_profile_and_overlay()
        self._inspector_tabs.setCurrentIndex(1)

    def _on_pixel_controls_changed(self, _value: int) -> None:
        """Select the pixel entered in the X/Y spin boxes.

        Parameters:
            _value: Changed spin-box value.

        Returns:
            None.
        """

        if self._scan_dataset is None:
            return
        self._set_selected_pixel(self._pixel_x_spin.value(), self._pixel_y_spin.value())

    def _on_orientation_source_changed(self, _index: int) -> None:
        """Invalidate stale overlays when the runtime Euler source changes.

        Parameters:
            _index: New combo-box index.

        Returns:
            None.
        """

        self._simulated_lines_by_index = {}
        self._orientation_diagnostics = None
        if hasattr(self, "_pattern_panel"):
            self._pattern_panel.canvas().clear_overlay_line()
        if self._scan_dataset is not None:
            self._simulate_preview_lines()

    def _on_pc_dragged(self, pcx: float, pcy: float) -> None:
        """Update the PC edit after dragging the diagnostic marker.

        Parameters:
            pcx: Normalized detector PC X coordinate.
            pcy: Normalized detector PC Y coordinate.

        Returns:
            None.
        """

        pc = self._parse_float_list(self._pc_edit.text(), expected=3, label="PC")
        pc[0], pc[1] = float(pcx), float(pcy)
        self._pc_edit.setText(", ".join(f"{value:.6f}" for value in pc))
        self._logger.info("Updated diagnostic PC to x*=%.6f, y*=%.6f.", pcx, pcy)
        if self._auto_resolve_checkbox.isChecked():
            self._auto_solve_timer.start()

    def _single_pattern_config(self) -> SinglePatternConfig:
        """Build an in-memory single-pattern diagnostic configuration.

        Returns:
            Configuration using the current source, pixel, phase, and detector controls.
        """

        x = int(self._pixel_x_spin.value())
        y = int(self._pixel_y_spin.value())
        input_cfg: dict[str, Any]
        if self._current_mode() == "ctf":
            input_cfg = {
                "type": "ctf",
                "ctf_path": self._source_edit.text().strip(),
                "pattern_dir": self._pattern_edit.text().strip(),
                "pattern_template": self._template_edit.text().strip() or None,
                "x": x,
                "y": y,
            }
        else:
            input_cfg = {
                "type": "oh5",
                "path": self._source_edit.text().strip(),
                "pattern_field": self._pattern_field or "Pattern",
                "x": x,
                "y": y,
            }
        phase_name = self._phase_name_edit.text().strip() or "Ni"
        raw = {
            "input": input_cfg,
            "phase": {
                "name": phase_name,
                "space_group": int(self._space_group_spin.value()),
                "lattice": self._parse_lattice(),
                "atoms": [{"element": phase_name, "position": [0, 0, 0]}],
            },
            "detector": {
                "convention": self._detector_convention_combo.currentText(),
                "pc": self._parse_float_list(self._pc_edit.text(), expected=3, label="PC"),
                "sample_tilt": float(self._sample_tilt_spin.value()),
                "tilt": float(self._camera_tilt_spin.value()),
                "azimuthal": float(self._azimuth_spin.value()),
                "px_size": 1.0,
                "binning": 1,
            },
            "orientation": {"direction": "lab2crystal"},
            "simulation": {"hkl_list": self._parse_hkl_list()},
            "band_profile": {
                "desired_hkl": self._desired_hkl_value(),
                "rectWidth": int(self._rect_width_spin.value()),
                "min_psnr": float(self._min_psnr_spin.value()),
                "smoothing_sigma": 2.0,
            },
            "hough": {
                "enabled": True,
                "use_indexed_orientation": self._orientation_source_combo.currentData()
                == "indexed",
                "n_bands": 5,
                "t_sigma": 2,
                "r_sigma": 2,
            },
        }
        config_path = Path(self._config_edit.text().strip() or "interactive_workflow.yml")
        return SinglePatternConfig(path=config_path, raw=raw)

    def _solve_selected_pattern(self) -> None:
        """Run the reusable single-pattern diagnostic solver and render it."""

        try:
            solution = solve_single_pattern(self._single_pattern_config(), logger=self._logger)
            self._single_solution = solution
            self._single_pattern_canvas.update_solution(solution)
            self._inspector_tabs.setCurrentIndex(0)
            summary = solution.hough_summary or {}
            self._logger.info(
                "Diagnostic solve x=%d y=%d; Hough success=%s; exported orientations remain unchanged.",
                solution.x,
                solution.y,
                summary.get("success", False),
            )
        except Exception as exc:
            self._logger.exception("Selected-pattern diagnostic solve failed: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Diagnostic solve failed", str(exc))

    def _on_map_click(self, event: Any) -> None:
        """Select a pixel from a map click.

        Parameters:
            event: Matplotlib mouse event.

        Returns:
            None.
        """

        dataset = self._output_dataset or self._scan_dataset
        if dataset is None or event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if 0 <= x < dataset.nx and 0 <= y < dataset.ny:
            self._set_selected_pixel(x, y)

    def _on_map_hover(self, event: Any) -> None:
        """Update cursor label from map hover.

        Parameters:
            event: Matplotlib mouse event.

        Returns:
            None.
        """

        if event.xdata is None or event.ydata is None:
            self._cursor_label.setText("Cursor: --")
        else:
            self._cursor_label.setText(f"Cursor: {event.xdata:.1f}, {event.ydata:.1f}")

    def _refresh_map(self, reset_view: bool = True) -> None:
        """Refresh every linked IQ/result map tab.

        Parameters:
            reset_view: Whether to reset axes limits.

        Returns:
            None.
        """

        dataset = self._output_dataset or self._scan_dataset
        if dataset is None:
            for iq_panel, result_panel in self._map_panels.values():
                iq_panel.canvas().update_data(np.zeros((2, 2), dtype=np.float32), reset_view=True)
                result_panel.canvas().update_data(np.zeros((2, 2), dtype=np.float32), reset_view=True)
            return
        try:
            iq = dataset.get_map("IQ")
        except Exception:
            iq = np.zeros((dataset.ny, dataset.nx), dtype=np.float32)
        for field, (iq_panel, result_panel) in self._map_panels.items():
            self._update_map_panel(iq_panel, iq, field="IQ", reset_view=reset_view)
            try:
                data = self._ipf_map(field) if field.startswith("IPF-") else dataset.get_map(field)
            except Exception:
                data = np.zeros((dataset.ny, dataset.nx), dtype=np.float32)
            self._update_map_panel(result_panel, data, field=field, reset_view=reset_view)
        if self._selected_xy is not None:
            x, y = self._selected_xy
            for iq_panel, result_panel in self._map_panels.values():
                iq_panel.canvas().set_marker(x, y)
                result_panel.canvas().set_marker(x, y)

    def _update_map_panel(
        self,
        panel: MapPanel,
        data: np.ndarray,
        *,
        field: str,
        reset_view: bool,
    ) -> None:
        """Display scalar or RGB data with appropriate contrast handling.

        Parameters:
            panel: Destination map panel.
            data: Scalar or RGB map array.
            field: Scientific field name used to resolve display settings.
            reset_view: Whether axes limits should be reset.

        Returns:
            None.
        """

        array = np.asarray(data)
        if array.ndim == 3 and array.shape[-1] in {3, 4}:
            panel.canvas().update_data(array, reset_view=reset_view)
            return
        settings = self._map_display_settings.setdefault(
            field, self._default_map_display_settings(field)
        )
        if settings.range_mode == "auto":
            low, high = panel.contrast_values()
            settings.percentile_low = float(low)
            settings.percentile_high = float(high)
        try:
            norm, cmap = settings.render_parameters(array)
            panel.clear_error()
        except ValueError as exc:
            panel.set_error(str(exc))
            fallback = self._default_map_display_settings(field)
            norm, cmap = fallback.render_parameters(array)
        panel.canvas().update_data(
            array, cmap=cmap, norm=norm, reset_view=reset_view
        )

    def _default_map_display_settings(self, field: str) -> MapDisplaySettings:
        """Return scientifically useful default display settings for a field.

        Parameters:
            field: Scalar field name.

        Returns:
            New settings instance.
        """

        if field == "IQ":
            return MapDisplaySettings(colormap="gray")
        if field in {"band_valid", "orientation_fallback", "indexing_success"}:
            return MapDisplaySettings(
                range_mode="manual", minimum=0.0, maximum=1.0, colormap="gray"
            )
        if field in {"strain", "stress"}:
            return MapDisplaySettings(
                scale="symlog",
                symmetric=True,
                linthresh=1.0e-3 if field == "strain" else 1.0e6,
                colormap="coolwarm",
                invalid_color="#808080",
            )
        return MapDisplaySettings(colormap="viridis")

    def _edit_map_display(self, field: str) -> None:
        """Open the scalar-map display settings dialog.

        Parameters:
            field: Scalar field being edited.

        Returns:
            None.
        """

        current = self._map_display_settings.setdefault(
            field, self._default_map_display_settings(field)
        )
        dialog = MapDisplaySettingsDialog(
            field,
            current,
            defaults=self._default_map_display_settings(field),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._map_display_settings[field] = dialog.settings()
        settings = self._map_display_settings[field]
        for map_field, (iq_panel, result_panel) in self._map_panels.items():
            target = iq_panel if field == "IQ" else result_panel if map_field == field else None
            if target is not None and settings.range_mode == "auto":
                target.set_contrast_values(
                    settings.percentile_low, settings.percentile_high, block_signals=True
                )
        self._logger.info("Updated %s map display settings: %s", field, settings.to_mapping())
        self._refresh_map(False)

    def _ipf_map(self, field: str) -> np.ndarray:
        """Return an IPF RGB map computed from original acquisition Euler angles.

        Parameters:
            field: One of ``IPF-X``, ``IPF-Y``, or ``IPF-Z``.

        Returns:
            RGB map shaped ``(ny, nx, 3)``.
        """

        if field in self._ipf_cache:
            return self._ipf_cache[field]
        dataset = self._output_dataset or self._scan_dataset
        if dataset is None:
            raise RuntimeError("No scan is loaded.")
        eulers = self._read_preview_eulers()
        phase = self._build_phase()
        orientations = Orientation.from_euler(
            eulers,
            symmetry=phase.point_group,
            direction="lab2crystal",
            degrees=False,
        )
        directions = {
            "IPF-X": Vector3d.xvector(),
            "IPF-Y": Vector3d.yvector(),
            "IPF-Z": Vector3d.zvector(),
        }
        rgb = plot.IPFColorKeyTSL(phase.point_group, directions[field]).orientation2color(orientations)
        result = np.asarray(rgb).reshape(dataset.ny, dataset.nx, 3)
        self._ipf_cache[field] = result
        return result

    def _refresh_pattern(self, reset_view: bool = True) -> None:
        """Refresh the selected pattern image.

        Parameters:
            reset_view: Whether to reset axes limits.

        Returns:
            None.
        """

        dataset = self._output_dataset or self._scan_dataset
        if dataset is None or self._pattern_field is None or self._selected_xy is None:
            self._pattern_panel.canvas().update_data(np.zeros((16, 16), dtype=np.float32), reset_view=True)
            return
        x, y = self._selected_xy
        pattern = dataset.get_pattern(self._pattern_field, x, y)
        if pattern is None:
            self._pattern_panel.canvas().update_data(np.zeros((16, 16), dtype=np.float32), reset_view=True)
            return
        low, high = self._pattern_panel.contrast_values()
        finite = pattern[np.isfinite(pattern)]
        vmin, vmax = (float(np.percentile(finite, low)), float(np.percentile(finite, high))) if finite.size else (0.0, 1.0)
        if vmin == vmax:
            vmax = vmin + 1.0
        self._pattern_panel.canvas().update_data(pattern, cmap="gray", vmin=vmin, vmax=vmax, reset_view=reset_view)
        if self._overlay_checkbox.isChecked():
            self._apply_selected_pattern_overlay()

    def _refresh_profile_and_overlay(self) -> None:
        """Refresh band profile and overlay from output datasets."""

        if self._selected_xy is None:
            return
        dataset = self._output_dataset or self._scan_dataset
        x, y = self._selected_xy
        if dataset is None or "band_profile" not in dataset.catalog.vectors:
            self._profile_plot.clear("Run analysis to populate band_profile.")
            if self._overlay_checkbox.isChecked():
                self._apply_selected_pattern_overlay()
            else:
                self._pattern_panel.canvas().clear_overlay_line()
            self._metrics_label.setText("Band metrics: not available")
            return
        payload = extract_band_profile_payload(dataset, x, y, logger=self._logger)
        if payload.profile is None or not np.isfinite(payload.profile).any():
            self._profile_plot.clear("No finite band profile at this pixel.")
            if self._overlay_checkbox.isChecked():
                self._apply_selected_pattern_overlay()
            else:
                self._pattern_panel.canvas().clear_overlay_line()
            self._metrics_label.setText("Band metrics: no finite profile")
            return
        self._profile_plot.update_plot(payload, None, normalize=True, show_markers=True)
        if self._overlay_checkbox.isChecked():
            self._apply_selected_pattern_overlay(payload)
        else:
            self._pattern_panel.canvas().clear_overlay_line()
        metrics = self._metrics_text(dataset, x, y)
        if not payload.band_valid:
            metrics += " (selected profile is marked invalid)"
        self._metrics_label.setText(metrics)

    def _apply_selected_pattern_overlay(
        self,
        payload: Optional[BandProfilePayload] = None,
    ) -> None:
        """Render simulated lines and the selected measured profile line together.

        Parameters:
            payload: Optional selected-pixel band profile payload.

        Returns:
            None.
        """

        canvas = self._pattern_panel.canvas()
        if not self._overlay_checkbox.isChecked():
            canvas.clear_overlay_line()
            return
        lines = self._selected_simulated_lines()
        if payload is not None and payload.central_line is not None:
            central_line = np.asarray(payload.central_line, dtype=np.float32).ravel()
            if central_line.size >= 4 and np.isfinite(central_line[:4]).all():
                lines.append(
                    {
                        "hkl": f"Profile {{{self._desired_hkl_value()}}}",
                        "central_line": central_line[:4].tolist(),
                        "color": "#ffeb3b",
                        "linewidth": 4.0,
                        "show_label": True,
                        "label_fraction": 0.42,
                        "alpha": 0.95,
                    }
                )
        if lines and hasattr(canvas, "set_overlay_lines"):
            canvas.set_overlay_lines(lines, color="#00e676", linewidth=1.8, show_labels=True)
        else:
            canvas.clear_overlay_line()

    def _simulate_preview_lines(self) -> None:
        """Simulate solved-orientation Kikuchi lines for the prepared preview data.

        Returns:
            None.
        """

        self._simulated_lines_by_index = {}
        if self._scan_dataset is None or self._pattern_field is None:
            return
        if self._orientation_source_combo.currentData() == "indexed":
            self._logger.info(
                "Preview acquisition-Euler overlays are suppressed because live indexed "
                "orientation mode is selected. Use Solve Selected Pattern; the full run "
                "will populate indexed/fallback overlays for every pixel."
            )
            return
        try:
            eulers = self._read_preview_eulers()
            phase = self._build_phase()
            pattern = self._first_available_pattern()
            detector_cfg = self._detector_config()
            detector = kp.detectors.EBSDDetector(
                shape=tuple(int(value) for value in pattern.shape),
                sample_tilt=float(detector_cfg["sample_tilt"]),
                tilt=float(detector_cfg["tilt"]),
                azimuthal=float(detector_cfg["azimuthal"]),
                convention=str(detector_cfg["convention"]),
                pc=tuple(float(value) for value in detector_cfg["pc"]),
            )
            rotations = Rotation.from_euler(eulers, direction="lab2crystal", degrees=False)
            rotations = rotations.reshape(self._scan_dataset.ny, self._scan_dataset.nx)
            reflectors = ReciprocalLatticeVector(phase=phase, hkl=self._parse_hkl_list()).symmetrise()
            simulator = CustomKikuchiPatternSimulator(reflectors)
            simulation = simulator.on_detector(detector, rotations)
            simulation.phase = phase
            self._simulated_lines_by_index = self._extract_simulated_lines(simulation, phase)
            line_count = sum(len(lines) for lines in self._simulated_lines_by_index.values())
            self._logger.info(
                "Simulated %d principal-family Kikuchi line overlays using %s PC convention.",
                line_count,
                detector_cfg["convention"],
            )
            self._draw_simulated_lines_for_selected_pixel()
        except Exception as exc:
            self._simulated_lines_by_index = {}
            self._logger.exception("Failed to simulate preview Kikuchi lines: %s", exc)

    def _read_preview_eulers(self) -> np.ndarray:
        """Read Euler angles from the preview HDF5/OH5 scalar fields.

        Returns:
            Euler angle array in radians with shape ``(n_pixels, 3)``.
        """

        if self._scan_dataset is None:
            raise RuntimeError("No preview dataset is loaded.")
        radians_fields = ("Phi1", "Phi", "Phi2")
        degrees_fields = ("Euler1", "Euler2", "Euler3")
        if all(field in self._scan_dataset.catalog.scalars for field in radians_fields):
            columns = [self._scan_dataset.get_map(field).reshape(-1) for field in radians_fields]
            return np.vstack(columns).T.astype(np.float64)
        if all(field in self._scan_dataset.catalog.scalars for field in degrees_fields):
            columns = [self._scan_dataset.get_map(field).reshape(-1) for field in degrees_fields]
            return np.deg2rad(np.vstack(columns).T.astype(np.float64))
        raise KeyError("Preview dataset does not contain Phi1/Phi/Phi2 or Euler1/Euler2/Euler3 fields.")

    def _build_phase(self) -> Phase:
        """Build the configured phase for solved-orientation line simulation.

        Returns:
            Orix phase instance.
        """

        name = self._phase_name_edit.text().strip() or "Cr"
        return Phase(
            name=name,
            space_group=int(self._space_group_spin.value()),
            structure=Structure(
                lattice=Lattice(*self._parse_lattice()),
                atoms=[Atom(name, [0, 0, 0])],
            ),
        )

    def _first_available_pattern(self) -> np.ndarray:
        """Return the first pattern image available in the prepared preview.

        Returns:
            Pattern image array.
        """

        if self._scan_dataset is None or self._pattern_field is None:
            raise RuntimeError("No preview pattern field is loaded.")
        for y in range(self._scan_dataset.ny):
            for x in range(self._scan_dataset.nx):
                pattern = self._scan_dataset.get_pattern(self._pattern_field, x, y)
                if pattern is not None:
                    return np.asarray(pattern)
        raise RuntimeError("No pattern image is available in the prepared preview.")

    def _detector_config(self) -> dict[str, object]:
        """Return detector settings from the GUI controls.

        Returns:
            Detector configuration dictionary.
        """

        return {
            "convention": self._detector_convention_combo.currentText(),
            "pc": self._parse_float_list(self._pc_edit.text(), expected=3, label="PC"),
            "sample_tilt": float(self._sample_tilt_spin.value()),
            "tilt": float(self._camera_tilt_spin.value()),
            "azimuthal": float(self._azimuth_spin.value()),
        }

    def _extract_simulated_lines(
        self,
        simulation: Any,
        phase: Phase,
    ) -> dict[int, list[dict[str, object]]]:
        """Extract visible principal-family Kikuchi line segments from a simulation.

        Parameters:
            simulation: Geometrical Kikuchi pattern simulation.
            phase: Crystal phase used to group equivalent HKLs.

        Returns:
            Row-major pixel index to line dictionaries.
        """

        coords = np.asarray(simulation.lines_coordinates(index=(), exclude_nan=False), dtype=np.float64)
        coords = np.around(coords, 3)
        reflectors = simulation._reflectors.coordinates.round().astype(int)
        rows, cols, n_lines, _ = coords.shape
        grouped: dict[int, list[dict[str, object]]] = {}
        detector_shape = tuple(float(value) for value in simulation.detector.shape)
        detector_mid = np.array([0.5 * detector_shape[1], 0.5 * detector_shape[0]])
        distance_threshold = 0.95 * 0.5 * min(detector_shape)
        family_specs = self._principal_hkl_families(phase)
        for line_index in range(n_lines):
            hkl = " ".join(str(int(value)) for value in reflectors[line_index])
            family_label, family_color = self._line_family_style(hkl, family_specs)
            if family_label is None:
                continue
            for row in range(rows):
                for col in range(cols):
                    line = coords[row, col, line_index, :]
                    if not np.isfinite(line).all():
                        continue
                    midpoint = np.array([0.5 * (line[0] + line[2]), 0.5 * (line[1] + line[3])])
                    is_near_center = float(np.linalg.norm(midpoint - detector_mid)) < distance_threshold
                    if not is_near_center:
                        continue
                    pixel_index = row * cols + col
                    entry = {
                        "hkl": family_label,
                        "reflector": hkl,
                        "color": family_color,
                        "central_line": line.tolist(),
                        "line_mid_xy": midpoint.tolist(),
                    }
                    grouped.setdefault(pixel_index, []).append(entry)
        return grouped

    def _principal_hkl_families(self, phase: Phase) -> list[dict[str, object]]:
        """Return configured principal HKL families with display styles.

        Parameters:
            phase: Crystal phase used for symmetry-aware HKL grouping.

        Returns:
            List of family dictionaries.
        """

        colors = ["#00e676", "#ffdd33", "#40c4ff", "#ff6d00", "#e040fb", "#ffffff"]
        families: list[dict[str, object]] = []
        for index, hkl in enumerate(self._parse_hkl_list()):
            hkl_text = ",".join(str(int(value)) for value in hkl)
            label = "{" + "".join(str(abs(int(value))) for value in hkl) + "}"
            families.append(
                {
                    "hkl": hkl_text,
                    "label": label,
                    "phase": phase,
                    "color": colors[index % len(colors)],
                }
            )
        return families

    def _line_family_style(
        self,
        hkl: str,
        families: list[dict[str, object]],
    ) -> tuple[Optional[str], str]:
        """Return the display label and color for a simulated reflector.

        Parameters:
            hkl: Simulated reflector string.
            families: Principal family definitions.

        Returns:
            Label and color. Label is ``None`` if the reflector is not in a
            configured family.
        """

        for family in families:
            try:
                belongs, _ = ut.belongs_to_group(hkl, family["hkl"], phase=family["phase"])
            except Exception:
                belongs = False
            if belongs:
                return str(family["label"]), str(family["color"])
        return None, "#00e676"

    def _draw_simulated_lines_for_selected_pixel(self) -> None:
        """Draw simulated solved-orientation Kikuchi lines for the selected pixel.

        Returns:
            None.
        """

        lines = self._selected_simulated_lines()
        canvas = self._pattern_panel.canvas()
        if lines and hasattr(canvas, "set_overlay_lines"):
            canvas.set_overlay_lines(lines, color="#00e676", linewidth=1.8, show_labels=True)
        else:
            canvas.clear_overlay_line()

    def _selected_simulated_lines(self) -> list[dict[str, object]]:
        """Return display-ready simulated lines for the selected pixel.

        Returns:
            List of overlay line dictionaries.
        """

        if self._selected_xy is None:
            return []
        dataset = self._output_dataset or self._scan_dataset
        if dataset is None:
            return []
        x, y = self._selected_xy
        pixel_index = int(y) * int(dataset.nx) + int(x)
        lines = self._simulated_lines_by_index.get(pixel_index, [])
        return self._prepare_overlay_labels(lines) if lines else []

    def _prepare_overlay_labels(
        self,
        lines: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Choose sparse, deterministic labels for simulated overlay lines.

        Parameters:
            lines: Simulated line dictionaries.

        Returns:
            Line dictionaries with label visibility and label positions.
        """

        counts_by_family: dict[str, int] = {}
        prepared: list[dict[str, object]] = []
        for index, line in enumerate(lines):
            item = dict(line)
            family = str(item.get("hkl", ""))
            count = counts_by_family.get(family, 0)
            item.setdefault("show_label", count < 2)
            counts_by_family[family] = count + 1
            item.setdefault(
                "label_fraction", 0.12 if (index + count) % 2 == 0 else 0.88
            )
            prepared.append(item)
        return prepared

    def _metrics_text(self, dataset: Any, x: int, y: int) -> str:
        """Format output metrics for a pixel.

        Parameters:
            dataset: Scan dataset.
            x: Column index.
            y: Row index.

        Returns:
            Human-readable metric string.
        """

        fields = [
            "Band_Width",
            "psnr",
            "band_valid",
            "strain",
            "stress",
            "band_intensity_ratio",
            "indexing_success",
            "orientation_fallback",
            "indexing_fit",
            "indexing_confidence",
        ]
        parts = []
        for field in fields:
            try:
                parts.append(f"{field}={dataset.get_scalar(field, x, y):.4g}")
            except Exception:
                continue
        return "Band metrics: " + (", ".join(parts) if parts else "not available")

    def _start_run(self) -> None:
        """Start the background automator worker."""

        if self._prepared_h5_path is None and self._prepared_oh5_path is None:
            QtWidgets.QMessageBox.warning(self, "Not prepared", "Prepare the inputs first.")
            return
        try:
            self._write_resolved_config()
        except Exception as exc:
            self._logger.exception("Failed to resolve current run settings: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Invalid run settings", str(exc))
            return
        self._logger.info("Starting workflow analysis with %s", self._resolved_config_path)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress.setValue(0)
        worker = AutomatorWorker(self._resolved_config_path, parent=self)
        worker.stage_changed.connect(self._on_worker_stage)
        worker.progress_changed.connect(self._on_worker_progress)
        worker.pixel_changed.connect(self._on_worker_pixel)
        worker.pixel_result.connect(self._on_worker_pixel_result)
        worker.visualization_ready.connect(self._on_worker_visualization_ready)
        worker.finished_success.connect(self._on_worker_finished)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _cancel_run(self) -> None:
        """Request cancellation from the worker."""

        if self._worker is not None:
            self._worker.request_cancel()
            self._cancel_button.setEnabled(False)

    def _stop_worker(self) -> None:
        """Stop the worker if it is still running."""

        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(2000)

    def _on_worker_stage(self, stage: str, index: int, total: int) -> None:
        """Update stage label from worker signal."""

        self._stage_label.setText(f"Stage: {stage} ({index}/{total})")

    def _on_worker_progress(self, processed: int, total: int, eta_seconds: float) -> None:
        """Update progress bar from worker signal."""

        self._progress.setValue(int(round(100.0 * processed / max(1, total))))
        self._eta_label.setText(f"ETA: {eta_seconds:.1f}s")

    def _on_worker_pixel(self, x: int, y: int, processed: int) -> None:
        """Update live current-pixel marker from worker signal."""

        self._pixel_label.setText(f"Pixel: X={x}, Y={y}")
        for iq_panel, result_panel in self._map_panels.values():
            iq_panel.canvas().set_secondary_marker(x, y, color="#ff0000")
            result_panel.canvas().set_secondary_marker(x, y, color="#ff0000")

    def _on_worker_pixel_result(self, payload: object) -> None:
        """Render a throttled live pattern, overlay, and profile during a scan.

        Parameters:
            payload: Worker dictionary containing pattern, annotations, and result entry.

        Returns:
            None.
        """

        if not isinstance(payload, dict):
            return
        if self._live_render_clock.elapsed() < 200 and int(payload.get("processed", 0)) > 1:
            return
        self._live_render_clock.restart()
        pattern = np.asarray(payload.get("pattern"), dtype=np.float32)
        if pattern.ndim != 2:
            return
        finite = pattern[np.isfinite(pattern)]
        vmin, vmax = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) if finite.size else (0.0, 1.0)
        self._pattern_panel.canvas().update_data(pattern, cmap="gray", vmin=vmin, vmax=vmax, reset_view=True)
        annotations = payload.get("annotations", {})
        lines = list(annotations.get("points", [])) if isinstance(annotations, dict) else []
        entry = payload.get("entry", {})
        bands = entry.get("bands", []) if isinstance(entry, dict) else []
        best = self._select_display_band(bands)
        if best is not None:
            profile = np.asarray(best.get("band_profile"), dtype=np.float32)
            central_line = np.asarray(best.get("central_line"), dtype=np.float32)
            profile_payload = BandProfilePayload(
                profile=profile,
                central_line=central_line,
                band_start_idx=int(best.get("band_start_idx", best.get("bandStart", -1))),
                central_peak_idx=int(best.get("central_peak_idx", best.get("centralPeak", -1))),
                band_end_idx=int(best.get("band_end_idx", best.get("bandEnd", -1))),
                profile_length=int(best.get("profile_length", profile.size)),
                band_valid=bool(best.get("band_valid")),
            )
            self._profile_plot.update_plot(profile_payload, None, normalize=True, show_markers=True)
            if central_line.size >= 4 and np.isfinite(central_line[:4]).all():
                lines.append(
                    {
                        "hkl": f"Profile {{{self._desired_hkl_value()}}}",
                        "central_line": central_line[:4].tolist(),
                        "color": "#ffeb3b",
                        "linewidth": 4.0,
                        "show_label": True,
                        "label_fraction": 0.42,
                    }
                )
        if lines:
            self._pattern_panel.canvas().set_overlay_lines(
                self._prepare_overlay_labels(lines),
                color="#00e676",
                linewidth=1.8,
                show_labels=True,
            )
        orientation = payload.get("orientation", {})
        if isinstance(orientation, dict):
            source = str(orientation.get("source", "unknown"))
            fallback = bool(orientation.get("fallback", False))
            fit = orientation.get("fit")
            confidence = orientation.get("confidence")
            self._metrics_label.setText(
                f"Orientation: {source}"
                + (" (acquisition fallback)" if fallback else "")
                + f"; fit={fit!s}; confidence={confidence!s}"
            )
        self._inspector_tabs.setCurrentIndex(1)

    def _on_worker_visualization_ready(self, payload: object) -> None:
        """Store completed runtime line annotations for post-run inspection.

        Parameters:
            payload: Worker visualization dictionary.

        Returns:
            None.
        """

        if not isinstance(payload, dict):
            return
        annotations = payload.get("annotations")
        if isinstance(annotations, list):
            converted: dict[int, list[dict[str, object]]] = {}
            for index, entry in enumerate(annotations):
                if isinstance(entry, dict):
                    points = entry.get("points", [])
                    if isinstance(points, list):
                        converted[index] = [dict(point) for point in points if isinstance(point, dict)]
            self._simulated_lines_by_index = converted
        self._orientation_diagnostics = payload.get("diagnostics")
        self._logger.info(
            "Loaded completed %s runtime overlay cache for %d pixels.",
            payload.get("orientation_source", "unknown"),
            len(self._simulated_lines_by_index),
        )

    def _select_display_band(self, bands: object) -> Optional[dict[str, object]]:
        """Choose the most useful band dictionary for GUI display.

        Parameters:
            bands: Sequence-like object from the batch processor result.

        Returns:
            Band dictionary with a finite profile, or ``None`` when unavailable.
        """

        if not isinstance(bands, list):
            return None
        candidates: list[dict[str, object]] = []
        for band in bands:
            if not isinstance(band, dict):
                continue
            profile = np.asarray(band.get("band_profile"), dtype=np.float32).ravel()
            if profile.size and np.isfinite(profile).any():
                candidates.append(band)
        if not candidates:
            return None
        valid = [band for band in candidates if band.get("band_valid")]
        ranked = valid or candidates
        return max(ranked, key=lambda band: float(band.get("psnr", 0.0) or 0.0))

    def _on_worker_finished(self, output_path: str, summary: object) -> None:
        """Load completed output and refresh GUI.

        Parameters:
            output_path: Modified HDF5 path.
            summary: Summary object emitted by worker.

        Returns:
            None.
        """

        self._logger.info("Workflow analysis completed: %s", output_path)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._progress.setValue(100)
        self._output_label.setText(f"Output: {output_path}")
        self._open_output_button.setEnabled(True)
        if self._output_dataset is not None:
            self._output_dataset.close()
        self._output_dataset = OH5ScanFileReader.from_path(Path(output_path))
        patterns = self._output_dataset.catalog.list_pattern_fields()
        if patterns:
            self._pattern_field = "Pattern" if "Pattern" in patterns else patterns[0]
        self._ipf_cache.clear()
        self._refresh_map(True)
        self._refresh_pattern(True)
        self._refresh_profile_and_overlay()
        self._export_map_images(Path(output_path))
        self._summary.appendPlainText(f"\nAnalysis complete.\n{summary}")

    def _export_map_images(self, output_path: Path) -> None:
        """Export rendered IQ and result map tabs as PNG images.

        Parameters:
            output_path: Completed modified HDF5 path used to derive file names.

        Returns:
            None.
        """

        destination = Path(output_path).parent
        base_name = Path(output_path).stem
        exported_iq = False
        for field, (iq_panel, result_panel) in self._map_panels.items():
            safe_field = field.lower().replace("-", "_").replace(" ", "_")
            if not exported_iq:
                iq_panel.canvas().figure.savefig(
                    destination / f"{base_name}_iq_map.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                exported_iq = True
            result_panel.canvas().figure.savefig(
                destination / f"{base_name}_{safe_field}_map.png",
                dpi=150,
                bbox_inches="tight",
            )
        self._logger.info("Exported IQ and result map PNG files to %s.", destination)

    def _on_worker_cancelled(self, message: str) -> None:
        """Handle worker cancellation."""

        self._logger.warning("Workflow cancelled: %s", message)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._stage_label.setText("Stage: cancelled")

    def _on_worker_failed(self, message: str) -> None:
        """Handle worker failure."""

        self._logger.error("Workflow failed: %s", message)
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._stage_label.setText("Stage: failed")
        QtWidgets.QMessageBox.critical(self, "Analysis failed", message)

    def _open_output_folder(self) -> None:
        """Open the current output folder."""

        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._resolved_output_dir())))

    def snapshot(self, output_path: Path) -> None:
        """Save a screenshot of the GUI.

        Parameters:
            output_path: Destination PNG path.

        Returns:
            None.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self.grab()
        pixmap.save(str(output_path))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the workflow GUI command-line parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description="Run the Kikuchi workflow GUI.")
    parser.add_argument("--config", type=Path, help="Optional workflow or single-pattern YAML file.")
    parser.add_argument("--mode", choices=["ctf", "tsl"], default="ctf")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--patterns", type=Path)
    parser.add_argument("--ang", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    """Run the workflow GUI application."""

    args = build_arg_parser().parse_args()
    configure_logging(False, None)
    app = QtWidgets.QApplication([])
    window = WorkflowGuiMainWindow(
        input_mode=args.mode,
        source_path=args.source,
        pattern_dir=args.patterns,
        ang_path=args.ang,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
