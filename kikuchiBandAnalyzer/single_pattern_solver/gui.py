"""PySide GUI for live single-pattern EBSP geometry debugging."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Callable, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.readers.ctf_reader import CtfPatternScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.single_pattern_solver.solver import (
    SinglePatternConfig,
    SinglePatternSolution,
    _draw_lines_on_axes,
    _draw_profile_on_axes,
    load_single_pattern_config,
    render_solution,
    solve_single_pattern,
    write_solution_json,
)

DISPLAY_HOUGH_PEAK_LIMIT = 5


class TextEditLogHandler(logging.Handler):
    """Logging handler that appends records to a QTextEdit."""

    def __init__(self, widget: QtWidgets.QTextEdit) -> None:
        """Initialize the handler.

        Parameters:
            widget: Target text edit widget.
        """

        super().__init__()
        self._widget: Optional[QtWidgets.QTextEdit] = widget

    def emit(self, record: logging.LogRecord) -> None:
        """Append one formatted log record to the widget.

        Parameters:
            record: Log record.

        Returns:
            None.
        """

        if self._widget is None:
            return
        message = self.format(record)
        try:
            self._widget.append(message)
        except RuntimeError:
            self._widget = None

    def detach(self) -> None:
        """Detach the target widget before it is destroyed."""

        self._widget = None


class SinglePatternCanvas(FigureCanvas):
    """Matplotlib canvas showing an EBSP overlay and band profile."""

    def __init__(self, pc_changed_callback: Optional[Callable[[float, float], None]] = None) -> None:
        """Initialize the canvas.

        Parameters:
            pc_changed_callback: Optional callback receiving normalized
                ``(pcx, pcy)`` after the PC marker is dragged.
        """

        self._figure = Figure(figsize=(10.5, 6.2))
        grid = self._figure.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
        self._pattern_axes = self._figure.add_subplot(grid[:, 0])
        self._profile_axes = self._figure.add_subplot(grid[0, 1])
        self._hough_axes = self._figure.add_subplot(grid[1, 1])
        self._pc_changed_callback = pc_changed_callback
        self._dragging_pc = False
        self._pc_artist = None
        super().__init__(self._figure)
        self.mpl_connect("button_press_event", self._on_button_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_button_release)

    def update_solution(self, solution: Optional[SinglePatternSolution]) -> None:
        """Render a solution or empty state.

        Parameters:
            solution: Optional single-pattern solution.

        Returns:
            None.
        """

        self._pattern_axes.clear()
        self._profile_axes.clear()
        self._hough_axes.clear()
        self._pc_artist = None
        if solution is None:
            self._pattern_axes.text(0.5, 0.5, "Load config", ha="center", va="center")
            self._profile_axes.text(0.5, 0.5, "No profile", ha="center", va="center")
            self._hough_axes.text(0.5, 0.5, "No Hough data", ha="center", va="center")
        else:
            self._pattern_axes.imshow(solution.pattern, cmap="gray")
            self._pattern_axes.set_title("EBSP: dotted Hough bands + solid simulated Kikuchi lines")
            self._pattern_axes.set_xticks([])
            self._pattern_axes.set_yticks([])
            self._draw_experimental_hough_lines(solution)
            _draw_lines_on_axes(self._pattern_axes, solution.pattern.shape, solution.lines)
            self._draw_selected_profile_band(solution)
            self._draw_pc_marker(solution)
            _draw_profile_on_axes(self._profile_axes, solution.profile_payload)
            self._draw_hough_panel(solution)
        self._figure.tight_layout()
        self.draw_idle()

    def _draw_pc_marker(self, solution: SinglePatternSolution) -> None:
        """Draw the draggable pattern-center marker.

        Parameters:
            solution: Current single-pattern solution.

        Returns:
            None.
        """

        height, width = solution.pattern.shape
        pc = solution.detector_summary.get("pc", [0.5, 0.5, 0.5])
        pcx = float(pc[0]) * (width - 1)
        pcy = float(pc[1]) * (height - 1)
        self._pc_artist = self._pattern_axes.scatter(
            [pcx],
            [pcy],
            marker="+",
            s=140,
            linewidths=2.2,
            color="#ff1744",
            zorder=8,
            label="PC",
        )
        self._pattern_axes.annotate(
            "PC",
            xy=(pcx, pcy),
            xytext=(6, -6),
            textcoords="offset points",
            color="#ff1744",
            fontsize=8,
            fontweight="bold",
            zorder=9,
        )

    def _draw_experimental_hough_lines(self, solution: SinglePatternSolution) -> None:
        """Draw backend-selected Hough bands inverted onto the EBSP.

        Parameters:
            solution: Current single-pattern solution.

        Returns:
            None.
        """

        diagnostic = solution.hough_diagnostic
        if diagnostic is None:
            return
        for peak in diagnostic.peaks[:DISPLAY_HOUGH_PEAK_LIMIT]:
            if peak.line is None:
                continue
            x1, y1, x2, y2 = peak.line
            color = "#ff1744" if peak.valid else "#9e9e9e"
            self._pattern_axes.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=1.1,
                linestyle=":",
                alpha=0.9,
                zorder=5,
                clip_on=True,
            )
            label_x = x1 + 0.82 * (x2 - x1)
            label_y = y1 + 0.82 * (y2 - y1)
            self._pattern_axes.text(
                label_x,
                label_y,
                str(peak.rank),
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 1.2},
                zorder=7,
                clip_on=True,
            )

    def _draw_selected_profile_band(self, solution: SinglePatternSolution) -> None:
        """Highlight the simulated band used for the band-profile plot.

        Parameters:
            solution: Current single-pattern solution.

        Returns:
            None.
        """

        band = solution.selected_band or {}
        coords = band.get("central_line")
        if coords is None:
            return
        x1, y1, x2, y2 = [float(value) for value in coords]
        self._pattern_axes.plot(
            [x1, x2],
            [y1, y2],
            color="#ffeb3b",
            linewidth=4.0,
            alpha=0.95,
            solid_capstyle="round",
            zorder=10,
            clip_on=True,
        )
        label_x = x1 + 0.62 * (x2 - x1)
        label_y = y1 + 0.62 * (y2 - y1)
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        hkl = str(band.get("hkl_group") or band.get("hkl") or "band")
        self._pattern_axes.text(
            label_x,
            label_y,
            f"profile {hkl}",
            color="black",
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "#ffeb3b", "alpha": 0.9, "edgecolor": "black", "pad": 1.6},
            zorder=11,
            clip_on=True,
        )

    def _draw_hough_panel(self, solution: SinglePatternSolution) -> None:
        """Draw the Hough/Radon image and selected backend peaks.

        Parameters:
            solution: Current single-pattern solution.

        Returns:
            None.
        """

        diagnostic = solution.hough_diagnostic
        self._hough_axes.set_title("Hough transform + top selected peaks")
        self._hough_axes.set_xlabel("Theta (deg)")
        self._hough_axes.set_ylabel("Rho (px)")
        if diagnostic is None:
            message = "Enable Hough indexing"
            summary = solution.hough_summary
            if summary is not None and not bool(summary.get("success", False)):
                message = str(summary.get("error", "Hough indexing failed"))
            self._hough_axes.text(
                0.5,
                0.5,
                message,
                transform=self._hough_axes.transAxes,
                ha="center",
                va="center",
                wrap=True,
            )
            return
        image = diagnostic.hough_image
        theta_axis = diagnostic.theta_axis
        rho_axis = diagnostic.rho_axis
        extent = [
            float(theta_axis.min()),
            float(theta_axis.max()),
            float(rho_axis.min()),
            float(rho_axis.max()),
        ]
        self._hough_axes.imshow(
            image,
            cmap="gray",
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        for peak in diagnostic.peaks[:DISPLAY_HOUGH_PEAK_LIMIT]:
            theta_deg = float(peak.display_theta_deg)
            self._hough_axes.scatter(
                [theta_deg],
                [peak.display_rho],
                c="#00e5ff" if peak.valid else "#bdbdbd",
                s=28,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            self._hough_axes.annotate(
                str(peak.rank),
                (theta_deg, peak.display_rho),
                xytext=(4, 3),
                textcoords="offset points",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

    def _on_button_press(self, event: object) -> None:
        """Start dragging the PC marker when clicked.

        Parameters:
            event: Matplotlib mouse event.

        Returns:
            None.
        """

        if event.inaxes is not self._pattern_axes or self._pc_artist is None:
            return
        contains, _ = self._pc_artist.contains(event)
        self._dragging_pc = bool(contains)

    def _on_motion(self, event: object) -> None:
        """Move the PC marker during dragging.

        Parameters:
            event: Matplotlib mouse event.

        Returns:
            None.
        """

        if not self._dragging_pc or event.inaxes is not self._pattern_axes:
            return
        if event.xdata is None or event.ydata is None or self._pc_artist is None:
            return
        xlim = self._pattern_axes.get_xlim()
        ylim = self._pattern_axes.get_ylim()
        x = min(max(float(event.xdata), min(xlim)), max(xlim))
        y = min(max(float(event.ydata), min(ylim)), max(ylim))
        self._pc_artist.set_offsets(np.asarray([[x, y]], dtype=np.float64))
        self.draw_idle()

    def _on_button_release(self, event: object) -> None:
        """Finish PC drag and emit normalized PC values.

        Parameters:
            event: Matplotlib mouse event.

        Returns:
            None.
        """

        if not self._dragging_pc:
            return
        self._dragging_pc = False
        if event.xdata is None or event.ydata is None or self._pc_changed_callback is None:
            return
        width = max(1.0, abs(np.diff(self._pattern_axes.get_xlim()))[0])
        height = max(1.0, abs(np.diff(self._pattern_axes.get_ylim()))[0])
        pcx = min(max(float(event.xdata) / width, 0.0), 1.0)
        pcy = min(max(float(event.ydata) / height, 0.0), 1.0)
        self._pc_changed_callback(pcx, pcy)


class SinglePatternSolverWindow(QtWidgets.QMainWindow):
    """Window for live single-pattern PC and detector-geometry debugging."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the window.

        Parameters:
            config_path: Optional YAML config path to load at startup.
        """

        super().__init__()
        self.setWindowTitle("Single Pattern Kikuchi Solver")
        self.resize(1450, 850)
        self._logger = logging.getLogger(__name__)
        self._config_path: Optional[Path] = None
        self._config: Optional[SinglePatternConfig] = None
        self._solution: Optional[SinglePatternSolution] = None
        self._log_handler: Optional[TextEditLogHandler] = None
        self._updating_controls = False
        self._source_extent: tuple[int, int] = (1, 1)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self.solve_current)
        self._init_ui()
        self._attach_logging()
        if config_path is not None:
            self.load_config(config_path)

    def _init_ui(self) -> None:
        """Create all widgets."""

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout()
        main_layout.addLayout(root, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left.setMaximumWidth(430)
        root.addWidget(left)

        source_group = QtWidgets.QGroupBox("Input Source")
        source_form = QtWidgets.QFormLayout(source_group)
        self._source_type_combo = QtWidgets.QComboBox()
        self._source_type_combo.addItem("HKL/Oxford CTF + pattern folder", "ctf")
        self._source_type_combo.addItem("TSL/EDAX OH5/H5", "oh5")
        self._source_type_combo.addItem("Single pattern image", "image")
        source_form.addRow("Mode", self._source_type_combo)
        self._source_path_edit = QtWidgets.QLineEdit()
        self._source_browse = QtWidgets.QPushButton("Browse")
        self._source_browse.clicked.connect(self._browse_source)
        source_form.addRow("Source", self._path_row(self._source_path_edit, self._source_browse))
        self._pattern_dir_edit = QtWidgets.QLineEdit()
        self._pattern_dir_browse = QtWidgets.QPushButton("Browse")
        self._pattern_dir_browse.clicked.connect(self._browse_pattern_dir)
        source_form.addRow("Pattern dir", self._path_row(self._pattern_dir_edit, self._pattern_dir_browse))
        self._pattern_template_edit = QtWidgets.QLineEdit("{x}_{y}.tiff")
        source_form.addRow("Pattern map", self._pattern_template_edit)
        self._pattern_field_edit = QtWidgets.QLineEdit("Pattern")
        source_form.addRow("OH5 field", self._pattern_field_edit)
        self._x_spin = self._int_spin(0, 0, 100000)
        self._y_spin = self._int_spin(0, 0, 100000)
        source_form.addRow("Pixel X", self._x_spin)
        source_form.addRow("Pixel Y", self._y_spin)
        self._scan_summary = QtWidgets.QLabel("No source loaded.")
        self._scan_summary.setWordWrap(True)
        source_form.addRow("Scan", self._scan_summary)
        load_source = QtWidgets.QPushButton("Load Source / Middle Pixel")
        load_source.clicked.connect(self._load_source_extent)
        source_form.addRow(load_source)
        left_layout.addWidget(source_group)

        form_group = QtWidgets.QGroupBox("Indexing Geometry")
        form = QtWidgets.QFormLayout(form_group)
        self._config_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse_config)
        form.addRow("YAML", self._path_row(self._config_edit, browse))
        self._convention_combo = QtWidgets.QComboBox()
        self._convention_combo.addItems(["oxford", "edax", "tsl", "emsoft"])
        form.addRow("PC convention", self._convention_combo)
        self._pcx_spin = self._double_spin(0.5, -5.0, 5.0, 6)
        self._pcy_spin = self._double_spin(0.5, -5.0, 5.0, 6)
        self._pcz_spin = self._double_spin(0.5, -5.0, 5.0, 6)
        form.addRow("PC x*", self._pcx_spin)
        form.addRow("PC y*", self._pcy_spin)
        form.addRow("PC z*", self._pcz_spin)
        self._sample_tilt_spin = self._double_spin(70.0, -180.0, 180.0, 3)
        self._tilt_spin = self._double_spin(0.0, -180.0, 180.0, 3)
        self._azimuth_spin = self._double_spin(0.0, -180.0, 180.0, 3)
        form.addRow("Sample tilt", self._sample_tilt_spin)
        form.addRow("Detector tilt", self._tilt_spin)
        form.addRow("Azimuthal", self._azimuth_spin)
        self._hough_enabled = QtWidgets.QCheckBox("Run kikuchipy Hough indexing")
        self._use_indexed_orientation = QtWidgets.QCheckBox("Overlay indexed orientation")
        form.addRow(self._hough_enabled)
        form.addRow(self._use_indexed_orientation)
        self._euler1_spin = self._double_spin(0.0, -360.0, 360.0, 3)
        self._euler2_spin = self._double_spin(0.0, -360.0, 360.0, 3)
        self._euler3_spin = self._double_spin(0.0, -360.0, 360.0, 3)
        form.addRow("Image Euler 1", self._euler1_spin)
        form.addRow("Image Euler 2", self._euler2_spin)
        form.addRow("Image Euler 3", self._euler3_spin)
        left_layout.addWidget(form_group)

        phase_group = QtWidgets.QGroupBox("Phase / Simulation")
        phase_form = QtWidgets.QFormLayout(phase_group)
        self._phase_name_edit = QtWidgets.QLineEdit("Ni")
        self._space_group_spin = self._int_spin(225, 1, 230)
        self._lattice_edit = QtWidgets.QLineEdit("3.5236, 3.5236, 3.5236, 90, 90, 90")
        self._hkl_edit = QtWidgets.QLineEdit("1,1,1; 2,0,0; 2,2,0; 3,1,1")
        self._desired_hkl_edit = QtWidgets.QLineEdit("1,1,1")
        phase_form.addRow("Phase", self._phase_name_edit)
        phase_form.addRow("Space group", self._space_group_spin)
        phase_form.addRow("Lattice", self._lattice_edit)
        phase_form.addRow("HKL families", self._hkl_edit)
        phase_form.addRow("Profile HKL", self._desired_hkl_edit)
        left_layout.addWidget(phase_group)

        actions = QtWidgets.QHBoxLayout()
        self._solve_button = QtWidgets.QPushButton("Solve")
        self._solve_button.clicked.connect(self.solve_current)
        self._save_json_button = QtWidgets.QPushButton("Save JSON")
        self._save_json_button.clicked.connect(self._save_json)
        self._save_png_button = QtWidgets.QPushButton("Save PNG")
        self._save_png_button.clicked.connect(self._save_png)
        actions.addWidget(self._solve_button)
        actions.addWidget(self._save_json_button)
        actions.addWidget(self._save_png_button)
        left_layout.addLayout(actions)

        self._summary = QtWidgets.QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumBlockCount(1000)
        left_layout.addWidget(self._summary, stretch=1)

        self._canvas = SinglePatternCanvas(self._set_pc_from_canvas)
        root.addWidget(self._canvas, stretch=1)

        log_group = QtWidgets.QGroupBox("Log Console")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self._log = QtWidgets.QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(130)
        self._log.setMaximumHeight(190)
        log_layout.addWidget(self._log)
        main_layout.addWidget(log_group)

        for widget in (
            self._x_spin,
            self._y_spin,
            self._pcx_spin,
            self._pcy_spin,
            self._pcz_spin,
            self._sample_tilt_spin,
            self._tilt_spin,
            self._azimuth_spin,
            self._euler1_spin,
            self._euler2_spin,
            self._euler3_spin,
            self._space_group_spin,
        ):
            widget.valueChanged.connect(self._schedule_solve)
        self._convention_combo.currentTextChanged.connect(self._schedule_solve)
        self._hough_enabled.stateChanged.connect(self._schedule_solve)
        self._use_indexed_orientation.stateChanged.connect(self._schedule_solve)
        self._source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        for widget in (
            self._source_path_edit,
            self._pattern_dir_edit,
            self._pattern_template_edit,
            self._pattern_field_edit,
            self._phase_name_edit,
            self._lattice_edit,
            self._hkl_edit,
            self._desired_hkl_edit,
        ):
            widget.editingFinished.connect(self._schedule_solve)
        self._on_source_type_changed()

    def _path_row(
        self,
        edit: QtWidgets.QLineEdit,
        button: QtWidgets.QPushButton,
    ) -> QtWidgets.QWidget:
        """Create a path edit row with a browse button.

        Parameters:
            edit: Line edit used for the path.
            button: Browse button.

        Returns:
            Composite row widget.
        """

        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        layout.addWidget(button)
        return row

    def _attach_logging(self) -> None:
        """Attach a QTextEdit logging handler."""

        handler = TextEditLogHandler(self._log)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        self._log_handler = handler

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Detach logging when the window closes.

        Parameters:
            event: Close event.

        Returns:
            None.
        """

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.detach()
            self._log_handler = None
        super().closeEvent(event)

    def _int_spin(self, value: int, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
        """Create an integer spin box.

        Parameters:
            value: Initial value.
            minimum: Minimum.
            maximum: Maximum.

        Returns:
            Spin box.
        """

        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _double_spin(
        self,
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
    ) -> QtWidgets.QDoubleSpinBox:
        """Create a double spin box.

        Parameters:
            value: Initial value.
            minimum: Minimum.
            maximum: Maximum.
            decimals: Decimal places.

        Returns:
            Spin box.
        """

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.001 if decimals >= 4 else 0.1)
        spin.setValue(value)
        return spin

    def _browse_config(self) -> None:
        """Browse for a YAML config file."""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select single-pattern YAML", filter="YAML (*.yml *.yaml)")
        if path:
            self.load_config(Path(path))

    def _browse_source(self) -> None:
        """Browse for a CTF, OH5/H5, or single-image source file."""

        mode = self._source_type()
        if mode == "ctf":
            title = "Select HKL/Oxford CTF"
            file_filter = "CTF files (*.ctf);;All files (*)"
        elif mode == "image":
            title = "Select single EBSP image"
            file_filter = "Images (*.png *.bmp *.tif *.tiff *.jpg *.jpeg);;All files (*)"
        else:
            title = "Select OH5/H5 file"
            file_filter = "HDF5 files (*.oh5 *.h5 *.hdf5);;All files (*)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, filter=file_filter)
        if path:
            self._source_path_edit.setText(path)
            self._load_source_extent()

    def _browse_pattern_dir(self) -> None:
        """Browse for a CTF external pattern folder."""

        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select pattern folder")
        if path:
            self._pattern_dir_edit.setText(path)
            self._load_source_extent()

    def _source_type(self) -> str:
        """Return the selected source type token.

        Returns:
            Source type token.
        """

        return str(self._source_type_combo.currentData())

    def _on_source_type_changed(self) -> None:
        """Refresh source-control enabled states after mode changes."""

        mode = self._source_type()
        is_ctf = mode == "ctf"
        is_oh5 = mode == "oh5"
        is_image = mode == "image"
        self._pattern_dir_edit.setEnabled(is_ctf)
        self._pattern_dir_browse.setEnabled(is_ctf)
        self._pattern_template_edit.setEnabled(is_ctf)
        self._pattern_field_edit.setEnabled(is_oh5)
        self._x_spin.setEnabled(not is_image)
        self._y_spin.setEnabled(not is_image)
        self._euler1_spin.setEnabled(is_image)
        self._euler2_spin.setEnabled(is_image)
        self._euler3_spin.setEnabled(is_image)
        if not self._updating_controls:
            self._convention_combo.setCurrentText("oxford" if is_ctf else "edax")

    def _load_source_extent(self) -> None:
        """Load scan dimensions and select the middle pixel by default."""

        path_text = self._source_path_edit.text().strip()
        if not path_text:
            return
        path = Path(path_text)
        mode = self._source_type()
        try:
            if mode == "ctf":
                reader = CtfPatternScanFileReader(
                    ctf_path=path,
                    pattern_dir=Path(self._pattern_dir_edit.text().strip())
                    if self._pattern_dir_edit.text().strip()
                    else None,
                    pattern_template=self._pattern_template_edit.text().strip() or None,
                )
                nx, ny = reader.nx, reader.ny
                reader.close()
            elif mode == "oh5":
                dataset = OH5ScanFileReader.from_path(path)
                nx, ny = dataset.nx, dataset.ny
                dataset.close()
            else:
                nx, ny = 1, 1
            self._source_extent = (nx, ny)
            self._updating_controls = True
            self._x_spin.setRange(0, max(0, nx - 1))
            self._y_spin.setRange(0, max(0, ny - 1))
            self._x_spin.setValue(max(0, nx // 2))
            self._y_spin.setValue(max(0, ny // 2))
            self._updating_controls = False
            self._scan_summary.setText(f"{mode.upper()} source loaded: nx={nx}, ny={ny}; defaulted to middle pixel.")
            self.solve_current()
        except Exception as exc:
            self._updating_controls = False
            self._logger.exception("Failed to load source extent: %s", exc)
            self._scan_summary.setText(f"Source load failed: {exc}")

    def load_config(self, path: Path) -> None:
        """Load a YAML config and solve it.

        Parameters:
            path: YAML config path.

        Returns:
            None.
        """

        self._config_path = Path(path)
        self._config_edit.setText(str(path))
        self._config = load_single_pattern_config(path)
        self._load_controls_from_config()
        self.solve_current()

    def _load_controls_from_config(self) -> None:
        """Populate controls from the current config."""

        if self._config is None:
            return
        self._updating_controls = True
        raw = self._config.raw
        input_cfg = raw.get("input", {})
        detector_cfg = raw.get("detector", {})
        phase_cfg = raw.get("phase", {})
        simulation_cfg = raw.get("simulation", {})
        profile_cfg = raw.get("band_profile", {})
        hough_cfg = raw.get("hough", {})
        input_type = str(input_cfg.get("type", "oh5")).lower()
        if input_type in {"h5", "hdf5"}:
            input_type = "oh5"
        if input_type in {"pattern", "single"}:
            input_type = "image"
        combo_index = self._source_type_combo.findData(input_type)
        if combo_index >= 0:
            self._source_type_combo.setCurrentIndex(combo_index)
        if input_type == "ctf":
            self._source_path_edit.setText(str(input_cfg.get("ctf_path", "")))
            self._pattern_dir_edit.setText(str(input_cfg.get("pattern_dir", "")))
            self._pattern_template_edit.setText(str(input_cfg.get("pattern_template", "{x}_{y}.tiff")))
        else:
            self._source_path_edit.setText(str(input_cfg.get("path", "")))
        self._pattern_field_edit.setText(str(input_cfg.get("pattern_field", "Pattern")))
        eulers_deg = input_cfg.get("eulers_deg", [0.0, 0.0, 0.0])
        self._euler1_spin.setValue(float(eulers_deg[0]))
        self._euler2_spin.setValue(float(eulers_deg[1]))
        self._euler3_spin.setValue(float(eulers_deg[2]))
        pc = detector_cfg.get("pc", [0.5, 0.5, 0.5])
        self._x_spin.setValue(int(input_cfg.get("x", 0)))
        self._y_spin.setValue(int(input_cfg.get("y", 0)))
        self._convention_combo.setCurrentText(str(detector_cfg.get("convention", "edax")))
        self._pcx_spin.setValue(float(pc[0]))
        self._pcy_spin.setValue(float(pc[1]))
        self._pcz_spin.setValue(float(pc[2]))
        self._sample_tilt_spin.setValue(float(detector_cfg.get("sample_tilt", 70.0)))
        self._tilt_spin.setValue(float(detector_cfg.get("tilt", 0.0)))
        self._azimuth_spin.setValue(float(detector_cfg.get("azimuthal", 0.0)))
        self._hough_enabled.setChecked(bool(hough_cfg.get("enabled", True)))
        self._use_indexed_orientation.setChecked(bool(hough_cfg.get("use_indexed_orientation", True)))
        self._phase_name_edit.setText(str(phase_cfg.get("name", "Ni")))
        self._space_group_spin.setValue(int(phase_cfg.get("space_group", 225)))
        self._lattice_edit.setText(", ".join(str(value) for value in phase_cfg.get("lattice", [])))
        hkl_values = simulation_cfg.get("hkl_list", [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]])
        self._hkl_edit.setText("; ".join(",".join(str(item) for item in row) for row in hkl_values))
        self._desired_hkl_edit.setText(str(profile_cfg.get("desired_hkl", "1,1,1")))
        self._on_source_type_changed()
        self._updating_controls = False

    def _config_from_controls(self) -> Optional[SinglePatternConfig]:
        """Return a config copy with current control values.

        Returns:
            Updated config or None.
        """

        if self._config is None:
            self._config = self._default_config_from_controls()
        raw = copy.deepcopy(self._config.raw)
        raw.setdefault("input", {})
        raw.setdefault("detector", {})
        raw.setdefault("phase", {})
        raw.setdefault("simulation", {})
        raw.setdefault("band_profile", {})
        raw.setdefault("hough", {})
        mode = self._source_type()
        raw["input"]["type"] = mode
        if mode == "ctf":
            raw["input"]["ctf_path"] = self._source_path_edit.text().strip()
            raw["input"]["pattern_dir"] = self._pattern_dir_edit.text().strip()
            raw["input"]["pattern_template"] = self._pattern_template_edit.text().strip() or None
        else:
            raw["input"]["path"] = self._source_path_edit.text().strip()
        raw["input"]["pattern_field"] = self._pattern_field_edit.text().strip() or "Pattern"
        raw["input"]["x"] = int(self._x_spin.value())
        raw["input"]["y"] = int(self._y_spin.value())
        raw["input"]["eulers_deg"] = [
            float(self._euler1_spin.value()),
            float(self._euler2_spin.value()),
            float(self._euler3_spin.value()),
        ]
        raw["detector"]["convention"] = self._convention_combo.currentText()
        raw["detector"]["pc"] = [
            float(self._pcx_spin.value()),
            float(self._pcy_spin.value()),
            float(self._pcz_spin.value()),
        ]
        raw["detector"]["sample_tilt"] = float(self._sample_tilt_spin.value())
        raw["detector"]["tilt"] = float(self._tilt_spin.value())
        raw["detector"]["azimuthal"] = float(self._azimuth_spin.value())
        raw["phase"]["name"] = self._phase_name_edit.text().strip() or "Ni"
        raw["phase"]["space_group"] = int(self._space_group_spin.value())
        raw["phase"]["lattice"] = self._parse_float_list(self._lattice_edit.text(), expected=6)
        raw["phase"]["atoms"] = [
            {"element": raw["phase"]["name"], "position": [0, 0, 0]},
        ]
        raw["simulation"]["hkl_list"] = self._parse_hkl_list(self._hkl_edit.text())
        raw["band_profile"]["desired_hkl"] = self._desired_hkl_edit.text().strip() or "1,1,1"
        raw["hough"]["enabled"] = bool(self._hough_enabled.isChecked())
        raw["hough"]["use_indexed_orientation"] = bool(self._use_indexed_orientation.isChecked())
        raw["hough"]["n_bands"] = DISPLAY_HOUGH_PEAK_LIMIT
        return SinglePatternConfig(path=self._config.path, raw=raw)

    def _set_pc_from_canvas(self, pcx: float, pcy: float) -> None:
        """Update PC controls from a dragged pattern marker.

        Parameters:
            pcx: Normalized detector x coordinate.
            pcy: Normalized detector y coordinate.

        Returns:
            None.
        """

        self._updating_controls = True
        self._pcx_spin.setValue(float(pcx))
        self._pcy_spin.setValue(float(pcy))
        self._updating_controls = False
        self._logger.info("Updated PC from draggable marker to x*=%.6f, y*=%.6f.", pcx, pcy)
        self.solve_current()

    def _default_config_from_controls(self) -> SinglePatternConfig:
        """Create a baseline config when no YAML file has been loaded.

        Returns:
            Baseline single-pattern config.
        """

        return SinglePatternConfig(
            path=Path("interactive_indexing_debug.yml"),
            raw={
                "input": {"type": self._source_type()},
                "phase": {
                    "name": "Ni",
                    "space_group": 225,
                    "lattice": [3.5236, 3.5236, 3.5236, 90, 90, 90],
                    "atoms": [{"element": "Ni", "position": [0, 0, 0]}],
                },
                "detector": {
                    "convention": self._convention_combo.currentText(),
                    "pc": [0.5, 0.5, 0.5],
                    "sample_tilt": 70.0,
                    "tilt": 0.0,
                    "azimuthal": 0.0,
                    "px_size": 1.0,
                    "binning": 1,
                },
                "orientation": {"direction": "lab2crystal"},
                "simulation": {"hkl_list": [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]},
                "band_profile": {"desired_hkl": "1,1,1", "rectWidth": 20},
                "hough": {
                    "enabled": True,
                    "use_indexed_orientation": True,
                    "n_bands": DISPLAY_HOUGH_PEAK_LIMIT,
                },
            },
        )

    def _parse_float_list(self, text: str, *, expected: int) -> list[float]:
        """Parse a comma-separated float list.

        Parameters:
            text: User-entered text.
            expected: Required number of values.

        Returns:
            Parsed float values.
        """

        values = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
        if len(values) != expected:
            raise ValueError(f"Expected {expected} numeric values, got {len(values)}.")
        return values

    def _parse_hkl_list(self, text: str) -> list[list[int]]:
        """Parse semicolon-separated HKL triplets.

        Parameters:
            text: User-entered HKL list.

        Returns:
            List of integer HKL triplets.
        """

        rows: list[list[int]] = []
        for chunk in text.split(";"):
            values = [int(part.strip()) for part in chunk.split(",") if part.strip()]
            if values:
                if len(values) != 3:
                    raise ValueError(f"HKL entry must have three values: {chunk}")
                rows.append(values)
        if not rows:
            raise ValueError("At least one HKL family is required.")
        return rows

    def _schedule_solve(self) -> None:
        """Schedule a debounced solve after control changes."""

        if not self._updating_controls:
            self._debounce.start()

    def solve_current(self) -> None:
        """Solve the current single-pattern config and refresh views."""

        config = self._config_from_controls()
        if config is None:
            return
        try:
            self._solution = solve_single_pattern(config, logger=self._logger)
            self._canvas.update_solution(self._solution)
            self._update_summary()
            self._logger.info(
                "Rendered pixel x=%d y=%d with %d simulated lines and %d displayed Hough bands.",
                self._solution.x,
                self._solution.y,
                len(self._solution.lines),
                min(
                    DISPLAY_HOUGH_PEAK_LIMIT,
                    len(self._solution.hough_diagnostic.peaks)
                    if self._solution.hough_diagnostic is not None
                    else 0,
                ),
            )
        except Exception as exc:
            self._logger.exception("Single-pattern solve failed: %s", exc)
            self._summary.setPlainText(str(exc))

    def _update_summary(self) -> None:
        """Update the text summary panel."""

        if self._solution is None:
            return
        band = self._solution.selected_band or {}
        lines_by_family: dict[str, int] = {}
        for line in self._solution.lines:
            family = str(line.get("hkl", ""))
            lines_by_family[family] = lines_by_family.get(family, 0) + 1
        self._summary.setPlainText(
            "\n".join(
                [
                    f"Pixel: x={self._solution.x}, y={self._solution.y}",
                    f"Pattern shape: {self._solution.pattern.shape}",
                    f"Euler deg: {np.rad2deg(self._solution.eulers_rad)}",
                    f"Detector: {self._solution.detector_summary}",
                    f"Simulated lines by family: {lines_by_family}",
                    f"Selected band: hkl={band.get('hkl')}, valid={band.get('band_valid')}, psnr={band.get('psnr')}",
                    f"Hough: {self._solution.hough_summary}",
                ]
            )
        )

    def _default_output_path(self, suffix: str) -> Path:
        """Return a default output path next to the config.

        Parameters:
            suffix: Filename suffix.

        Returns:
            Path.
        """

        base = self._config_path or Path("single_pattern")
        return base.with_name(f"{base.stem}_solution{suffix}")

    def _save_json(self) -> None:
        """Save the current solution JSON."""

        if self._solution is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save solution JSON",
            str(self._default_output_path(".json")),
            "JSON (*.json)",
        )
        if path:
            write_solution_json(self._solution, path)

    def _save_png(self) -> None:
        """Save the current solution PNG."""

        if self._solution is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save solution PNG",
            str(self._default_output_path(".png")),
            "PNG (*.png)",
        )
        if path:
            render_solution(self._solution, path)

    def snapshot(self, path: Path) -> None:
        """Save a screenshot of the window.

        Parameters:
            path: Destination image path.

        Returns:
            None.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self.grab()
        pixmap.save(str(path))


def build_parser() -> argparse.ArgumentParser:
    """Build the GUI parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description="Single-pattern solver GUI.")
    parser.add_argument("--config", type=Path, help="YAML config to open.")
    return parser


def main() -> None:
    """Run the single-pattern solver GUI."""

    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Arial", 9))
    window = SinglePatternSolverWindow(args.config)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
