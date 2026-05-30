"""PySide GUI for live single-pattern EBSP geometry debugging."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

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

    def __init__(self) -> None:
        """Initialize the canvas."""

        self._figure = Figure(figsize=(9.0, 4.6))
        self._pattern_axes = self._figure.add_subplot(1, 2, 1)
        self._profile_axes = self._figure.add_subplot(1, 2, 2)
        super().__init__(self._figure)

    def update_solution(self, solution: Optional[SinglePatternSolution]) -> None:
        """Render a solution or empty state.

        Parameters:
            solution: Optional single-pattern solution.

        Returns:
            None.
        """

        self._pattern_axes.clear()
        self._profile_axes.clear()
        if solution is None:
            self._pattern_axes.text(0.5, 0.5, "Load config", ha="center", va="center")
            self._profile_axes.text(0.5, 0.5, "No profile", ha="center", va="center")
        else:
            self._pattern_axes.imshow(solution.pattern, cmap="gray")
            self._pattern_axes.set_title("Experimental EBSP + simulated Kikuchi lines")
            self._pattern_axes.set_xticks([])
            self._pattern_axes.set_yticks([])
            _draw_lines_on_axes(self._pattern_axes, solution.pattern.shape, solution.lines)
            _draw_profile_on_axes(self._profile_axes, solution.profile_payload)
        self._figure.tight_layout()
        self.draw_idle()


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
        root = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left.setMaximumWidth(430)
        root.addWidget(left)

        form_group = QtWidgets.QGroupBox("Config")
        form = QtWidgets.QFormLayout(form_group)
        self._config_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse_config)
        path_row = QtWidgets.QWidget()
        path_layout = QtWidgets.QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self._config_edit)
        path_layout.addWidget(browse)
        form.addRow("YAML", path_row)

        self._x_spin = self._int_spin(0, 0, 100000)
        self._y_spin = self._int_spin(0, 0, 100000)
        form.addRow("Pixel X", self._x_spin)
        form.addRow("Pixel Y", self._y_spin)
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
        left_layout.addWidget(form_group)

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

        self._log = QtWidgets.QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(170)
        left_layout.addWidget(self._log)

        self._canvas = SinglePatternCanvas()
        root.addWidget(self._canvas, stretch=1)

        for widget in (
            self._x_spin,
            self._y_spin,
            self._pcx_spin,
            self._pcy_spin,
            self._pcz_spin,
            self._sample_tilt_spin,
            self._tilt_spin,
            self._azimuth_spin,
        ):
            widget.valueChanged.connect(self._schedule_solve)
        self._convention_combo.currentTextChanged.connect(self._schedule_solve)

    def _attach_logging(self) -> None:
        """Attach a QTextEdit logging handler."""

        handler = TextEditLogHandler(self._log)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
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
        self._updating_controls = False

    def _config_from_controls(self) -> Optional[SinglePatternConfig]:
        """Return a config copy with current control values.

        Returns:
            Updated config or None.
        """

        if self._config is None:
            return None
        raw = copy.deepcopy(self._config.raw)
        raw.setdefault("input", {})
        raw.setdefault("detector", {})
        raw["input"]["x"] = int(self._x_spin.value())
        raw["input"]["y"] = int(self._y_spin.value())
        raw["detector"]["convention"] = self._convention_combo.currentText()
        raw["detector"]["pc"] = [
            float(self._pcx_spin.value()),
            float(self._pcy_spin.value()),
            float(self._pcz_spin.value()),
        ]
        raw["detector"]["sample_tilt"] = float(self._sample_tilt_spin.value())
        raw["detector"]["tilt"] = float(self._tilt_spin.value())
        raw["detector"]["azimuthal"] = float(self._azimuth_spin.value())
        return SinglePatternConfig(path=self._config.path, raw=raw)

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
