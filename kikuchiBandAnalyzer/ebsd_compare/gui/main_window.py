"""Main window for the EBSD scan comparator GUI."""

from __future__ import annotations

import argparse
from functools import partial
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.gui.auto_scan import AutoScanController
from kikuchiBandAnalyzer.ebsd_compare.gui.logging_widget import (
    GuiLogHandler,
    LogEmitter,
    LogViewer,
)
from kikuchiBandAnalyzer.ebsd_compare.gui.registration_window import RegistrationDialog
from kikuchiBandAnalyzer.ebsd_compare.gui.selection import SelectionController, SelectionState
from kikuchiBandAnalyzer.ebsd_compare.gui.validation import (
    validate_int_in_range,
    validate_speed_ms,
)
from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    AlignmentResult,
    alignment_from_config,
)
from kikuchiBandAnalyzer.ebsd_compare.simulated import SimulatedScanFactory
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging, load_yaml_config

matplotlib.use("QtAgg")


class CompactNavigationToolbar(NavigationToolbar2QT):
    """Navigation toolbar with only home, pan, and zoom actions."""

    toolitems = (
        ("Home", "Reset original view", "home", "home"),
        ("Pan", "Pan axes with left mouse, zoom with right", "move", "pan"),
        ("Zoom", "Zoom to rectangle", "zoom_to_rect", "zoom"),
    )

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the compact toolbar without coordinate readouts.

        Parameters:
            canvas: Matplotlib canvas to control.
            parent: Optional parent widget.

        Returns:
            None.
        """

        super().__init__(canvas, parent, coordinates=False)
        self.setMovable(False)
        self.setFloatable(False)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )


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
        self._marker_coords: Optional[Tuple[int, int]] = None
        self._marker_artist = None
        self._marker_color = "#ffdd00"
        self._axes.set_title(title)
        self._axes.set_xticks([])
        self._axes.set_yticks([])

    @property
    def axes(self) -> matplotlib.axes.Axes:
        """Return the underlying Matplotlib axes.

        Returns:
            Matplotlib axes instance.
        """

        return self._axes

    def update_data(
        self,
        data: np.ndarray,
        cmap: str = "gray",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        reset_view: bool = False,
    ) -> None:
        """Update the displayed data.

        Parameters:
            data: 2D array to display.
            cmap: Matplotlib colormap name.
            vmin: Optional lower bound for color scaling.
            vmax: Optional upper bound for color scaling.
            reset_view: Whether to reset the axes view.
        """

        if not reset_view and self._image is not None:
            current_shape = getattr(self._image.get_array(), "shape", None)
            if current_shape is not None and current_shape != data.shape:
                reset_view = True
        if reset_view or self._image is None:
            self._axes.clear()
            self._axes.set_title(self._title)
            self._axes.set_xticks([])
            self._axes.set_yticks([])
            self._image = self._axes.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            self._marker_artist = None
            if self._marker_coords is not None:
                self._draw_marker()
        else:
            self._image.set_data(data)
            self._image.set_cmap(cmap)
            if vmin is not None or vmax is not None:
                self._image.set_clim(vmin=vmin, vmax=vmax)
        self._figure.tight_layout()
        self.draw_idle()

    def connect_click(self, handler: QtCore.Slot) -> None:
        """Connect a click handler to the canvas.

        Parameters:
            handler: Matplotlib event handler.
        """

        self.mpl_connect("button_press_event", handler)

    def set_marker(self, x: int, y: int, color: str = "#ffdd00") -> None:
        """Set the marker location on the map.

        Parameters:
            x: Column index.
            y: Row index.
            color: Marker color.
        """

        self._marker_coords = (x, y)
        self._marker_color = color
        if self._image is not None:
            self._draw_marker()
            self.draw_idle()

    def clear_marker(self) -> None:
        """Clear the marker from the map.

        Returns:
            None.
        """

        self._marker_coords = None
        if self._marker_artist is not None:
            self._marker_artist.remove()
            self._marker_artist = None
            self.draw_idle()

    def _draw_marker(self) -> None:
        """Draw the marker on the map.

        Returns:
            None.
        """

        if self._marker_coords is None:
            return
        if self._marker_artist is not None:
            self._marker_artist.remove()
        x, y = self._marker_coords
        self._marker_artist = self._axes.scatter(
            [x],
            [y],
            s=90,
            c=self._marker_color,
            marker="x",
            linewidths=2.0,
            zorder=3,
        )


class MapPanel(QtWidgets.QWidget):
    """Panel with a map canvas, toolbar, and contrast controls.

    Parameters:
        title: Title for the map canvas.
        default_low: Default low percentile for contrast.
        default_high: Default high percentile for contrast.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        title: str,
        default_low: float,
        default_high: float,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the map panel.

        Parameters:
            title: Title for the map canvas.
            default_low: Default low percentile.
            default_high: Default high percentile.
            parent: Optional parent widget.
        """

        super().__init__(parent=parent)
        self._canvas = MapCanvas(title)
        self._toolbar = CompactNavigationToolbar(self._canvas, self)
        self._toolbar.setOrientation(QtCore.Qt.Vertical)
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._toolbar.setIconSize(QtCore.QSize(14, 14))
        self._low_spin = QtWidgets.QDoubleSpinBox()
        self._low_spin.setRange(0.0, 100.0)
        self._low_spin.setDecimals(1)
        self._low_spin.setSingleStep(0.5)
        self._low_spin.setValue(default_low)
        self._low_spin.setFixedWidth(58)
        self._low_spin.setAlignment(QtCore.Qt.AlignRight)
        self._high_spin = QtWidgets.QDoubleSpinBox()
        self._high_spin.setRange(0.0, 100.0)
        self._high_spin.setDecimals(1)
        self._high_spin.setSingleStep(0.5)
        self._high_spin.setValue(default_high)
        self._high_spin.setFixedWidth(58)
        self._high_spin.setAlignment(QtCore.Qt.AlignRight)
        self._low_spin.setToolTip(
            f"Low percentile for contrast (0-100). Default: {default_low}. Example: 2.0."
        )
        self._high_spin.setToolTip(
            f"High percentile for contrast (0-100). Default: {default_high}. Example: 98.0."
        )
        self._error_label = QtWidgets.QLabel()
        self._error_label.setStyleSheet("color: #c62828;")
        self._error_label.setVisible(False)

        overlay = QtWidgets.QFrame()
        overlay.setObjectName("mapOverlay")
        overlay.setStyleSheet(
            "QFrame#mapOverlay {"
            "background-color: rgba(255, 255, 255, 200);"
            "border: 1px solid #c0c0c0;"
            "border-radius: 4px;"
            "}"
        )
        overlay.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        overlay_layout = QtWidgets.QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(4, 4, 4, 4)
        overlay_layout.setSpacing(2)
        overlay_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        overlay_layout.addWidget(self._toolbar)
        label_l = QtWidgets.QLabel("L")
        label_l.setAlignment(QtCore.Qt.AlignCenter)
        label_h = QtWidgets.QLabel("H")
        label_h.setAlignment(QtCore.Qt.AlignCenter)
        overlay_layout.addWidget(label_l)
        overlay_layout.addWidget(self._low_spin)
        overlay_layout.addWidget(label_h)
        overlay_layout.addWidget(self._high_spin)
        overlay_layout.addWidget(self._error_label)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QGridLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self._canvas, 0, 0)
        container_layout.addWidget(
            overlay, 0, 0, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(container)

        self._canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def canvas(self) -> MapCanvas:
        """Return the map canvas.

        Returns:
            MapCanvas instance.
        """

        return self._canvas

    def contrast_values(self) -> Tuple[float, float]:
        """Return the low/high contrast percentiles.

        Returns:
            Tuple of (low, high) percentiles.
        """

        return self._low_spin.value(), self._high_spin.value()

    def set_contrast_values(self, low: float, high: float, block_signals: bool = False) -> None:
        """Set the contrast percentiles.

        Parameters:
            low: Low percentile.
            high: High percentile.
            block_signals: Whether to block spinbox signals.

        Returns:
            None.
        """

        if block_signals:
            self._low_spin.blockSignals(True)
            self._high_spin.blockSignals(True)
        self._low_spin.setValue(low)
        self._high_spin.setValue(high)
        if block_signals:
            self._low_spin.blockSignals(False)
            self._high_spin.blockSignals(False)

    def connect_contrast_changed(self, handler: QtCore.Slot) -> None:
        """Connect contrast changes to a handler.

        Parameters:
            handler: Slot to invoke on change.

        Returns:
            None.
        """

        self._low_spin.valueChanged.connect(handler)
        self._high_spin.valueChanged.connect(handler)

    def set_error(self, message: str) -> None:
        """Set an inline error message.

        Parameters:
            message: Error message text.

        Returns:
            None.
        """

        self._error_label.setText(message)
        self._error_label.setVisible(bool(message))

    def clear_error(self) -> None:
        """Clear the inline error message.

        Returns:
            None.
        """

        self._error_label.setText("")
        self._error_label.setVisible(False)


class PatternPanel(QtWidgets.QWidget):
    """Panel with a pattern canvas and toolbar.

    Parameters:
        title: Title for the pattern canvas.
        parent: Optional parent widget.
    """

    def __init__(
        self, title: str, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        """Initialize the pattern panel.

        Parameters:
            title: Title for the pattern canvas.
            parent: Optional parent widget.
        """

        super().__init__(parent=parent)
        self._canvas = MapCanvas(title)
        self._toolbar = CompactNavigationToolbar(self._canvas, self)
        self._toolbar.setOrientation(QtCore.Qt.Vertical)
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._toolbar.setIconSize(QtCore.QSize(14, 14))
        overlay = QtWidgets.QFrame()
        overlay.setObjectName("patternOverlay")
        overlay.setStyleSheet(
            "QFrame#patternOverlay {"
            "background-color: rgba(255, 255, 255, 200);"
            "border: 1px solid #c0c0c0;"
            "border-radius: 4px;"
            "}"
        )
        overlay.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        overlay_layout = QtWidgets.QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(4, 4, 4, 4)
        overlay_layout.setSpacing(2)
        overlay_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        overlay_layout.addWidget(self._toolbar)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QGridLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self._canvas, 0, 0)
        container_layout.addWidget(
            overlay, 0, 0, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(container)
        self._canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def canvas(self) -> MapCanvas:
        """Return the pattern canvas.

        Returns:
            MapCanvas instance.
        """

        return self._canvas

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
        self._log_dock: Optional[QtWidgets.QDockWidget] = None
        self._selection_controller: Optional[SelectionController] = None
        self._auto_scan_controller: Optional[AutoScanController] = None
        self._syncing_coords = False
        self._syncing_map_view = False
        self._syncing_pattern_view = False
        self._map_panels: Dict[str, MapPanel] = {}
        self._pattern_panels: Dict[str, PatternPanel] = {}
        self._map_triplet: Optional[Dict[str, np.ndarray]] = None
        self._inline_status_label: Optional[QtWidgets.QLabel] = None
        self._inline_error_label: Optional[QtWidgets.QLabel] = None
        self._x_validator: Optional[QtGui.QIntValidator] = None
        self._y_validator: Optional[QtGui.QIntValidator] = None
        self._map_contrast_low_default = 2.0
        self._map_contrast_high_default = 98.0
        self._pattern_reset_view = True
        auto_config = self._config.get("auto_scan", {})
        self._auto_min_ms = int(auto_config.get("min_delay_ms", 25))
        self._auto_max_ms = int(auto_config.get("max_delay_ms", 2000))
        self._auto_default_ms = int(auto_config.get("delay_ms", 200))
        self._init_ui()
        self._selection_controller = SelectionController(
            self._apply_selection_update,
            self._show_coord_error,
            logger=self._logger,
        )
        self._auto_scan_controller = AutoScanController(
            self._on_auto_scan_step,
            self._on_auto_scan_finished,
            logger=self._logger,
            parent=self,
        )
        self._attach_log_handler()

    def _init_ui(self) -> None:
        """Initialize the GUI layout and widgets."""

        self.setWindowTitle("EBSD Scan Comparator")
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)

        self._file_a_edit = QtWidgets.QLineEdit()
        self._file_b_edit = QtWidgets.QLineEdit()
        self._file_a_button = QtWidgets.QPushButton("Browse A")
        self._file_b_button = QtWidgets.QPushButton("Browse B")
        self._load_button = QtWidgets.QPushButton("Load Scans")
        self._file_a_button.clicked.connect(self._browse_file_a)
        self._file_b_button.clicked.connect(self._browse_file_b)
        self._load_button.clicked.connect(self._load_from_inputs)
        self._file_a_edit.textChanged.connect(self._clear_file_error)
        self._file_b_edit.textChanged.connect(self._clear_file_error)
        self._file_a_edit.setMinimumWidth(220)
        self._file_a_edit.setMaximumWidth(340)
        self._file_b_edit.setMinimumWidth(220)
        self._file_b_edit.setMaximumWidth(340)
        self._load_button.setFixedWidth(120)
        self._file_a_edit.setToolTip(
            "Path to scan A (.oh5). Required. Example: testData/Test_Ti.oh5"
        )
        self._file_b_edit.setToolTip(
            "Path to scan B (.oh5). Required. Example: testData/Test_Ti_noisy.oh5"
        )
        self._file_a_button.setToolTip("Browse for the scan A file.")
        self._file_b_button.setToolTip("Browse for the scan B file.")
        self._load_button.setToolTip(
            "Load both scans. Both paths must exist and have matching grids."
        )
        self._x_input = QtWidgets.QLineEdit()
        self._y_input = QtWidgets.QLineEdit()
        coord_max_chars = 6
        metrics = QtGui.QFontMetrics(self._x_input.font())
        coord_width = metrics.horizontalAdvance("0" * coord_max_chars) + 12
        self._x_input.setFixedWidth(coord_width)
        self._y_input.setFixedWidth(coord_width)
        self._x_input.setMaxLength(coord_max_chars)
        self._y_input.setMaxLength(coord_max_chars)
        self._x_input.setAlignment(QtCore.Qt.AlignRight)
        self._y_input.setAlignment(QtCore.Qt.AlignRight)
        self._x_input.setEnabled(False)
        self._y_input.setEnabled(False)
        self._x_input.editingFinished.connect(self._on_coord_edit_finished)
        self._y_input.editingFinished.connect(self._on_coord_edit_finished)
        self._x_input.setToolTip(
            "Column index (X). Enter an integer within the scan width."
        )
        self._y_input.setToolTip(
            "Row index (Y). Enter an integer within the scan height."
        )
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)
        top_row.addWidget(QtWidgets.QLabel("Scan A"))
        top_row.addWidget(self._file_a_edit)
        top_row.addWidget(self._file_a_button)
        top_row.addSpacing(8)
        top_row.addWidget(QtWidgets.QLabel("Scan B"))
        top_row.addWidget(self._file_b_edit)
        top_row.addWidget(self._file_b_button)
        top_row.addSpacing(8)
        top_row.addWidget(self._load_button)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        control_row = QtWidgets.QHBoxLayout()
        control_row.setSpacing(6)
        control_row.addWidget(QtWidgets.QLabel("Map Field"))
        self._map_field_combo = QtWidgets.QComboBox()
        self._map_field_combo.currentTextChanged.connect(self._update_maps)
        self._map_field_combo.setMaximumWidth(160)
        self._map_field_combo.setToolTip(
            "Scalar map to display for both scans and the delta/ratio view. "
            "Default comes from the configuration."
        )
        control_row.addWidget(self._map_field_combo)
        self._registration_button = QtWidgets.QPushButton("Registration")
        self._registration_button.clicked.connect(self._open_registration_from_button)
        self._registration_button.setFixedWidth(120)
        self._registration_button.setToolTip(
            "Open the registration tool to align mismatched scan grids."
        )
        control_row.addWidget(self._registration_button)
        self._screenshot_button = QtWidgets.QPushButton("Save Screenshot")
        self._screenshot_button.clicked.connect(self._on_save_screenshot)
        self._screenshot_button.setFixedWidth(140)
        self._screenshot_button.setToolTip(
            "Save a PNG screenshot of the current comparison view."
        )
        control_row.addWidget(self._screenshot_button)
        control_row.addSpacing(12)
        control_row.addWidget(QtWidgets.QLabel("X (col)"))
        control_row.addWidget(self._x_input)
        control_row.addWidget(QtWidgets.QLabel("Y (row)"))
        control_row.addWidget(self._y_input)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        autoscan_row = QtWidgets.QHBoxLayout()
        autoscan_row.setSpacing(6)
        self._auto_play_button = QtWidgets.QPushButton("Play")
        self._auto_pause_button = QtWidgets.QPushButton("Pause")
        self._auto_stop_button = QtWidgets.QPushButton("Stop")
        self._auto_play_button.clicked.connect(self._on_auto_scan_play)
        self._auto_pause_button.clicked.connect(self._on_auto_scan_pause_resume)
        self._auto_stop_button.clicked.connect(self._on_auto_scan_stop)
        self._auto_play_button.setEnabled(False)
        self._auto_pause_button.setEnabled(False)
        self._auto_stop_button.setEnabled(False)
        self._auto_play_button.setToolTip(
            "Start auto-scan from (0, 0) and step across the raster."
        )
        self._auto_pause_button.setToolTip("Pause or resume the auto-scan.")
        self._auto_stop_button.setToolTip("Stop the auto-scan.")
        autoscan_row.addWidget(QtWidgets.QLabel("Auto Scan"))
        autoscan_row.addWidget(self._auto_play_button)
        autoscan_row.addWidget(self._auto_pause_button)
        autoscan_row.addWidget(self._auto_stop_button)
        autoscan_row.addSpacing(12)
        autoscan_row.addWidget(QtWidgets.QLabel("Speed (ms)"))
        self._auto_speed_spin = QtWidgets.QSpinBox()
        self._auto_speed_spin.setRange(self._auto_min_ms, self._auto_max_ms)
        self._auto_speed_spin.setSingleStep(25)
        self._auto_speed_spin.setValue(self._auto_default_ms)
        self._auto_speed_spin.setSuffix(" ms")
        self._auto_speed_spin.setEnabled(False)
        self._auto_speed_spin.valueChanged.connect(self._on_auto_scan_speed_changed)
        self._auto_speed_spin.setToolTip(
            f"Delay between steps in milliseconds. Range: {self._auto_min_ms}-{self._auto_max_ms}. "
            f"Default: {self._auto_default_ms}. Example: {self._auto_default_ms}."
        )
        autoscan_row.addWidget(self._auto_speed_spin)
        autoscan_row.addSpacing(12)
        self._inline_status_label = QtWidgets.QLabel("X=--, Y=--")
        self._inline_status_label.setStyleSheet("color: #1f77b4;")
        autoscan_row.addWidget(self._inline_status_label)
        self._inline_error_label = QtWidgets.QLabel()
        self._inline_error_label.setStyleSheet("color: #c62828;")
        self._inline_error_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        autoscan_row.addWidget(self._inline_error_label)
        autoscan_row.addStretch(1)
        layout.addLayout(autoscan_row)

        map_layout = QtWidgets.QHBoxLayout()
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(6)
        self._map_panel_a = MapPanel(
            "Scan A",
            self._map_contrast_low_default,
            self._map_contrast_high_default,
        )
        self._map_panel_b = MapPanel(
            "Scan B",
            self._map_contrast_low_default,
            self._map_contrast_high_default,
        )
        self._map_panel_d = MapPanel(
            "Δ/Ratio",
            self._map_contrast_low_default,
            self._map_contrast_high_default,
        )
        self._map_panels = {
            "A": self._map_panel_a,
            "B": self._map_panel_b,
            "D": self._map_panel_d,
        }
        for key, panel in self._map_panels.items():
            panel.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            panel.canvas().connect_click(self._on_map_click)
            panel.canvas().setToolTip(
                "Click to select a pixel (X=column, Y=row)."
            )
            panel.connect_contrast_changed(
                partial(self._on_map_contrast_changed, key)
            )
        map_layout.addWidget(self._map_panel_a, stretch=1)
        map_layout.addWidget(self._map_panel_b, stretch=1)
        map_layout.addWidget(self._map_panel_d, stretch=1)
        layout.addLayout(map_layout)
        self._connect_map_view_sync()

        probe_layout = QtWidgets.QHBoxLayout()
        probe_layout.setContentsMargins(0, 0, 0, 0)
        probe_layout.setSpacing(6)
        self._probe_table = QtWidgets.QTableWidget()
        self._probe_table.setColumnCount(5)
        self._probe_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
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
        pattern_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        pattern_layout = QtWidgets.QHBoxLayout(pattern_group)
        self._pattern_panel_a = PatternPanel("Pattern A")
        self._pattern_panel_b = PatternPanel("Pattern B")
        self._pattern_panel_d = PatternPanel("Pattern Δ/Ratio")
        self._pattern_panels = {
            "A": self._pattern_panel_a,
            "B": self._pattern_panel_b,
            "D": self._pattern_panel_d,
        }
        for panel in self._pattern_panels.values():
            panel.canvas().setToolTip(
                "Pattern view. Use the toolbar overlay to zoom/pan (synced)."
            )
        pattern_layout.addWidget(self._pattern_panel_a, stretch=1)
        pattern_layout.addWidget(self._pattern_panel_b, stretch=1)
        pattern_layout.addWidget(self._pattern_panel_d, stretch=1)
        probe_layout.addWidget(pattern_group)
        probe_layout.setStretch(0, 1)
        probe_layout.setStretch(1, 3)
        self._pattern_group = pattern_group
        layout.addLayout(probe_layout)
        self._connect_pattern_view_sync()
        layout.setStretch(0, 0)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)
        layout.setStretch(3, 6)
        layout.setStretch(4, 4)
        self.setCentralWidget(central_widget)
        log_config = self._config.get("logging", {})
        max_lines = int(log_config.get("gui_max_lines", 1000))
        self._log_viewer = LogViewer(max_lines=max_lines)
        log_dock = QtWidgets.QDockWidget("Log Console", self)
        log_dock.setObjectName("log_console")
        log_dock.setWidget(self._log_viewer)
        log_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        log_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        log_dock.setMinimumHeight(180)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, log_dock)
        self._log_dock = log_dock
        QtCore.QTimer.singleShot(0, self._resize_log_console)

    def _attach_log_handler(self) -> None:
        """Attach the GUI log handler to the root logger.

        Returns:
            None.
        """

        if self._log_viewer is None:
            return
        log_config = self._config.get("logging", {})
        emitter = LogEmitter()
        emitter.message.connect(self._log_viewer.append_entry)
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

    def _resize_log_console(self) -> None:
        """Resize the log console dock to a reasonable default height.

        Returns:
            None.
        """

        if self._log_dock is None:
            return
        target = int(max(self.height(), 600) * 0.22)
        self.resizeDocks([self._log_dock], [target], QtCore.Qt.Vertical)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window close events.

        Parameters:
            event: Qt close event instance.

        Returns:
            None.
        """

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        if self._auto_scan_controller is not None:
            self._auto_scan_controller.stop()
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
        self._logger.info("Loaded scan-A: %s", dataset_a.file_path)
        self._logger.info(
            "Detected scan-A shape: (%s x %s)", dataset_a.nx, dataset_a.ny
        )
        self._logger.info("Loaded scan-B: %s", dataset_b.file_path)
        self._logger.info(
            "Detected scan-B shape: (%s x %s)", dataset_b.nx, dataset_b.ny
        )
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
                self._logger.error("Scan grids mismatch and no alignment provided.")
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
        self._configure_coordinate_inputs(self._scan_a.nx, self._scan_a.ny)
        if self._selection_controller is not None:
            self._selection_controller.set_bounds(self._scan_a.nx, self._scan_a.ny)
        if self._auto_scan_controller is not None:
            self._auto_scan_controller.set_bounds(self._scan_a.nx, self._scan_a.ny)
            self._auto_scan_controller.set_interval_ms(self._auto_speed_spin.value())
        if self._alignment_result is not None:
            self._logger.info(
                "Alignment active: rotation=%.3f deg, translation=(%.3f, %.3f)",
                self._alignment_result.rotation_deg,
                self._alignment_result.translation[0],
                self._alignment_result.translation[1],
            )
        self._populate_map_fields()
        self._select_pattern_field()
        self._pattern_reset_view = True
        self._log_missing_fields()
        self._update_maps()
        x, y = self._engine.default_probe_xy()
        self.set_selected_pixel(x, y, source="init")

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

        paths = self._validate_scan_paths()
        if paths is None:
            return
        self.load_scans(paths[0], paths[1])

    def _validate_scan_paths(self) -> Optional[Tuple[Path, Path]]:
        """Validate scan path inputs from the GUI.

        Returns:
            Tuple of (path_a, path_b) if valid, otherwise None.
        """

        text_a = self._file_a_edit.text().strip()
        text_b = self._file_b_edit.text().strip()
        errors = []
        self._set_line_edit_error_state(self._file_a_edit, False)
        self._set_line_edit_error_state(self._file_b_edit, False)
        if not text_a:
            errors.append("Scan A path is required.")
            self._set_line_edit_error_state(self._file_a_edit, True)
        if not text_b:
            errors.append("Scan B path is required.")
            self._set_line_edit_error_state(self._file_b_edit, True)
        if errors:
            message = " ".join(errors)
            self._set_inline_error(message)
            self._logger.error(message)
            self.statusBar().showMessage(message, 5000)
            return None
        path_a = Path(text_a)
        path_b = Path(text_b)
        if not path_a.exists():
            message = f"Scan A file not found: {path_a}"
            self._set_inline_error(message)
            self._set_line_edit_error_state(self._file_a_edit, True)
            self._logger.error(message)
            self.statusBar().showMessage(message, 5000)
            return None
        if not path_b.exists():
            message = f"Scan B file not found: {path_b}"
            self._set_inline_error(message)
            self._set_line_edit_error_state(self._file_b_edit, True)
            self._logger.error(message)
            self.statusBar().showMessage(message, 5000)
            return None
        self._clear_inline_error()
        return path_a, path_b

    def _clear_file_error(self) -> None:
        """Clear file input error feedback.

        Returns:
            None.
        """

        self._clear_inline_error()
        self._set_line_edit_error_state(self._file_a_edit, False)
        self._set_line_edit_error_state(self._file_b_edit, False)

    def _set_line_edit_error_state(
        self, widget: QtWidgets.QLineEdit, has_error: bool
    ) -> None:
        """Apply or clear error styling on a line edit.

        Parameters:
            widget: Line edit to style.
            has_error: Whether to apply error styling.

        Returns:
            None.
        """

        if has_error:
            widget.setStyleSheet("border: 1px solid #c62828;")
        else:
            widget.setStyleSheet("")

    def _set_inline_error(self, message: str) -> None:
        """Set the inline error label message.

        Parameters:
            message: Error message to display.

        Returns:
            None.
        """

        if self._inline_error_label is not None:
            self._inline_error_label.setText(message)

    def _clear_inline_error(self) -> None:
        """Clear the inline error label message.

        Returns:
            None.
        """

        if self._inline_error_label is not None:
            self._inline_error_label.setText("")

    def _configure_coordinate_inputs(self, width: int, height: int) -> None:
        """Configure coordinate input widgets for the active scan.

        Parameters:
            width: Number of columns in scan A.
            height: Number of rows in scan A.

        Returns:
            None.
        """

        self._x_validator = QtGui.QIntValidator(0, width - 1)
        self._y_validator = QtGui.QIntValidator(0, height - 1)
        self._x_input.setValidator(self._x_validator)
        self._y_input.setValidator(self._y_validator)
        self._x_input.setEnabled(True)
        self._y_input.setEnabled(True)
        self._auto_play_button.setEnabled(True)
        self._auto_pause_button.setEnabled(False)
        self._auto_stop_button.setEnabled(False)
        self._auto_pause_button.setText("Pause")
        self._auto_speed_spin.setEnabled(True)
        example_x = min(10, width - 1)
        example_y = min(10, height - 1)
        default_x = width // 2
        default_y = height // 2
        self._x_input.setToolTip(
            f"Column index (X). Range: 0 to {width - 1}. "
            f"Default: {default_x}. Example: {example_x}."
        )
        self._y_input.setToolTip(
            f"Row index (Y). Range: 0 to {height - 1}. "
            f"Default: {default_y}. Example: {example_y}."
        )
        self._clear_inline_error()

    def _on_coord_edit_finished(self) -> None:
        """Validate coordinate edits and update the selection.

        Returns:
            None.
        """

        if self._syncing_coords:
            return
        if self._engine is None or self._scan_a is None:
            self._show_coord_error("Load scans before setting coordinates.")
            return
        x_result = validate_int_in_range(
            self._x_input.text(), 0, self._scan_a.nx - 1, "X (col)"
        )
        y_result = validate_int_in_range(
            self._y_input.text(), 0, self._scan_a.ny - 1, "Y (row)"
        )
        self._set_line_edit_error_state(self._x_input, not x_result.is_valid())
        self._set_line_edit_error_state(self._y_input, not y_result.is_valid())
        if not x_result.is_valid():
            self._logger.error(
                "Invalid X=%s (max=%s).", self._x_input.text(), self._scan_a.nx - 1
            )
            self._show_coord_error(x_result.error or "Invalid X input.")
            return
        if not y_result.is_valid():
            self._logger.error(
                "Invalid Y=%s (max=%s).", self._y_input.text(), self._scan_a.ny - 1
            )
            self._show_coord_error(y_result.error or "Invalid Y input.")
            return
        self._clear_coord_error()
        self.set_selected_pixel(x_result.value, y_result.value, source="manual input")

    def _show_coord_error(self, message: str) -> None:
        """Show an inline coordinate validation error.

        Parameters:
            message: Error message to display.

        Returns:
            None.
        """

        self._set_inline_error(message)
        self.statusBar().showMessage(message, 5000)

    def _clear_coord_error(self) -> None:
        """Clear coordinate validation error styling and text.

        Returns:
            None.
        """

        self._clear_inline_error()
        self._set_line_edit_error_state(self._x_input, False)
        self._set_line_edit_error_state(self._y_input, False)

    def set_selected_pixel(self, x: int, y: int, source: str) -> None:
        """Centralized selection update entry point.

        Parameters:
            x: Column index.
            y: Row index.
            source: Description of the update source.

        Returns:
            None.
        """

        if self._selection_controller is None:
            return
        self._selection_controller.set_selected_pixel(x, y, source)

    def _apply_selection_update(self, state: SelectionState) -> None:
        """Apply a validated selection update to the GUI.

        Parameters:
            state: SelectionState describing the update.

        Returns:
            None.
        """

        if self._engine is None:
            return
        self._clear_coord_error()
        self._syncing_coords = True
        try:
            self._x_input.setText(str(state.x))
            self._y_input.setText(str(state.y))
        finally:
            self._syncing_coords = False
        for panel in self._map_panels.values():
            panel.canvas().set_marker(state.x, state.y)
        self._update_probe(state.x, state.y)
        message = f"Selected X={state.x}, Y={state.y}."
        self.statusBar().showMessage(message, 5000)
        if self._inline_status_label is not None:
            self._inline_status_label.setText(f"X={state.x}, Y={state.y}")
        if state.source == "auto_scan":
            self._logger.debug(message)
        else:
            self._logger.info(message)

    def _on_auto_scan_play(self) -> None:
        """Start the auto-scan animation.

        Returns:
            None.
        """

        if self._engine is None or self._auto_scan_controller is None:
            self._show_auto_scan_error("Load scans before starting auto-scan.")
            return
        self._clear_auto_scan_error()
        self._auto_scan_controller.play()
        self._update_auto_scan_controls()
        if self._auto_scan_controller.is_running():
            self.statusBar().showMessage("Auto-scan running.", 3000)
        else:
            self._show_auto_scan_error("Auto-scan could not start.")

    def _on_auto_scan_pause_resume(self) -> None:
        """Pause or resume the auto-scan animation.

        Returns:
            None.
        """

        if self._auto_scan_controller is None:
            return
        if self._auto_scan_controller.is_paused():
            self._auto_scan_controller.resume()
            self.statusBar().showMessage("Auto-scan resumed.", 3000)
        else:
            self._auto_scan_controller.pause()
            self.statusBar().showMessage("Auto-scan paused.", 3000)
        self._update_auto_scan_controls()

    def _on_auto_scan_stop(self) -> None:
        """Stop the auto-scan animation.

        Returns:
            None.
        """

        if self._auto_scan_controller is None:
            return
        self._auto_scan_controller.stop()
        self._update_auto_scan_controls()
        self.statusBar().showMessage("Auto-scan stopped.", 3000)

    def _on_auto_scan_speed_changed(self, value: int) -> None:
        """Validate and apply the auto-scan speed change.

        Parameters:
            value: Delay in milliseconds.

        Returns:
            None.
        """

        result = validate_speed_ms(value, self._auto_min_ms, self._auto_max_ms)
        if not result.is_valid():
            self._show_auto_scan_error(result.error or "Invalid auto-scan speed.")
            return
        self._clear_auto_scan_error()
        if self._auto_scan_controller is not None:
            self._auto_scan_controller.set_interval_ms(value)

    def _update_auto_scan_controls(self) -> None:
        """Update auto-scan control states based on controller status.

        Returns:
            None.
        """

        if self._auto_scan_controller is None:
            return
        running = self._auto_scan_controller.is_running()
        paused = self._auto_scan_controller.is_paused()
        self._auto_play_button.setEnabled(not running)
        self._auto_pause_button.setEnabled(running)
        self._auto_stop_button.setEnabled(running)
        if running:
            self._auto_pause_button.setText("Resume" if paused else "Pause")
        else:
            self._auto_pause_button.setText("Pause")

    def _show_auto_scan_error(self, message: str) -> None:
        """Show an inline auto-scan validation error.

        Parameters:
            message: Error message to display.

        Returns:
            None.
        """

        self._set_inline_error(message)
        self.statusBar().showMessage(message, 5000)
        self._logger.error(message)

    def _clear_auto_scan_error(self) -> None:
        """Clear auto-scan error feedback.

        Returns:
            None.
        """

        self._clear_inline_error()

    def _on_auto_scan_step(self, x: int, y: int) -> None:
        """Handle auto-scan step updates.

        Parameters:
            x: Column index.
            y: Row index.

        Returns:
            None.
        """

        self.set_selected_pixel(x, y, source="auto_scan")

    def _on_auto_scan_finished(self) -> None:
        """Handle auto-scan completion.

        Returns:
            None.
        """

        self._update_auto_scan_controls()
        message = "Auto-scan complete."
        self.statusBar().showMessage(message, 5000)
        self._logger.info(message)

    def _log_missing_fields(self) -> None:
        """Log warnings for configured fields missing from either scan.

        Returns:
            None.
        """

        if self._scan_a is None or self._scan_b is None:
            return
        compare_config = self._config.get("compare_fields", {})
        scalar_fields = compare_config.get("scalars", [])
        pattern_fields = compare_config.get("patterns", [])
        for field in scalar_fields:
            if field not in self._scan_a.catalog.scalars:
                self._logger.warning('Field "%s" missing in scan-A; choose another field.', field)
            if field not in self._scan_b.catalog.scalars:
                self._logger.warning('Field "%s" missing in scan-B; choose another field.', field)
        for field in pattern_fields:
            if field not in self._scan_a.catalog.patterns:
                self._logger.warning('Pattern field "%s" missing in scan-A; choose another field.', field)
            if field not in self._scan_b.catalog.patterns:
                self._logger.warning('Pattern field "%s" missing in scan-B; choose another field.', field)

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
        if self._auto_scan_controller is not None:
            self._auto_scan_controller.stop()
            self._update_auto_scan_controls()
        self._reset_display()

    def _reset_display(self) -> None:
        """Reset GUI panels to an empty state.

        Returns:
            None.
        """

        blank = np.zeros((1, 1))
        for panel in self._map_panels.values():
            panel.canvas().update_data(blank, cmap="gray", reset_view=True)
            panel.canvas().clear_marker()
            panel.clear_error()
        for panel in self._pattern_panels.values():
            panel.canvas().update_data(blank, cmap="gray", reset_view=True)
        self._map_field_combo.clear()
        self._probe_table.setRowCount(0)
        self._pattern_group.setVisible(False)
        self._map_triplet = None
        self._pattern_reset_view = True
        self._x_input.setText("")
        self._y_input.setText("")
        self._x_input.setEnabled(False)
        self._y_input.setEnabled(False)
        self._auto_play_button.setEnabled(False)
        self._auto_pause_button.setEnabled(False)
        self._auto_stop_button.setEnabled(False)
        self._auto_speed_spin.setEnabled(False)
        if self._inline_status_label is not None:
            self._inline_status_label.setText("X=--, Y=--")
        self._clear_inline_error()

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
        self.set_selected_pixel(x, y, source="alignment")
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
        if self._engine is None:
            self._pattern_group.setVisible(False)
            return
        if not pattern_config:
            common_patterns = sorted(
                set(self._scan_a.catalog.patterns.keys())
                & set(self._scan_b.catalog.patterns.keys())
            )
            if common_patterns:
                self._pattern_field = common_patterns[0]
                self._pattern_group.setVisible(True)
                self._logger.info(
                    'Pattern field not configured; using "%s".', self._pattern_field
                )
                return
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
        common_patterns = sorted(
            set(self._scan_a.catalog.patterns.keys())
            & set(self._scan_b.catalog.patterns.keys())
        )
        if common_patterns:
            self._pattern_field = common_patterns[0]
            self._pattern_group.setVisible(True)
            self._logger.warning(
                'Configured pattern fields not found; using "%s" instead.',
                self._pattern_field,
            )
            return
        self._pattern_group.setVisible(False)

    def _update_maps(self, *_: object) -> None:
        """Update the map panels based on current selection."""

        self._refresh_maps(reset_view=True)

    def _refresh_maps(self, reset_view: bool) -> None:
        """Refresh all map panels from the current field selection.

        Parameters:
            reset_view: Whether to reset the axes view.

        Returns:
            None.
        """

        if self._engine is None:
            return
        field_name = self._map_field_combo.currentText()
        if not field_name:
            return
        mode = self._config.get("display", {}).get("map_diff_mode", "delta")
        self._map_triplet = self._engine.map_triplet(field_name, mode)
        self._refresh_map_panel("A", cmap="gray", reset_view=reset_view)
        self._refresh_map_panel("B", cmap="gray", reset_view=reset_view)
        self._refresh_map_panel("D", cmap="gray", reset_view=reset_view)

    def _refresh_map_panel(self, key: str, cmap: str, reset_view: bool) -> None:
        """Refresh a single map panel using cached map data.

        Parameters:
            key: Map key ("A", "B", or "D").
            cmap: Matplotlib colormap name.
            reset_view: Whether to reset the axes view.

        Returns:
            None.
        """

        if self._map_triplet is None:
            return
        panel = self._map_panels.get(key)
        if panel is None:
            return
        data = self._map_triplet.get(key)
        if data is None:
            return
        low_pct, high_pct = panel.contrast_values()
        if high_pct <= low_pct:
            message = "Contrast high must be greater than low."
            panel.set_error(message)
            self.statusBar().showMessage(message, 5000)
            high_pct = min(100.0, low_pct + 1.0)
            panel.set_contrast_values(low_pct, high_pct, block_signals=True)
        else:
            panel.clear_error()
        vmin, vmax = self._calculate_limits(data, low_pct, high_pct)
        panel.canvas().update_data(
            data, cmap=cmap, vmin=vmin, vmax=vmax, reset_view=reset_view
        )

    def _on_map_contrast_changed(self, key: str) -> None:
        """Handle contrast control changes for a specific map panel.

        Parameters:
            key: Map key ("A", "B", or "D").

        Returns:
            None.
        """

        if self._engine is None:
            return
        if self._map_triplet is None:
            self._refresh_maps(reset_view=True)
            return
        self._refresh_map_panel(key, cmap="gray", reset_view=False)

    def _calculate_limits(
        self, data: np.ndarray, low_pct: float, high_pct: float
    ) -> Tuple[float, float]:
        """Compute contrast limits from percentiles.

        Parameters:
            data: Input data array.
            low_pct: Lower percentile.
            high_pct: Upper percentile.

        Returns:
            Tuple of (vmin, vmax).
        """

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(finite, low_pct))
        vmax = float(np.percentile(finite, high_pct))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _connect_map_view_sync(self) -> None:
        """Connect synchronized zoom/pan for the map panels.

        Returns:
            None.
        """

        for panel in self._map_panels.values():
            canvas = panel.canvas()
            canvas.axes.callbacks.connect(
                "xlim_changed", partial(self._sync_map_view, canvas)
            )
            canvas.axes.callbacks.connect(
                "ylim_changed", partial(self._sync_map_view, canvas)
            )

    def _sync_map_view(self, source: MapCanvas, *_: object) -> None:
        """Synchronize map view limits across all panels.

        Parameters:
            source: Canvas that initiated the view change.

        Returns:
            None.
        """

        if self._syncing_map_view:
            return
        self._syncing_map_view = True
        try:
            xlim = source.axes.get_xlim()
            ylim = source.axes.get_ylim()
            for panel in self._map_panels.values():
                canvas = panel.canvas()
                if canvas is source:
                    continue
                canvas.axes.set_xlim(xlim)
                canvas.axes.set_ylim(ylim)
                canvas.draw_idle()
        finally:
            self._syncing_map_view = False

    def _connect_pattern_view_sync(self) -> None:
        """Connect synchronized zoom/pan for the pattern panels.

        Returns:
            None.
        """

        for panel in self._pattern_panels.values():
            canvas = panel.canvas()
            canvas.axes.callbacks.connect(
                "xlim_changed", partial(self._sync_pattern_view, canvas)
            )
            canvas.axes.callbacks.connect(
                "ylim_changed", partial(self._sync_pattern_view, canvas)
            )

    def _sync_pattern_view(self, source: MapCanvas, *_: object) -> None:
        """Synchronize pattern view limits across all panels.

        Parameters:
            source: Canvas that initiated the view change.

        Returns:
            None.
        """

        if self._syncing_pattern_view:
            return
        self._syncing_pattern_view = True
        try:
            xlim = source.axes.get_xlim()
            ylim = source.axes.get_ylim()
            for panel in self._pattern_panels.values():
                canvas = panel.canvas()
                if canvas is source:
                    continue
                canvas.axes.set_xlim(xlim)
                canvas.axes.set_ylim(ylim)
                canvas.draw_idle()
        finally:
            self._syncing_pattern_view = False

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
        self.set_selected_pixel(x, y, source="map_click")

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
        if not pattern_triplet:
            self._logger.warning(
                'Pattern field "%s" returned no data at X=%s, Y=%s.',
                self._pattern_field,
                x,
                y,
            )
            return
        pattern_a = pattern_triplet.get("A")
        pattern_b = pattern_triplet.get("B")
        pattern_d = pattern_triplet.get("D")
        if pattern_a is None:
            self._logger.warning(
                'Pattern A missing for field "%s" at X=%s, Y=%s.',
                self._pattern_field,
                x,
                y,
            )
            return
        reset_view = self._pattern_reset_view
        self._pattern_panels["A"].canvas().update_data(
            pattern_a, cmap="gray", reset_view=reset_view
        )
        if pattern_b is None or pattern_d is None:
            placeholder = np.zeros_like(pattern_a)
            self._pattern_panels["B"].canvas().update_data(
                placeholder, cmap="gray", reset_view=reset_view
            )
            self._pattern_panels["D"].canvas().update_data(
                placeholder, cmap="gray", reset_view=reset_view
            )
        else:
            self._pattern_panels["B"].canvas().update_data(
                pattern_b, cmap="gray", reset_view=reset_view
            )
            self._pattern_panels["D"].canvas().update_data(
                pattern_d, cmap="gray", reset_view=reset_view
            )
        self._pattern_reset_view = False
        self._log_pattern_update("A", pattern_a, x, y)
        self._log_pattern_update("B", pattern_b, x, y)
        self._log_pattern_update("D", pattern_d, x, y)

    def _log_pattern_update(
        self, label: str, pattern: Optional[np.ndarray], x: int, y: int
    ) -> None:
        """Log pattern update details for debugging.

        Parameters:
            label: Pattern label ("A", "B", or "D").
            pattern: Pattern array or None.
            x: Column index.
            y: Row index.

        Returns:
            None.
        """

        if pattern is None:
            self._logger.warning("Pattern %s missing at X=%s, Y=%s.", label, x, y)
            return
        finite = pattern[np.isfinite(pattern)]
        if finite.size == 0:
            min_val = float("nan")
            max_val = float("nan")
        else:
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
        level = logging.DEBUG
        if self._auto_scan_controller is None or not self._auto_scan_controller.is_running():
            level = logging.INFO
        self._logger.log(
            level,
            "Pattern %s updated at X=%s, Y=%s; shape=%s dtype=%s min=%.4g max=%.4g",
            label,
            x,
            y,
            pattern.shape,
            pattern.dtype,
            min_val,
            max_val,
        )
        if finite.size > 0 and np.allclose(finite, 0):
            self._logger.warning("Pattern %s appears to be all zeros at X=%s, Y=%s.", label, x, y)

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
    apply_input_border_styles(app)
    window = EbsdCompareMainWindow(args.config)
    window.show()
    if args.scan_a and args.scan_b:
        window.load_scans(args.scan_a, args.scan_b)
    elif args.debug:
        factory = SimulatedScanFactory.from_config(config.get("debug", {}), logger)
        scan_a, scan_b = factory.create_pair()
        window.load_scan_datasets(scan_a, scan_b)
    app.exec()


def apply_input_border_styles(app: QtWidgets.QApplication) -> None:
    """Apply a minimal stylesheet to keep input borders visible on dark themes.

    Parameters:
        app: Application instance to style.

    Returns:
        None.
    """

    app.setStyleSheet(
        """
        QLineEdit,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
        QAbstractSpinBox {
            border: 1px solid #8a8a8a;
            border-radius: 2px;
        }
        QLineEdit:focus,
        QComboBox:focus,
        QSpinBox:focus,
        QDoubleSpinBox:focus,
        QAbstractSpinBox:focus {
            border: 1px solid #4a90e2;
        }
        """
    )


if __name__ == "__main__":
    main()
