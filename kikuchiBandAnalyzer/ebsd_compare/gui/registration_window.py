"""Registration dialog for aligning mismatched EBSD scans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    AlignmentResult,
    AlignmentSettings,
    alignment_settings_from_config,
    compute_residuals,
    estimate_alignment,
    save_alignment_to_yaml,
)

matplotlib.use("QtAgg")


@dataclass
class PointPair:
    """Container for a matched point pair.

    Parameters:
        index: Sequential index for the point pair.
        point_a: Coordinate in scan A.
        point_b: Coordinate in scan B.
        inlier: Whether the pair is an inlier after RANSAC.
        residual: Residual distance for the pair.
    """

    index: int
    point_a: Tuple[float, float]
    point_b: Tuple[float, float]
    inlier: Optional[bool] = None
    residual: Optional[float] = None


class RegistrationMapCanvas(FigureCanvas):
    """Matplotlib canvas for displaying registration maps."""

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
        self._marker_artists: List = []
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

    def update_data(self, data: np.ndarray, vmin: float, vmax: float, cmap: str) -> None:
        """Update the displayed image data.

        Parameters:
            data: 2D array to display.
            vmin: Lower contrast limit.
            vmax: Upper contrast limit.
            cmap: Matplotlib colormap name.

        Returns:
            None.
        """

        if self._image is None:
            self._image = self._axes.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            self._image.set_data(data)
            self._image.set_clim(vmin, vmax)
        self._axes.set_title(self._title)
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._figure.tight_layout()
        self.draw_idle()

    def update_markers(
        self,
        points: List[Tuple[float, float]],
        labels: List[str],
        inliers: Optional[List[Optional[bool]]],
        selected_index: Optional[int],
        pending_point: Optional[Tuple[float, float]],
    ) -> None:
        """Update point markers and labels.

        Parameters:
            points: List of point coordinates to display.
            labels: Text labels for each point.
            inliers: Optional inlier flags for each point.
            selected_index: Index of the selected point.
            pending_point: Pending point coordinate awaiting a match.

        Returns:
            None.
        """

        for artist in self._marker_artists:
            artist.remove()
        self._marker_artists.clear()
        for idx, (point, label) in enumerate(zip(points, labels)):
            color = "#ffa500"
            if inliers and inliers[idx] is True:
                color = "#2ca02c"
            elif inliers and inliers[idx] is False:
                color = "#d62728"
            size = 40
            if selected_index is not None and idx == selected_index:
                size = 80
                color = "#1f77b4"
            marker = self._axes.scatter(point[0], point[1], s=size, c=color, marker="o")
            text = self._axes.text(
                point[0],
                point[1],
                label,
                color=color,
                fontsize=9,
                ha="left",
                va="bottom",
            )
            self._marker_artists.extend([marker, text])
        if pending_point is not None:
            pending = self._axes.scatter(
                pending_point[0],
                pending_point[1],
                s=80,
                c="#ffdd00",
                marker="x",
            )
            self._marker_artists.append(pending)
        self.draw_idle()

    def connect_click(self, handler: QtCore.Slot) -> None:
        """Connect a click handler to the canvas.

        Parameters:
            handler: Matplotlib event handler.

        Returns:
            None.
        """

        self.mpl_connect("button_press_event", handler)

    def connect_motion(self, handler: QtCore.Slot) -> None:
        """Connect a motion handler to the canvas.

        Parameters:
            handler: Matplotlib event handler.

        Returns:
            None.
        """

        self.mpl_connect("motion_notify_event", handler)


class RegistrationDialog(QtWidgets.QDialog):
    """Dialog for selecting control points and computing alignment."""

    def __init__(
        self,
        scan_a: ScanDataset,
        scan_b: ScanDataset,
        config: Dict,
        logger: logging.Logger,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_alignment: Optional[AlignmentResult] = None,
    ) -> None:
        """Initialize the registration dialog.

        Parameters:
            scan_a: Scan dataset A.
            scan_b: Scan dataset B.
            config: EBSD compare configuration dictionary.
            logger: Logger instance.
            parent: Optional parent widget.
            initial_alignment: Optional initial alignment result to display.
        """

        super().__init__(parent=parent)
        self._scan_a = scan_a
        self._scan_b = scan_b
        self._config = config
        self._alignment_config = config.get("alignment", {})
        self._settings = alignment_settings_from_config(self._alignment_config)
        self._logger = logger
        self._points: List[PointPair] = []
        self._pending_point_a: Optional[Tuple[float, float]] = None
        self._alignment_result: Optional[AlignmentResult] = initial_alignment
        self._selected_index: Optional[int] = None
        self._syncing_view = False
        self._init_ui()
        self._populate_fields()
        self._update_maps()
        if initial_alignment is not None:
            self._update_alignment_summary(initial_alignment)

    def alignment_result(self) -> Optional[AlignmentResult]:
        """Return the computed alignment result.

        Returns:
            AlignmentResult instance if available, otherwise None.
        """

        return self._alignment_result

    def _init_ui(self) -> None:
        """Initialize the registration UI widgets."""

        Returns:
            None.

        self.setWindowTitle("EBSD Registration")
        layout = QtWidgets.QVBoxLayout(self)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(QtWidgets.QLabel("Registration Field"))
        self._field_combo = QtWidgets.QComboBox()
        self._field_combo.currentTextChanged.connect(self._update_maps)
        header_layout.addWidget(self._field_combo)
        self._link_contrast_checkbox = QtWidgets.QCheckBox("Link Contrast")
        self._link_contrast_checkbox.setChecked(
            bool(self._alignment_config.get("contrast", {}).get("link_maps", True))
        )
        self._link_contrast_checkbox.stateChanged.connect(self._update_maps)
        header_layout.addWidget(self._link_contrast_checkbox)
        self._link_view_checkbox = QtWidgets.QCheckBox("Link View")
        self._link_view_checkbox.setChecked(
            bool(self._alignment_config.get("link_view", True))
        )
        header_layout.addWidget(self._link_view_checkbox)
        header_layout.addStretch(1)
        layout.addLayout(header_layout)

        maps_layout = QtWidgets.QHBoxLayout()
        self._canvas_a = RegistrationMapCanvas("Scan A")
        self._canvas_b = RegistrationMapCanvas("Scan B")
        self._canvas_a.connect_click(self._on_click_a)
        self._canvas_b.connect_click(self._on_click_b)
        self._canvas_a.axes.callbacks.connect("xlim_changed", self._sync_view_from_a)
        self._canvas_a.axes.callbacks.connect("ylim_changed", self._sync_view_from_a)
        self._canvas_b.axes.callbacks.connect("xlim_changed", self._sync_view_from_b)
        self._canvas_b.axes.callbacks.connect("ylim_changed", self._sync_view_from_b)
        maps_layout.addWidget(self._build_canvas_panel(self._canvas_a))
        maps_layout.addWidget(self._build_canvas_panel(self._canvas_b))
        layout.addLayout(maps_layout)

        coord_layout = QtWidgets.QHBoxLayout()
        self._coord_a = QtWidgets.QLineEdit()
        self._coord_a.setReadOnly(True)
        self._coord_b = QtWidgets.QLineEdit()
        self._coord_b.setReadOnly(True)
        coord_layout.addWidget(QtWidgets.QLabel("Selected A"))
        coord_layout.addWidget(self._coord_a)
        coord_layout.addWidget(QtWidgets.QLabel("Selected B"))
        coord_layout.addWidget(self._coord_b)
        layout.addLayout(coord_layout)

        contrast_layout = QtWidgets.QHBoxLayout()
        contrast_layout.addWidget(QtWidgets.QLabel("Contrast Low/High %"))
        self._contrast_low = QtWidgets.QDoubleSpinBox()
        self._contrast_low.setRange(0.0, 100.0)
        self._contrast_low.setDecimals(1)
        self._contrast_low.setSingleStep(0.5)
        self._contrast_high = QtWidgets.QDoubleSpinBox()
        self._contrast_high.setRange(0.0, 100.0)
        self._contrast_high.setDecimals(1)
        self._contrast_high.setSingleStep(0.5)
        contrast_config = self._alignment_config.get("contrast", {})
        self._contrast_low.setValue(float(contrast_config.get("low_percentile", 2.0)))
        self._contrast_high.setValue(float(contrast_config.get("high_percentile", 98.0)))
        self._contrast_low.valueChanged.connect(self._update_maps)
        self._contrast_high.valueChanged.connect(self._update_maps)
        contrast_layout.addWidget(self._contrast_low)
        contrast_layout.addWidget(self._contrast_high)
        contrast_layout.addStretch(1)
        layout.addLayout(contrast_layout)

        point_layout = QtWidgets.QHBoxLayout()
        self._point_table = QtWidgets.QTableWidget()
        self._point_table.setColumnCount(7)
        self._point_table.setHorizontalHeaderLabels(
            ["ID", "Ax", "Ay", "Bx", "By", "Inlier", "Residual"]
        )
        self._point_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._point_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self._point_table.itemSelectionChanged.connect(self._on_table_selection)
        point_layout.addWidget(self._point_table, stretch=3)
        edit_layout = QtWidgets.QVBoxLayout()
        edit_layout.addWidget(QtWidgets.QLabel("Edit Selected Pair"))
        self._edit_ax = QtWidgets.QDoubleSpinBox()
        self._edit_ax.setDecimals(2)
        self._edit_ay = QtWidgets.QDoubleSpinBox()
        self._edit_ay.setDecimals(2)
        self._edit_bx = QtWidgets.QDoubleSpinBox()
        self._edit_bx.setDecimals(2)
        self._edit_by = QtWidgets.QDoubleSpinBox()
        self._edit_by.setDecimals(2)
        for widget in (self._edit_ax, self._edit_ay, self._edit_bx, self._edit_by):
            widget.valueChanged.connect(self._on_edit_change)
        edit_layout.addWidget(QtWidgets.QLabel("A.x"))
        edit_layout.addWidget(self._edit_ax)
        edit_layout.addWidget(QtWidgets.QLabel("A.y"))
        edit_layout.addWidget(self._edit_ay)
        edit_layout.addWidget(QtWidgets.QLabel("B.x"))
        edit_layout.addWidget(self._edit_bx)
        edit_layout.addWidget(QtWidgets.QLabel("B.y"))
        edit_layout.addWidget(self._edit_by)
        edit_layout.addStretch(1)
        point_layout.addLayout(edit_layout, stretch=1)
        layout.addLayout(point_layout)

        ransac_layout = QtWidgets.QHBoxLayout()
        ransac_layout.addWidget(QtWidgets.QLabel("RANSAC min samples"))
        self._ransac_min_samples = QtWidgets.QSpinBox()
        self._ransac_min_samples.setRange(2, 20)
        self._ransac_min_samples.setValue(self._settings.ransac_min_samples)
        ransac_layout.addWidget(self._ransac_min_samples)
        ransac_layout.addWidget(QtWidgets.QLabel("Residual threshold"))
        self._ransac_threshold = QtWidgets.QDoubleSpinBox()
        self._ransac_threshold.setRange(0.1, 50.0)
        self._ransac_threshold.setDecimals(2)
        self._ransac_threshold.setValue(self._settings.ransac_residual_threshold)
        ransac_layout.addWidget(self._ransac_threshold)
        ransac_layout.addWidget(QtWidgets.QLabel("Max trials"))
        self._ransac_trials = QtWidgets.QSpinBox()
        self._ransac_trials.setRange(100, 10000)
        self._ransac_trials.setValue(self._settings.ransac_max_trials)
        ransac_layout.addWidget(self._ransac_trials)
        ransac_layout.addStretch(1)
        layout.addLayout(ransac_layout)

        action_layout = QtWidgets.QHBoxLayout()
        self._status_label = QtWidgets.QLabel(
            "Click a point in Scan A, then the corresponding point in Scan B."
        )
        action_layout.addWidget(self._status_label)
        action_layout.addStretch(1)
        self._undo_button = QtWidgets.QPushButton("Undo Last")
        self._undo_button.clicked.connect(self._undo_last_point)
        action_layout.addWidget(self._undo_button)
        self._delete_button = QtWidgets.QPushButton("Delete Selected")
        self._delete_button.clicked.connect(self._delete_selected_point)
        action_layout.addWidget(self._delete_button)
        self._clear_button = QtWidgets.QPushButton("Clear All")
        self._clear_button.clicked.connect(self._clear_points)
        action_layout.addWidget(self._clear_button)
        self._compute_button = QtWidgets.QPushButton("Compute Alignment")
        self._compute_button.clicked.connect(self._compute_alignment)
        action_layout.addWidget(self._compute_button)
        self._apply_button = QtWidgets.QPushButton("Apply Alignment")
        self._apply_button.clicked.connect(self._apply_alignment)
        action_layout.addWidget(self._apply_button)
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        action_layout.addWidget(self._cancel_button)
        layout.addLayout(action_layout)

        self._summary_label = QtWidgets.QLabel("Alignment summary: pending")
        layout.addWidget(self._summary_label)

    def _build_canvas_panel(self, canvas: RegistrationMapCanvas) -> QtWidgets.QWidget:
        """Build a widget containing a map canvas and toolbar.

        Parameters:
            canvas: RegistrationMapCanvas instance.

        Returns:
            QWidget with canvas and toolbar.
        """

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(canvas)
        layout.addWidget(NavigationToolbar2QT(canvas, panel))
        return panel

    def _populate_fields(self) -> None:
        """Populate the map field dropdown.

        Returns:
            None.
        """

        fields = sorted(
            set(self._scan_a.catalog.list_scalar_fields())
            & set(self._scan_b.catalog.list_scalar_fields())
        )
        preferred_fields = self._alignment_config.get("map_fields", [])
        ordered = [field for field in preferred_fields if field in fields]
        for field in fields:
            if field not in ordered:
                ordered.append(field)
        self._field_combo.addItems(ordered)
        default_field = self._alignment_config.get("default_map_field")
        if default_field and default_field in ordered:
            self._field_combo.setCurrentText(default_field)

    def _update_maps(self) -> None:
        """Update the registration map panels.

        Returns:
            None.
        """

        field = self._field_combo.currentText()
        if not field:
            return
        map_a = self._scan_a.get_map(field)
        map_b = self._scan_b.get_map(field)
        low_pct = self._contrast_low.value()
        high_pct = self._contrast_high.value()
        if high_pct <= low_pct:
            high_pct = low_pct + 1.0
        if self._link_contrast_checkbox.isChecked():
            vmin, vmax = self._calculate_limits(
                np.concatenate([map_a.flatten(), map_b.flatten()]), low_pct, high_pct
            )
            limits_a = limits_b = (vmin, vmax)
        else:
            limits_a = self._calculate_limits(map_a, low_pct, high_pct)
            limits_b = self._calculate_limits(map_b, low_pct, high_pct)
        self._canvas_a.update_data(map_a, limits_a[0], limits_a[1], cmap="viridis")
        self._canvas_b.update_data(map_b, limits_b[0], limits_b[1], cmap="viridis")
        self._refresh_markers()

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

    def _on_click_a(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Handle clicks on scan A map.

        Returns:
            None.
        """

        point = self._event_to_point(event, self._scan_a)
        if point is None:
            return
        self._pending_point_a = point
        self._coord_a.setText(f"({point[0]:.2f}, {point[1]:.2f})")
        self._status_label.setText("Select the matching point in Scan B.")
        self._refresh_markers()

    def _on_click_b(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Handle clicks on scan B map.

        Returns:
            None.
        """

        point = self._event_to_point(event, self._scan_b)
        if point is None:
            return
        self._coord_b.setText(f"({point[0]:.2f}, {point[1]:.2f})")
        if self._pending_point_a is None:
            self._status_label.setText("Select a point in Scan A first.")
            return
        index = len(self._points) + 1
        self._points.append(
            PointPair(index=index, point_a=self._pending_point_a, point_b=point)
        )
        self._pending_point_a = None
        self._status_label.setText("Point pair added. Select another point in Scan A.")
        self._logger.debug(
            "Added point pair %s at A=%s B=%s", index, self._points[-1].point_a, point
        )
        self._invalidate_alignment()
        self._refresh_table()
        self._refresh_markers()

    def _event_to_point(
        self, event: matplotlib.backend_bases.MouseEvent, scan: ScanDataset
    ) -> Optional[Tuple[float, float]]:
        """Convert a Matplotlib event to a valid point.

        Parameters:
            event: Matplotlib mouse event.
            scan: Scan dataset for bounds checking.

        Returns:
            Point coordinate or None if invalid.
        """

        if event.xdata is None or event.ydata is None:
            return None
        x = float(event.xdata)
        y = float(event.ydata)
        if x < 0 or y < 0 or x > scan.nx - 1 or y > scan.ny - 1:
            return None
        return (x, y)

    def _refresh_table(self) -> None:
        """Refresh the points table.

        Returns:
            None.
        """

        self._renumber_points()
        self._point_table.setRowCount(len(self._points))
        for row, pair in enumerate(self._points):
            self._point_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(pair.index)))
            self._point_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{pair.point_a[0]:.2f}")
            )
            self._point_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(f"{pair.point_a[1]:.2f}")
            )
            self._point_table.setItem(
                row, 3, QtWidgets.QTableWidgetItem(f"{pair.point_b[0]:.2f}")
            )
            self._point_table.setItem(
                row, 4, QtWidgets.QTableWidgetItem(f"{pair.point_b[1]:.2f}")
            )
            inlier_text = ""
            if pair.inlier is True:
                inlier_text = "Yes"
            elif pair.inlier is False:
                inlier_text = "No"
            self._point_table.setItem(
                row, 5, QtWidgets.QTableWidgetItem(inlier_text)
            )
            residual_text = "" if pair.residual is None else f"{pair.residual:.3f}"
            self._point_table.setItem(
                row, 6, QtWidgets.QTableWidgetItem(residual_text)
            )
        self._point_table.resizeRowsToContents()

    def _refresh_markers(self) -> None:
        """Refresh markers on both canvases.

        Returns:
            None.
        """

        points_a = [pair.point_a for pair in self._points]
        points_b = [pair.point_b for pair in self._points]
        labels = [str(pair.index) for pair in self._points]
        inliers = [pair.inlier for pair in self._points]
        self._canvas_a.update_markers(
            points_a, labels, inliers, self._selected_index, self._pending_point_a
        )
        self._canvas_b.update_markers(
            points_b, labels, inliers, self._selected_index, None
        )

    def _on_table_selection(self) -> None:
        """Handle table selection changes.

        Returns:
            None.
        """

        selected = self._point_table.selectedIndexes()
        if not selected:
            self._selected_index = None
            self._refresh_markers()
            return
        row = selected[0].row()
        self._selected_index = row
        pair = self._points[row]
        self._set_edit_fields(pair)
        self._refresh_markers()

    def _set_edit_fields(self, pair: PointPair) -> None:
        """Populate edit fields for the selected point pair.

        Parameters:
            pair: PointPair instance to edit.

        Returns:
            None.
        """

        for widget, value in (
            (self._edit_ax, pair.point_a[0]),
            (self._edit_ay, pair.point_a[1]),
            (self._edit_bx, pair.point_b[0]),
            (self._edit_by, pair.point_b[1]),
        ):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)

    def _on_edit_change(self) -> None:
        """Handle edits to the selected point pair.

        Returns:
            None.
        """

        if self._selected_index is None:
            return
        pair = self._points[self._selected_index]
        pair.point_a = (self._edit_ax.value(), self._edit_ay.value())
        pair.point_b = (self._edit_bx.value(), self._edit_by.value())
        self._invalidate_alignment()
        self._refresh_table()
        self._refresh_markers()

    def _undo_last_point(self) -> None:
        """Remove the most recently added point pair.

        Returns:
            None.
        """

        if not self._points:
            return
        removed = self._points.pop()
        self._logger.info("Removed point pair %s", removed.index)
        self._invalidate_alignment()
        self._refresh_table()
        self._refresh_markers()

    def _delete_selected_point(self) -> None:
        """Delete the currently selected point pair.

        Returns:
            None.
        """

        if self._selected_index is None:
            return
        removed = self._points.pop(self._selected_index)
        self._logger.info("Deleted point pair %s", removed.index)
        self._selected_index = None
        self._invalidate_alignment()
        self._refresh_table()
        self._refresh_markers()

    def _clear_points(self) -> None:
        """Clear all selected point pairs.

        Returns:
            None.
        """

        self._points.clear()
        self._pending_point_a = None
        self._selected_index = None
        self._logger.info("Cleared all point pairs.")
        self._invalidate_alignment()
        self._refresh_table()
        self._refresh_markers()

    def _renumber_points(self) -> None:
        """Renumber point pairs sequentially.

        Returns:
            None.
        """

        for idx, pair in enumerate(self._points, start=1):
            pair.index = idx

    def _invalidate_alignment(self) -> None:
        """Clear alignment results after point edits.

        Returns:
            None.
        """

        self._alignment_result = None
        for pair in self._points:
            pair.inlier = None
            pair.residual = None
        self._summary_label.setText("Alignment summary: pending")

    def _compute_alignment(self) -> None:
        """Compute alignment using current point pairs.

        Returns:
            None.
        """

        if len(self._points) < self._settings.min_point_pairs:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient points",
                f"Select at least {self._settings.min_point_pairs} point pairs.",
            )
            return
        points_a = np.array([pair.point_a for pair in self._points], dtype=float)
        points_b = np.array([pair.point_b for pair in self._points], dtype=float)
        settings = alignment_settings_from_config(self._alignment_config)
        settings = AlignmentSettings(
            min_point_pairs=settings.min_point_pairs,
            ransac_min_samples=int(self._ransac_min_samples.value()),
            ransac_residual_threshold=float(self._ransac_threshold.value()),
            ransac_max_trials=int(self._ransac_trials.value()),
            warp_order=settings.warp_order,
            warp_mode=settings.warp_mode,
            warp_cval=settings.warp_cval,
            preserve_range=settings.preserve_range,
            pattern_sampling=settings.pattern_sampling,
        )
        try:
            self._alignment_result = estimate_alignment(
                points_a, points_b, settings, logger=self._logger
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Alignment failed", str(exc))
            self._logger.exception("Alignment failed.")
            return
        residuals = compute_residuals(
            points_a, points_b, self._alignment_result.transform
        )
        inliers = self._alignment_result.inliers
        for idx, pair in enumerate(self._points):
            pair.residual = float(residuals[idx])
            if inliers is not None:
                pair.inlier = bool(inliers[idx])
        self._refresh_table()
        self._refresh_markers()
        self._update_alignment_summary(self._alignment_result)
        output_path = self._alignment_config.get("save_alignment_path")
        if output_path:
            save_alignment_to_yaml(Path(output_path), self._alignment_result)
            self._logger.info("Saved alignment to %s", output_path)

    def _apply_alignment(self) -> None:
        """Accept the current alignment result.

        Returns:
            None.
        """

        if self._alignment_result is None:
            QtWidgets.QMessageBox.warning(
                self, "No alignment", "Compute an alignment first."
            )
            return
        self.accept()

    def _update_alignment_summary(self, result: AlignmentResult) -> None:
        """Update the alignment summary label.

        Parameters:
            result: AlignmentResult instance.

        Returns:
            None.
        """

        rms_text = "n/a" if result.rms_error is None else f"{result.rms_error:.4f}"
        self._summary_label.setText(
            "Alignment summary: rotation={:.3f} deg, translation=({:.3f}, {:.3f}), rms={}".format(
                result.rotation_deg,
                result.translation[0],
                result.translation[1],
                rms_text,
            )
        )

    def _sync_view_from_a(self, _axes) -> None:
        """Sync view limits from scan A to scan B.

        Returns:
            None.
        """

        if not self._link_view_checkbox.isChecked() or self._syncing_view:
            return
        self._syncing_view = True
        self._canvas_b.axes.set_xlim(self._canvas_a.axes.get_xlim())
        self._canvas_b.axes.set_ylim(self._canvas_a.axes.get_ylim())
        self._canvas_b.draw_idle()
        self._syncing_view = False

    def _sync_view_from_b(self, _axes) -> None:
        """Sync view limits from scan B to scan A.

        Returns:
            None.
        """

        if not self._link_view_checkbox.isChecked() or self._syncing_view:
            return
        self._syncing_view = True
        self._canvas_a.axes.set_xlim(self._canvas_b.axes.get_xlim())
        self._canvas_a.axes.set_ylim(self._canvas_b.axes.get_ylim())
        self._canvas_a.draw_idle()
        self._syncing_view = False
