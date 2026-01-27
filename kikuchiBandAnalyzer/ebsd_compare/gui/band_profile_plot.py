"""Qt widget for comparing exported band profile vectors.

The EBSD Comparator uses this widget to overlay band profiles from two scans
on shared axes, optionally normalizing each profile. Marker lines indicate the
band start/end indices and (optionally) the central peak index used by the
band-width pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.band_data import (
    BandProfilePayload,
    normalize_profile,
)


class BandProfilePlot(QtWidgets.QWidget):
    """Widget that renders band profile comparisons using Matplotlib."""

    def __init__(
        self,
        *,
        title: str = "Band Profile Comparison",
        label_a: str = "Scan A",
        label_b: str = "Scan B",
        marker_labels_include_series: bool = True,
        logger: Optional[logging.Logger] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the plot widget.

        Parameters:
            title: Plot title string.
            label_a: Display label for the first series.
            label_b: Display label for the second series.
            marker_labels_include_series: When True, marker legend labels include the
                series name (e.g., "Scan A start"). When False, markers use generic
                labels ("Band start/end/peak"), which is useful for single-scan views.
            logger: Optional logger instance.
            parent: Optional parent widget.
        """

        super().__init__(parent=parent)
        self._logger = logger or logging.getLogger(__name__)
        self._title = str(title)
        self._label_a = str(label_a)
        self._label_b = str(label_b)
        self._marker_labels_include_series = bool(marker_labels_include_series)
        self._figure = Figure(figsize=(5, 2.8))
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvas(self._figure)
        self._axes.set_title(self._title)
        self._axes.set_xlabel("Profile index")
        self._axes.set_ylabel("Intensity")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.clear("Load scans and select a pixel.")

    def _apply_margins(self) -> None:
        """Apply stable subplot margins for readability at small widget sizes."""

        self._figure.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.26)

    def clear(self, message: str = "") -> None:
        """Clear the plot and optionally display a message.

        Parameters:
            message: Optional message rendered in the axes.
        """

        self._axes.clear()
        self._axes.set_title(self._title)
        self._axes.set_xlabel("Profile index")
        self._axes.set_ylabel("Intensity")
        if message:
            self._axes.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=self._axes.transAxes,
                fontsize=10,
                color="#404040",
            )
        self._apply_margins()
        self._canvas.draw_idle()

    def update_plot(
        self,
        payload_a: Optional[BandProfilePayload],
        payload_b: Optional[BandProfilePayload],
        *,
        normalize: bool = True,
        show_markers: bool = True,
    ) -> None:
        """Update the plot for the given scan payloads.

        Parameters:
            payload_a: Payload for scan A (or None).
            payload_b: Payload for scan B (or None).
            normalize: Whether to normalize each profile by its own max.
            show_markers: Whether to draw start/end/peak marker lines.
        """

        self._axes.clear()
        self._axes.set_title(self._title)
        self._axes.set_xlabel("Profile index")
        ylabel = "Intensity (normalized)" if normalize else "Intensity"
        self._axes.set_ylabel(ylabel)

        if payload_a is None and payload_b is None:
            self.clear("No valid band profile for this selection.")
            return

        any_series = False
        any_marker = False
        for series_name, payload, color, dash in (
            (self._label_a, payload_a, "#1f77b4", "--"),
            (self._label_b, payload_b, "#ff7f0e", ":"),
        ):
            if payload is None or payload.profile is None:
                continue
            profile = np.asarray(payload.profile, dtype=np.float32).ravel()
            profile = np.where(np.isfinite(profile), profile, np.nan)
            if normalize:
                profile = normalize_profile(
                    profile,
                    logger=self._logger,
                    context=f"{series_name}",
                )
            x_vals = np.arange(profile.size, dtype=np.int32)
            self._axes.plot(
                x_vals,
                profile,
                color=color,
                linewidth=1.8,
                label=f"{series_name} profile",
            )
            any_series = True

            if not show_markers:
                continue
            marker_specs = [
                ("start", payload.band_start_idx, dash, 0.55),
                ("end", payload.band_end_idx, dash, 0.55),
                ("peak", payload.central_peak_idx, "-", 0.7),
            ]
            for marker_label, idx, linestyle, alpha in marker_specs:
                if idx is None:
                    continue
                if idx < 0 or idx >= profile.size:
                    self._logger.warning(
                        "Marker %s idx=%s out of range for %s (len=%s).",
                        marker_label,
                        idx,
                        series_name,
                        profile.size,
                    )
                    continue
                if self._marker_labels_include_series:
                    draw_label = f"{series_name} {marker_label}"
                else:
                    draw_label = f"Band {marker_label}"
                self._axes.axvline(
                    float(idx),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2 if marker_label != "peak" else 1.0,
                    alpha=alpha,
                    label=draw_label,
                )
                any_marker = True

        if not any_series:
            self.clear("No valid band profile at this pixel.")
            return

        if any_marker:
            self._axes.legend(fontsize=8, loc="upper right", framealpha=0.85)
        else:
            self._axes.legend(fontsize=8, loc="upper right", framealpha=0.85)
        self._axes.grid(True, which="both", alpha=0.15)
        self._apply_margins()
        self._canvas.draw_idle()
