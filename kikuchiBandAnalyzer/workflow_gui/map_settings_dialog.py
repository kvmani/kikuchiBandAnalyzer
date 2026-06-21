"""Qt editor for reusable scientific map display settings."""

from __future__ import annotations

from dataclasses import replace

from matplotlib import colormaps
from PySide6 import QtWidgets

from kikuchiBandAnalyzer.ebsd_compare.map_display import MapDisplaySettings


class MapDisplaySettingsDialog(QtWidgets.QDialog):
    """Edit scalar-map range, normalization, and colormap settings."""

    def __init__(
        self,
        field_name: str,
        settings: MapDisplaySettings,
        defaults: MapDisplaySettings | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Parameters:
            field_name: Scientific field being configured.
            settings: Current display settings.
            defaults: Optional field-specific reset defaults.
            parent: Optional parent widget.
        """

        super().__init__(parent)
        self._defaults = replace(defaults or MapDisplaySettings())
        self.setWindowTitle(f"{field_name} display properties")
        form = QtWidgets.QFormLayout(self)
        self._range_mode = QtWidgets.QComboBox()
        self._range_mode.addItem("Automatic percentiles", "auto")
        self._range_mode.addItem("Manual limits", "manual")
        self._low = self._spin(-1.0e30, 1.0e30, 6)
        self._high = self._spin(-1.0e30, 1.0e30, 6)
        self._minimum = self._spin(-1.0e30, 1.0e30, 8)
        self._maximum = self._spin(-1.0e30, 1.0e30, 8)
        self._scale = QtWidgets.QComboBox()
        self._scale.addItems(["linear", "log", "symlog"])
        self._symmetric = QtWidgets.QCheckBox("Symmetric around zero")
        self._linthresh = self._spin(1.0e-15, 1.0e30, 10)
        self._colormap = QtWidgets.QComboBox()
        preferred = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu", "gray"]
        available = [name for name in preferred if name in colormaps]
        available.extend(
            name
            for name in sorted(colormaps)
            if not name.endswith("_r") and name not in available
        )
        self._colormap.addItems(available)
        self._reversed = QtWidgets.QCheckBox("Reverse colormap")
        self._invalid_color = QtWidgets.QLineEdit()
        form.addRow("Range", self._range_mode)
        form.addRow("Low percentile", self._low)
        form.addRow("High percentile", self._high)
        form.addRow("Minimum", self._minimum)
        form.addRow("Maximum", self._maximum)
        form.addRow("Scale", self._scale)
        form.addRow("", self._symmetric)
        form.addRow("Symlog threshold", self._linthresh)
        form.addRow("Colormap", self._colormap)
        form.addRow("", self._reversed)
        form.addRow("Invalid color", self._invalid_color)
        self._error = QtWidgets.QLabel()
        self._error.setStyleSheet("color: #c62828;")
        self._error.setWordWrap(True)
        form.addRow(self._error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        form.addRow(buttons)
        self._load(settings)
        self._range_mode.currentIndexChanged.connect(self._update_enabled)
        self._scale.currentTextChanged.connect(self._update_enabled)
        self._update_enabled()

    def settings(self) -> MapDisplaySettings:
        """Return validated settings represented by the controls.

        Returns:
            Map display settings.
        """

        settings = MapDisplaySettings(
            range_mode=str(self._range_mode.currentData()),
            percentile_low=float(self._low.value()),
            percentile_high=float(self._high.value()),
            minimum=float(self._minimum.value()),
            maximum=float(self._maximum.value()),
            scale=self._scale.currentText(),
            symmetric=self._symmetric.isChecked(),
            linthresh=float(self._linthresh.value()),
            colormap=self._colormap.currentText(),
            reversed=self._reversed.isChecked(),
            invalid_color=self._invalid_color.text().strip() or "#808080",
        )
        settings.validate()
        return settings

    def _spin(self, minimum: float, maximum: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        """Create a scientific-notation-capable numeric control.

        Parameters:
            minimum: Minimum value.
            maximum: Maximum value.
            decimals: Display precision.

        Returns:
            Configured spin box.
        """

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setKeyboardTracking(False)
        return spin

    def _load(self, settings: MapDisplaySettings) -> None:
        """Load settings into controls.

        Parameters:
            settings: Source settings.
        """

        self._range_mode.setCurrentIndex(0 if settings.range_mode == "auto" else 1)
        self._low.setValue(settings.percentile_low)
        self._high.setValue(settings.percentile_high)
        self._minimum.setValue(settings.minimum)
        self._maximum.setValue(settings.maximum)
        self._scale.setCurrentText(settings.scale)
        self._symmetric.setChecked(settings.symmetric)
        self._linthresh.setValue(settings.linthresh)
        self._colormap.setCurrentText(settings.colormap)
        self._reversed.setChecked(settings.reversed)
        self._invalid_color.setText(settings.invalid_color)

    def _restore_defaults(self) -> None:
        """Restore field defaults supplied when the dialog was opened."""

        self._load(self._defaults)
        self._error.clear()
        self._update_enabled()

    def _update_enabled(self) -> None:
        """Enable controls relevant to the selected range and scale modes."""

        automatic = self._range_mode.currentData() == "auto"
        self._low.setEnabled(automatic)
        self._high.setEnabled(automatic)
        self._minimum.setEnabled(not automatic)
        self._maximum.setEnabled(not automatic)
        self._linthresh.setEnabled(self._scale.currentText() == "symlog")

    def _accept_if_valid(self) -> None:
        """Accept only when current values form valid settings."""

        try:
            self.settings()
        except ValueError as exc:
            self._error.setText(str(exc))
            return
        self.accept()
