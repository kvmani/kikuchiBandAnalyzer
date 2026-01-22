"""Logging widgets for GUI log display."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class LogEntry:
    """Structured log entry for the GUI console.

    Parameters:
        timestamp: Formatted timestamp string.
        level: Log level name.
        message: Log message text.
    """

    timestamp: str
    level: str
    message: str


class LogEmitter(QtCore.QObject):
    """Qt signal emitter for log messages."""

    message = QtCore.Signal(object)


class GuiLogHandler(logging.Handler):
    """Logging handler that forwards formatted messages to a Qt signal."""

    def __init__(self, emitter: LogEmitter) -> None:
        """Initialize the handler.

        Parameters:
            emitter: LogEmitter instance.

        Returns:
            None.
        """

        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the Qt signal.

        Parameters:
            record: Log record to emit.

        Returns:
            None.
        """

        formatter = self.formatter or logging.Formatter()
        timestamp = formatter.formatTime(record, formatter.datefmt)
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{formatter.formatException(record.exc_info)}"
        entry = LogEntry(timestamp=timestamp, level=record.levelname, message=message)
        self._emitter.message.emit(entry)


class LogFilterProxyModel(QtCore.QSortFilterProxyModel):
    """Filter proxy for log level filtering.

    Parameters:
        parent: Optional QObject parent.
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        """Initialize the proxy model.

        Parameters:
            parent: Optional parent object.
        """

        super().__init__(parent=parent)
        self._level_filter: Optional[str] = None

    def set_level_filter(self, level: Optional[str]) -> None:
        """Set the level filter string.

        Parameters:
            level: Level name or None to disable filtering.

        Returns:
            None.
        """

        self._level_filter = level
        self.invalidateFilter()

    def filterAcceptsRow(
        self, source_row: int, source_parent: QtCore.QModelIndex
    ) -> bool:
        """Return True if the row passes the level filter.

        Parameters:
            source_row: Row index in the source model.
            source_parent: Parent index in the source model.

        Returns:
            True if row should be visible, otherwise False.
        """

        if not self._level_filter or self._level_filter == "All":
            return True
        level_index = self.sourceModel().index(source_row, 1, source_parent)
        level_text = self.sourceModel().data(level_index, QtCore.Qt.DisplayRole)
        if level_text is None:
            return False
        return str(level_text).upper() == self._level_filter.upper()


class LogViewer(QtWidgets.QWidget):
    """Widget for displaying log output in the GUI."""

    def __init__(self, max_lines: int = 1000, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize the log viewer.

        Parameters:
            max_lines: Maximum number of log lines to retain.
            parent: Optional parent widget.

        Returns:
            None.
        """

        super().__init__(parent=parent)
        self._max_lines = max_lines
        self._model = QtGui.QStandardItemModel(0, 3, self)
        self._model.setHorizontalHeaderLabels(["Time", "Level", "Message"])
        self._proxy = LogFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._table = QtWidgets.QTableView()
        self._table.setModel(self._proxy)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)

        self._auto_scroll_checkbox = QtWidgets.QCheckBox("Auto-scroll")
        self._auto_scroll_checkbox.setChecked(True)
        self._auto_scroll_checkbox.setToolTip(
            "Keep the newest log entry in view when enabled."
        )
        self._filter_combo = QtWidgets.QComboBox()
        self._filter_combo.addItems(["All", "Debug", "Info", "Warning", "Error", "Critical"])
        self._filter_combo.currentTextChanged.connect(self._on_filter_change)
        self._filter_combo.setToolTip(
            "Filter log entries by level. Set to All to show everything."
        )
        self._copy_selected_button = QtWidgets.QPushButton("Copy Selected")
        self._copy_selected_button.clicked.connect(self._copy_selected)
        self._copy_selected_button.setToolTip(
            "Copy the selected log rows to the clipboard."
        )
        self._copy_all_button = QtWidgets.QPushButton("Copy All")
        self._copy_all_button.clicked.connect(self._copy_all)
        self._copy_all_button.setToolTip("Copy all visible log rows to the clipboard.")
        self._clear_button = QtWidgets.QPushButton("Clear Logs")
        self._clear_button.clicked.connect(self.clear)
        self._clear_button.setToolTip("Clear the log console output.")

        self._icon_map = {
            "DEBUG": self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload),
            "INFO": self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation),
            "WARNING": self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning),
            "ERROR": self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical),
            "CRITICAL": self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical),
        }
        self._level_colors = {
            "DEBUG": "#6c757d",
            "INFO": "#1f77b4",
            "WARNING": "#b58900",
            "ERROR": "#c62828",
            "CRITICAL": "#8e0000",
        }

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(self._auto_scroll_checkbox)
        controls_layout.addWidget(QtWidgets.QLabel("Level"))
        controls_layout.addWidget(self._filter_combo)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._copy_selected_button)
        controls_layout.addWidget(self._copy_all_button)
        controls_layout.addWidget(self._clear_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls_layout)
        layout.addWidget(self._table)
        self._table.setToolTip(
            "Application log output. Select rows to copy for bug reports."
        )

    def append_entry(self, entry: LogEntry) -> None:
        """Append a log entry to the viewer.

        Parameters:
            entry: LogEntry to append.

        Returns:
            None.
        """

        level_key = entry.level.upper()
        time_item = QtGui.QStandardItem(entry.timestamp)
        level_item = QtGui.QStandardItem(entry.level)
        icon = self._icon_map.get(level_key)
        if icon:
            level_item.setIcon(icon)
        message_item = QtGui.QStandardItem(entry.message)
        color = self._level_colors.get(level_key)
        if color:
            brush = QtGui.QBrush(QtGui.QColor(color))
            level_item.setForeground(brush)
            message_item.setForeground(brush)
        for item in (time_item, level_item, message_item):
            item.setEditable(False)
        self._model.appendRow([time_item, level_item, message_item])
        self._trim_rows()
        if self._auto_scroll_checkbox.isChecked():
            self._table.scrollToBottom()

    def clear(self) -> None:
        """Clear all log entries from the viewer.

        Returns:
            None.
        """

        self._model.removeRows(0, self._model.rowCount())

    def _trim_rows(self) -> None:
        """Trim the model to the maximum number of lines.

        Returns:
            None.
        """

        overflow = self._model.rowCount() - self._max_lines
        if overflow > 0:
            self._model.removeRows(0, overflow)

    def _on_filter_change(self, text: str) -> None:
        """Update the proxy filter from the dropdown selection.

        Parameters:
            text: Current filter text.

        Returns:
            None.
        """

        self._proxy.set_level_filter(text)

    def _copy_selected(self) -> None:
        """Copy the selected log rows to the clipboard.

        Returns:
            None.
        """

        selection = self._table.selectionModel().selectedRows()
        rows = sorted(index.row() for index in selection)
        lines = []
        for row in rows:
            timestamp = self._proxy.index(row, 0).data()
            level = self._proxy.index(row, 1).data()
            message = self._proxy.index(row, 2).data()
            lines.append(f"{timestamp} | {level} | {message}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def _copy_all(self) -> None:
        """Copy all visible log rows to the clipboard.

        Returns:
            None.
        """

        lines = []
        for row in range(self._proxy.rowCount()):
            timestamp = self._proxy.index(row, 0).data()
            level = self._proxy.index(row, 1).data()
            message = self._proxy.index(row, 2).data()
            lines.append(f"{timestamp} | {level} | {message}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
