"""Logging widgets for GUI log display."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


class LogEmitter(QtCore.QObject):
    """Qt signal emitter for log messages."""

    message = QtCore.Signal(str)


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

        message = self.format(record)
        self._emitter.message.emit(message)


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
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(max_lines)
        self._text.setFont(QtGui.QFont("Monospace"))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._text)

    def append_message(self, message: str) -> None:
        """Append a log message to the viewer.

        Parameters:
            message: Log message string.

        Returns:
            None.
        """

        self._text.appendPlainText(message)

    def text_widget(self) -> QtWidgets.QPlainTextEdit:
        """Return the underlying text widget.

        Returns:
            QPlainTextEdit instance.
        """

        return self._text
