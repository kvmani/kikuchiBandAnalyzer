"""Tests for GUI logging handler formatting."""

from __future__ import annotations

import logging
import time

from kikuchiBandAnalyzer.ebsd_compare.gui.logging_widget import GuiLogHandler, LogEntry


class DummySignal:
    """Minimal signal stub for capturing emitted payloads."""

    def __init__(self) -> None:
        """Initialize the signal stub."""

        self.payloads = []

    def emit(self, payload: LogEntry) -> None:
        """Capture an emitted payload.

        Parameters:
            payload: LogEntry instance.
        """

        self.payloads.append(payload)


class DummyEmitter:
    """Minimal emitter stub matching the expected interface."""

    def __init__(self) -> None:
        """Initialize the emitter stub."""

        self.message = DummySignal()


def test_gui_log_handler_emits_entry() -> None:
    """Ensure GUI log handler emits structured entries."""

    emitter = DummyEmitter()
    handler = GuiLogHandler(emitter)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    record = logging.LogRecord("test", logging.INFO, __file__, 10, "Hello %s", ("World",), None)
    record.created = 0
    record.msecs = 0
    handler.emit(record)
    assert len(emitter.message.payloads) == 1
    entry = emitter.message.payloads[0]
    assert isinstance(entry, LogEntry)
    assert entry.timestamp == "1970-01-01 00:00:00"
    assert entry.level == "INFO"
    assert entry.message == "Hello World"
