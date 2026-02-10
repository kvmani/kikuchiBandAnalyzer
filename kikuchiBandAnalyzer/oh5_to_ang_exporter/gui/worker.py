"""Worker thread for OH5-to-ANG export operations."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from PySide6 import QtCore

from kikuchiBandAnalyzer.oh5_to_ang_exporter.workflow import (
    ColumnMapping,
    export_oh5_to_ang,
)


@dataclass(frozen=True)
class ExportRequest:
    """Input payload for one OH5-to-ANG export run.

    Parameters:
        oh5_path: Modified OH5 path.
        ang_path: ANG template path.
        output_path: Destination ANG path.
        include_mapping_note: Whether to append one mapping note header line.
        user_mappings: User-provided source->target mappings.
    """

    oh5_path: Path
    ang_path: Path
    output_path: Path
    include_mapping_note: bool
    user_mappings: tuple[ColumnMapping, ...]


class Oh5ToAngExportWorker(QtCore.QThread):
    """QThread wrapper for running OH5-to-ANG export without blocking the UI."""

    finished_success = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Initialize the worker.

        Parameters:
            parent: Optional QObject parent.

        Returns:
            None.
        """

        super().__init__(parent)
        self._logger = logging.getLogger(__name__)
        self._request: ExportRequest | None = None

    def configure(self, request: ExportRequest) -> None:
        """Set the request payload for the next run.

        Parameters:
            request: ExportRequest payload.

        Returns:
            None.
        """

        self._request = request

    def run(self) -> None:
        """Execute the configured export request.

        Returns:
            None.
        """

        if self._request is None:
            self.failed.emit("Export request is not configured.")
            return

        request = self._request
        try:
            result = export_oh5_to_ang(
                oh5_path=request.oh5_path,
                ang_path=request.ang_path,
                user_mappings=request.user_mappings,
                output_ang_path=request.output_path,
                include_mapping_note=request.include_mapping_note,
                logger=self._logger,
            )
        except Exception as exc:
            self._logger.exception("OH5-to-ANG export failed: %s", exc)
            self.failed.emit(str(exc))
            return

        self.finished_success.emit(result)
