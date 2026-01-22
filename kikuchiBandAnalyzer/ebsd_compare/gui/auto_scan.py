"""Auto-scan raster animation utilities for the EBSD compare GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PySide6 import QtCore


@dataclass
class RasterStepper:
    """Raster stepping helper for grid coordinates.

    Parameters:
        width: Number of columns in the grid.
        height: Number of rows in the grid.
        start_x: Starting column index.
        start_y: Starting row index.
    """

    width: int
    height: int
    start_x: int = 0
    start_y: int = 0

    def __post_init__(self) -> None:
        """Validate the raster stepper inputs.

        Returns:
            None.
        """

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Raster dimensions must be positive.")
        self.reset(self.start_x, self.start_y)

    def reset(self, start_x: int = 0, start_y: int = 0) -> None:
        """Reset the stepper to a starting coordinate.

        Parameters:
            start_x: Starting column index.
            start_y: Starting row index.

        Returns:
            None.
        """

        if start_x < 0 or start_x >= self.width:
            raise ValueError("start_x must be within grid bounds.")
        if start_y < 0 or start_y >= self.height:
            raise ValueError("start_y must be within grid bounds.")
        self._x = start_x
        self._y = start_y
        self._finished = False

    def next_coordinate(self) -> Optional[Tuple[int, int]]:
        """Return the next coordinate in raster order.

        Returns:
            Tuple of (x, y) or None when finished.
        """

        if self._finished:
            return None
        current = (self._x, self._y)
        self._x += 1
        if self._x >= self.width:
            self._x = 0
            self._y += 1
        if self._y >= self.height:
            self._finished = True
        return current


class AutoScanController(QtCore.QObject):
    """Controller for auto-scan raster animation.

    Parameters:
        on_step: Callback invoked with each new coordinate.
        on_finished: Callback invoked when scanning completes.
        logger: Logger instance for reporting.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        on_step: Callable[[int, int], None],
        on_finished: Callable[[], None],
        logger: logging.Logger,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        """Initialize the controller.

        Parameters:
            on_step: Callback for each coordinate.
            on_finished: Callback when scan completes.
            logger: Logger instance.
            parent: Optional Qt parent.
        """

        super().__init__(parent=parent)
        self._logger = logger
        self._on_step = on_step
        self._on_finished = on_finished
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timeout)
        self._stepper: Optional[RasterStepper] = None
        self._running = False
        self._paused = False

    def set_bounds(self, width: int, height: int) -> None:
        """Set the raster bounds for scanning.

        Parameters:
            width: Number of columns.
            height: Number of rows.

        Returns:
            None.
        """

        self._stepper = RasterStepper(width=width, height=height)

    def set_interval_ms(self, interval_ms: int) -> None:
        """Set the timer interval in milliseconds.

        Parameters:
            interval_ms: Delay between steps in milliseconds.

        Returns:
            None.
        """

        self._timer.setInterval(interval_ms)

    def is_running(self) -> bool:
        """Return True if auto-scan is currently running.

        Returns:
            True if running, otherwise False.
        """

        return self._running

    def is_paused(self) -> bool:
        """Return True if auto-scan is currently paused.

        Returns:
            True if paused, otherwise False.
        """

        return self._paused

    def play(self) -> None:
        """Start scanning from the beginning of the raster.

        Returns:
            None.
        """

        if self._stepper is None:
            self._logger.warning("Auto-scan cannot start without scan bounds.")
            return
        self._stepper.reset(0, 0)
        self._running = True
        self._paused = False
        self._timer.start()
        self._logger.info("Auto-scan started.")

    def pause(self) -> None:
        """Pause the auto-scan if running.

        Returns:
            None.
        """

        if not self._running or self._paused:
            return
        self._timer.stop()
        self._paused = True
        self._logger.info("Auto-scan paused.")

    def resume(self) -> None:
        """Resume the auto-scan if paused.

        Returns:
            None.
        """

        if not self._running or not self._paused:
            return
        self._paused = False
        self._timer.start()
        self._logger.info("Auto-scan resumed.")

    def stop(self) -> None:
        """Stop the auto-scan and reset running state.

        Returns:
            None.
        """

        if not self._running and not self._paused:
            return
        self._timer.stop()
        self._running = False
        self._paused = False
        self._logger.info("Auto-scan stopped.")

    def step_once(self) -> Optional[Tuple[int, int]]:
        """Advance one step without starting the timer.

        Returns:
            Tuple of (x, y) or None when finished.
        """

        if self._stepper is None:
            return None
        return self._stepper.next_coordinate()

    def _on_timeout(self) -> None:
        """Handle timer ticks for raster updates.

        Returns:
            None.
        """

        if self._stepper is None:
            self.stop()
            return
        coord = self._stepper.next_coordinate()
        if coord is None:
            self.stop()
            self._on_finished()
            return
        self._on_step(coord[0], coord[1])
