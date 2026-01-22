"""Selection coordination helpers for the EBSD compare GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class SelectionBounds:
    """Bounds for valid selections.

    Parameters:
        width: Number of columns (x dimension).
        height: Number of rows (y dimension).
    """

    width: int
    height: int


@dataclass(frozen=True)
class SelectionState:
    """State container for a selection update.

    Parameters:
        x: Selected column index.
        y: Selected row index.
        source: Description of the update source.
    """

    x: int
    y: int
    source: str


class SelectionController:
    """Coordinate selection updates with validation and callbacks.

    Parameters:
        on_selection: Callback invoked after a valid selection.
        on_error: Callback invoked when a selection is invalid.
        logger: Logger instance for reporting.
    """

    def __init__(
        self,
        on_selection: Callable[[SelectionState], None],
        on_error: Callable[[str], None],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the controller.

        Parameters:
            on_selection: Callback for valid selections.
            on_error: Callback for invalid selections.
            logger: Optional logger instance.
        """

        self._logger = logger or logging.getLogger(__name__)
        self._on_selection = on_selection
        self._on_error = on_error
        self._bounds: Optional[SelectionBounds] = None
        self._state: Optional[SelectionState] = None

    def set_bounds(self, width: int, height: int) -> None:
        """Set the valid selection bounds.

        Parameters:
            width: Number of columns.
            height: Number of rows.

        Returns:
            None.
        """

        if width <= 0 or height <= 0:
            raise ValueError("Selection bounds must be positive.")
        self._bounds = SelectionBounds(width=width, height=height)

    def selected(self) -> Optional[SelectionState]:
        """Return the current selection state.

        Returns:
            SelectionState if available, otherwise None.
        """

        return self._state

    def set_selected_pixel(self, x: int, y: int, source: str) -> bool:
        """Validate and apply a new selection.

        Parameters:
            x: Column index.
            y: Row index.
            source: Description of the update source.

        Returns:
            True if the selection is applied, otherwise False.
        """

        if self._bounds is None:
            message = "Selection bounds are not set."
            self._logger.error(message)
            self._on_error(message)
            return False
        if x < 0 or x >= self._bounds.width or y < 0 or y >= self._bounds.height:
            message = (
                f"Invalid selection X={x}, Y={y} "
                f"(max X={self._bounds.width - 1}, Y={self._bounds.height - 1})."
            )
            self._logger.error(message)
            self._on_error(message)
            return False
        state = SelectionState(x=x, y=y, source=source)
        self._state = state
        self._on_selection(state)
        return True
