"""Abstract base reader for EBSD scan files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kikuchiBandAnalyzer.ebsd_compare.model import FieldCatalog


class ScanFileReader(ABC):
    """Abstract base class for scan file readers."""

    @abstractmethod
    def catalog(self) -> "FieldCatalog":
        """Return the discovered field catalog.

        Returns:
            FieldCatalog containing scalar and pattern fields.
        """

    @abstractmethod
    def get_map(self, field_name: str) -> np.ndarray:
        """Return a 2D scalar map for the requested field.

        Parameters:
            field_name: Name of the scalar field.

        Returns:
            2D NumPy array.
        """

    @abstractmethod
    def get_scalar(self, field_name: str, x: int, y: int) -> float:
        """Return a scalar value at the specified coordinate.

        Parameters:
            field_name: Name of the scalar field.
            x: Column index.
            y: Row index.

        Returns:
            Scalar value at the specified location.
        """

    @abstractmethod
    def get_pattern(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return a pattern image at the specified coordinate.

        Parameters:
            field_name: Name of the pattern field.
            x: Column index.
            y: Row index.

        Returns:
            2D NumPy array for the pattern, or None if unavailable.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any open resources."""
