"""Data models for EBSD scan comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kikuchiBandAnalyzer.ebsd_compare.readers.base import ScanFileReader


@dataclass(frozen=True)
class FieldRef:
    """Reference metadata for a discovered field within a scan.

    Parameters:
        name: Human-readable field name.
        path: HDF5 dataset path for the field.
        kind: The field kind ("scalar" or "pattern").
        shape: Shape of the underlying dataset.
        dtype: Data type of the underlying dataset.
    """

    name: str
    path: str
    kind: str
    shape: tuple
    dtype: str


@dataclass
class FieldCatalog:
    """Collection of scalar and pattern fields discovered in a scan.

    Parameters:
        scalars: Mapping of scalar field names to references.
        patterns: Mapping of pattern field names to references.
    """

    scalars: Dict[str, FieldRef]
    patterns: Dict[str, FieldRef]

    def list_scalar_fields(self) -> list[str]:
        """Return the scalar field names.

        Returns:
            List of scalar field names.
        """

        return sorted(self.scalars.keys())

    def list_pattern_fields(self) -> list[str]:
        """Return the pattern field names.

        Returns:
            List of pattern field names.
        """

        return sorted(self.patterns.keys())


@dataclass
class ScanDataset:
    """Represents a loaded scan and provides data access helpers.

    Parameters:
        file_path: Path to the source file.
        scan_name: Scan group name.
        nx: Number of columns.
        ny: Number of rows.
        catalog: Discovered field catalog.
        reader: Reader instance used for data access.
    """

    file_path: Path
    scan_name: str
    nx: int
    ny: int
    catalog: FieldCatalog
    reader: ScanFileReader

    def get_map(self, field_name: str) -> np.ndarray:
        """Return a 2D scalar map for the requested field.

        Parameters:
            field_name: Name of the scalar field.

        Returns:
            2D NumPy array shaped (ny, nx).
        """

        return self.reader.get_map(field_name)

    def get_scalar(self, field_name: str, x: int, y: int) -> float:
        """Return a scalar value at the specified coordinate.

        Parameters:
            field_name: Name of the scalar field.
            x: Column index.
            y: Row index.

        Returns:
            Scalar value at the specified location.
        """

        return self.reader.get_scalar(field_name, x, y)

    def get_pattern(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return a pattern image at the specified coordinate.

        Parameters:
            field_name: Name of the pattern field.
            x: Column index.
            y: Row index.

        Returns:
            2D NumPy array for the pattern, or None if unavailable.
        """

        return self.reader.get_pattern(field_name, x, y)

    def close(self) -> None:
        """Close the underlying reader resources."""

        self.reader.close()
