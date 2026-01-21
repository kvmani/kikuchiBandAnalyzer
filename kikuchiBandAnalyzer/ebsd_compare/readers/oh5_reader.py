"""Reader implementation for OH5 (HDF5) EBSD scan files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.model import FieldCatalog, FieldRef, ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.readers.base import ScanFileReader


class OH5ScanFileReader(ScanFileReader):
    """Scan reader for OH5 files.

    Parameters:
        file_path: Path to the OH5 file.
        field_aliases: Optional mapping of canonical field names to aliases.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        file_path: Path,
        field_aliases: Optional[Dict[str, list[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the reader and discover fields.

        Parameters:
            file_path: Path to the OH5 file.
            field_aliases: Optional mapping of canonical field names to aliases.
            logger: Optional logger instance.
        """

        self._file_path = Path(file_path)
        self._logger = logger or logging.getLogger(__name__)
        self._file = h5py.File(self._file_path, "r")
        self._scan_name = self._discover_scan_group()
        self._data_group = self._file[f"{self._scan_name}/EBSD/Data"]
        self._header_group = self._file[f"{self._scan_name}/EBSD/Header"]
        self._nx, self._ny = self._read_grid_shape()
        self._pattern_height, self._pattern_width = self._read_pattern_shape()
        self._alias_map = self._build_alias_map(field_aliases or {})
        self._catalog = self._discover_fields()

    @classmethod
    def from_path(
        cls,
        file_path: Path,
        field_aliases: Optional[Dict[str, list[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> ScanDataset:
        """Create a ScanDataset from the provided file path.

        Parameters:
            file_path: Path to the OH5 file.
            field_aliases: Optional mapping of canonical field names to aliases.
            logger: Optional logger instance.

        Returns:
            ScanDataset with metadata and reader access.
        """

        reader = cls(file_path=file_path, field_aliases=field_aliases, logger=logger)
        return ScanDataset(
            file_path=Path(file_path),
            scan_name=reader._scan_name,
            nx=reader._nx,
            ny=reader._ny,
            catalog=reader._catalog,
            reader=reader,
        )

    def catalog(self) -> FieldCatalog:
        """Return the discovered field catalog.

        Returns:
            FieldCatalog describing scalar and pattern fields.
        """

        return self._catalog

    def get_map(self, field_name: str) -> np.ndarray:
        """Return a 2D scalar map for the requested field.

        Parameters:
            field_name: Name of the scalar field.

        Returns:
            2D NumPy array shaped (ny, nx).
        """

        field_ref = self._resolve_scalar_field(field_name)
        dataset = self._file[field_ref.path]
        data = dataset[()]
        if data.ndim == 1:
            return np.reshape(data, (self._ny, self._nx))
        if data.ndim == 2 and data.shape == (self._ny, self._nx):
            return data
        raise ValueError(
            f"Scalar field '{field_ref.name}' has unsupported shape {data.shape}."
        )

    def get_scalar(self, field_name: str, x: int, y: int) -> float:
        """Return a scalar value at the specified coordinate.

        Parameters:
            field_name: Name of the scalar field.
            x: Column index.
            y: Row index.

        Returns:
            Scalar value at the specified location.
        """

        field_ref = self._resolve_scalar_field(field_name)
        dataset = self._file[field_ref.path]
        if dataset.ndim == 1:
            index = y * self._nx + x
            return float(dataset[index])
        if dataset.ndim == 2:
            return float(dataset[y, x])
        raise ValueError(
            f"Scalar field '{field_ref.name}' has unsupported shape {dataset.shape}."
        )

    def get_pattern(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return a pattern image at the specified coordinate.

        Parameters:
            field_name: Name of the pattern field.
            x: Column index.
            y: Row index.

        Returns:
            2D NumPy array for the pattern, or None if unavailable.
        """

        field_ref = self._resolve_pattern_field(field_name)
        if field_ref is None:
            return None
        dataset = self._file[field_ref.path]
        if dataset.ndim == 3 and dataset.shape[0] == self._nx * self._ny:
            index = y * self._nx + x
            pattern = np.asarray(dataset[index], dtype=np.float32)
            return self._reshape_pattern(pattern)
        if dataset.ndim >= 3 and dataset.shape[:2] == (self._ny, self._nx):
            pattern = np.asarray(dataset[y, x], dtype=np.float32)
            return self._reshape_pattern(pattern)
        raise ValueError(
            f"Pattern field '{field_ref.name}' has unsupported shape {dataset.shape}."
        )

    def close(self) -> None:
        """Close the underlying HDF5 file."""

        if self._file is not None:
            self._file.close()

    def _discover_scan_group(self) -> str:
        """Find the first scan group in the file.

        Returns:
            Name of the scan group.
        """

        excluded = {"Manufacturer", "Version"}
        for key in self._file.keys():
            if key not in excluded and isinstance(self._file[key], h5py.Group):
                return key
        raise ValueError("No scan group found in OH5 file.")

    def _read_grid_shape(self) -> tuple[int, int]:
        """Read the grid shape from the EBSD header.

        Returns:
            Tuple of (nx, ny).
        """

        n_columns = self._read_scalar(self._header_group["nColumns"])
        n_rows = self._read_scalar(self._header_group["nRows"])
        return int(n_columns), int(n_rows)

    def _read_pattern_shape(self) -> tuple[Optional[int], Optional[int]]:
        """Read the pattern height and width from the header if available.

        Returns:
            Tuple of (height, width), or (None, None) if unavailable.
        """

        height_dataset = self._header_group.get("Pattern Height")
        width_dataset = self._header_group.get("Pattern Width")
        if height_dataset is None or width_dataset is None:
            return None, None
        return int(self._read_scalar(height_dataset)), int(self._read_scalar(width_dataset))

    def _read_scalar(self, dataset: h5py.Dataset) -> float:
        """Read a scalar value from a dataset.

        Parameters:
            dataset: HDF5 dataset containing the scalar.

        Returns:
            Scalar value.
        """

        value = dataset[()]
        if np.ndim(value) == 0:
            return float(value)
        return float(np.ravel(value)[0])

    def _discover_fields(self) -> FieldCatalog:
        """Discover scalar and pattern fields in the scan.

        Returns:
            FieldCatalog with scalar and pattern fields.
        """

        scalars: Dict[str, FieldRef] = {}
        patterns: Dict[str, FieldRef] = {}
        for name, dataset in self._data_group.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            field_ref = FieldRef(
                name=name,
                path=dataset.name,
                kind="scalar",
                shape=dataset.shape,
                dtype=str(dataset.dtype),
            )
            if self._is_scalar_dataset(dataset):
                scalars[name] = field_ref
                continue
            if self._is_pattern_dataset(dataset):
                field_ref = FieldRef(
                    name=name,
                    path=dataset.name,
                    kind="pattern",
                    shape=dataset.shape,
                    dtype=str(dataset.dtype),
                )
                patterns[name] = field_ref
        self._logger.debug(
            "Discovered %d scalar fields and %d pattern fields.",
            len(scalars),
            len(patterns),
        )
        return FieldCatalog(scalars=scalars, patterns=patterns)

    def _is_scalar_dataset(self, dataset: h5py.Dataset) -> bool:
        """Determine if a dataset is a scalar map.

        Parameters:
            dataset: HDF5 dataset.

        Returns:
            True if dataset represents a scalar map.
        """

        if dataset.ndim == 1 and dataset.shape[0] == self._nx * self._ny:
            return True
        if dataset.ndim == 2 and dataset.shape == (self._ny, self._nx):
            return True
        return False

    def _is_pattern_dataset(self, dataset: h5py.Dataset) -> bool:
        """Determine if a dataset is a pattern stack.

        Parameters:
            dataset: HDF5 dataset.

        Returns:
            True if dataset represents patterns.
        """

        if dataset.ndim >= 3 and dataset.shape[0] == self._nx * self._ny:
            return True
        if dataset.ndim >= 3 and dataset.shape[:2] == (self._ny, self._nx):
            return True
        return False

    def _reshape_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Reshape a pattern to 2D if needed.

        Parameters:
            pattern: Pattern array extracted from the dataset.

        Returns:
            2D pattern array when possible, otherwise the original array.
        """

        if pattern.ndim == 1 and self._pattern_height and self._pattern_width:
            expected = self._pattern_height * self._pattern_width
            if pattern.size == expected:
                return np.reshape(pattern, (self._pattern_height, self._pattern_width))
        return pattern

    def _normalize_field_name(self, name: str) -> str:
        """Normalize a field name for alias resolution.

        Parameters:
            name: Field name.

        Returns:
            Normalized field name.
        """

        return name.strip().lower()

    def _build_alias_map(self, aliases: Dict[str, list[str]]) -> Dict[str, str]:
        """Build an alias map for field name resolution.

        Parameters:
            aliases: Mapping of canonical names to alias lists.

        Returns:
            Mapping of normalized alias to canonical field name.
        """

        alias_map: Dict[str, str] = {}
        for canonical, alias_list in aliases.items():
            alias_map[self._normalize_field_name(canonical)] = canonical
            for alias in alias_list:
                alias_map[self._normalize_field_name(alias)] = canonical
        return alias_map

    def _resolve_scalar_field(self, field_name: str) -> FieldRef:
        """Resolve a scalar field name using aliases.

        Parameters:
            field_name: Requested field name.

        Returns:
            FieldRef for the scalar field.
        """

        if field_name in self._catalog.scalars:
            return self._catalog.scalars[field_name]
        normalized = self._normalize_field_name(field_name)
        canonical = self._alias_map.get(normalized)
        if canonical and canonical in self._catalog.scalars:
            return self._catalog.scalars[canonical]
        raise KeyError(f"Scalar field '{field_name}' not found.")

    def _resolve_pattern_field(self, field_name: str) -> Optional[FieldRef]:
        """Resolve a pattern field name using aliases.

        Parameters:
            field_name: Requested field name.

        Returns:
            FieldRef for the pattern field, or None if unavailable.
        """

        if field_name in self._catalog.patterns:
            return self._catalog.patterns[field_name]
        normalized = self._normalize_field_name(field_name)
        canonical = self._alias_map.get(normalized)
        if canonical and canonical in self._catalog.patterns:
            return self._catalog.patterns[canonical]
        return None
