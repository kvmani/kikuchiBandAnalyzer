"""Reader implementation for HKL/Oxford CTF files plus pattern folders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from kikuchiBandAnalyzer.ebsd_compare.model import FieldCatalog, FieldRef, ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.readers.base import ScanFileReader

_IMAGE_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


class CtfPatternScanFileReader(ScanFileReader):
    """Scan reader for HKL/Oxford CTF metadata and external pattern images.

    Parameters:
        ctf_path: Path to the CTF text file.
        pattern_dir: Optional directory containing one pattern image per pixel.
        pattern_template: Optional filename template using ``x``, ``y``,
            ``row``, ``col``, and ``index`` format fields.
        field_aliases: Optional mapping of canonical field names to aliases.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        ctf_path: Path,
        pattern_dir: Optional[Path] = None,
        pattern_template: Optional[str] = None,
        field_aliases: Optional[Dict[str, list[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the reader and parse the CTF file.

        Parameters:
            ctf_path: Path to the CTF file.
            pattern_dir: Directory containing pattern images.
            pattern_template: Optional filename template.
            field_aliases: Optional field alias mapping.
            logger: Optional logger instance.
        """

        self._file_path = Path(ctf_path)
        self._pattern_dir = Path(pattern_dir) if pattern_dir else None
        self._pattern_template = pattern_template
        self._logger = logger or logging.getLogger(__name__)
        self._alias_map = self._build_alias_map(field_aliases or {})
        self._header, self._columns, self._table = self._parse_ctf(self._file_path)
        self._nx, self._ny = self._read_grid_shape()
        self._scan_name = self._file_path.stem
        self._pattern_files = self._discover_pattern_files()
        self._catalog = self._discover_fields()

    @classmethod
    def from_path(
        cls,
        ctf_path: Path,
        pattern_dir: Optional[Path] = None,
        pattern_template: Optional[str] = None,
        field_aliases: Optional[Dict[str, list[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> ScanDataset:
        """Create a ScanDataset from a CTF file and optional pattern folder.

        Parameters:
            ctf_path: Path to the CTF file.
            pattern_dir: Directory containing pattern images.
            pattern_template: Optional filename template.
            field_aliases: Optional field alias mapping.
            logger: Optional logger instance.

        Returns:
            ScanDataset with CTF scalar fields and optional pattern access.
        """

        reader = cls(
            ctf_path=ctf_path,
            pattern_dir=pattern_dir,
            pattern_template=pattern_template,
            field_aliases=field_aliases,
            logger=logger,
        )
        return ScanDataset(
            file_path=Path(ctf_path),
            scan_name=reader._scan_name,
            nx=reader._nx,
            ny=reader._ny,
            catalog=reader._catalog,
            reader=reader,
        )

    def catalog(self) -> FieldCatalog:
        """Return the discovered field catalog.

        Returns:
            FieldCatalog describing CTF scalar fields and patterns.
        """

        return self._catalog

    @property
    def nx(self) -> int:
        """Return the number of scan columns.

        Returns:
            Number of CTF grid columns.
        """

        return self._nx

    @property
    def ny(self) -> int:
        """Return the number of scan rows.

        Returns:
            Number of CTF grid rows.
        """

        return self._ny

    def get_map(self, field_name: str) -> np.ndarray:
        """Return a 2D scalar map for a CTF column.

        Parameters:
            field_name: Scalar column name or alias.

        Returns:
            2D NumPy array shaped (ny, nx).
        """

        field_ref = self._resolve_scalar_field(field_name)
        values = self._table[field_ref.name]
        if values.size != self._nx * self._ny:
            raise ValueError(
                f"Scalar field '{field_ref.name}' has {values.size} values; expected {self._nx * self._ny}."
            )
        return np.reshape(values, (self._ny, self._nx))

    def get_scalar(self, field_name: str, x: int, y: int) -> float:
        """Return a scalar value at the specified coordinate.

        Parameters:
            field_name: Scalar column name or alias.
            x: Column index.
            y: Row index.

        Returns:
            Scalar value at the specified pixel.
        """

        self._validate_xy(x, y)
        return float(self.get_map(field_name)[y, x])

    def get_pattern(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return a pattern image from the external pattern folder.

        Parameters:
            field_name: Pattern field name; currently ``Pattern``.
            x: Column index.
            y: Row index.

        Returns:
            2D pattern image, or None if no pattern folder is configured.
        """

        if self._resolve_pattern_field(field_name) is None:
            return None
        self._validate_xy(x, y)
        pattern_path = self._pattern_path_for_xy(x, y)
        if pattern_path is None:
            return None
        with Image.open(pattern_path) as image:
            return np.asarray(image.convert("L"), dtype=np.float32)

    def get_vector(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return a vector value at the specified coordinate.

        Parameters:
            field_name: Vector field name.
            x: Column index.
            y: Row index.

        Returns:
            None because CTF columns are currently exposed as scalar fields.
        """

        return None

    def close(self) -> None:
        """Release reader resources.

        CTF readers keep no open file handles.
        """

        return None

    def _parse_ctf(self, ctf_path: Path) -> tuple[dict[str, str], list[str], dict[str, np.ndarray]]:
        """Parse CTF header and numeric data table.

        Parameters:
            ctf_path: Path to the CTF file.

        Returns:
            Tuple of header mapping, column names, and column arrays.
        """

        header: dict[str, str] = {}
        columns: list[str] | None = None
        rows: list[list[float]] = []
        for raw_line in ctf_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if columns is None and parts and parts[0].lower() == "phase":
                columns = parts
                continue
            if columns is None:
                if len(parts) >= 2:
                    header[parts[0]] = " ".join(parts[1:])
                continue
            if line.lower().startswith("phase "):
                continue
            if len(parts) < len(columns):
                self._logger.warning("Skipping short CTF row: %s", line)
                continue
            try:
                rows.append([float(value) for value in parts[: len(columns)]])
            except ValueError:
                self._logger.warning("Skipping non-numeric CTF row: %s", line)
        if columns is None:
            raise ValueError(f"CTF file has no data header line beginning with 'Phase': {ctf_path}")
        if not rows:
            raise ValueError(f"CTF file contains no numeric rows: {ctf_path}")
        array = np.asarray(rows, dtype=np.float64)
        table = {name: array[:, index] for index, name in enumerate(columns)}
        return header, columns, table

    def _read_grid_shape(self) -> tuple[int, int]:
        """Read grid shape from CTF header or X/Y coordinates.

        Returns:
            Tuple of (nx, ny).
        """

        x_cells = self._read_header_int("XCells")
        y_cells = self._read_header_int("YCells")
        if x_cells and y_cells:
            return x_cells, y_cells
        if "X" in self._table and "Y" in self._table:
            nx = int(np.unique(self._table["X"]).size)
            ny = int(np.unique(self._table["Y"]).size)
            if nx * ny == len(next(iter(self._table.values()))):
                return nx, ny
        count = len(next(iter(self._table.values())))
        return count, 1

    def _read_header_int(self, key: str) -> Optional[int]:
        """Read an integer CTF header value.

        Parameters:
            key: Header key.

        Returns:
            Integer value, or None if unavailable.
        """

        value = self._header.get(key)
        if value is None:
            return None
        try:
            return int(float(value.split()[0]))
        except ValueError:
            return None

    def _discover_fields(self) -> FieldCatalog:
        """Discover CTF scalar fields and optional pattern field.

        Returns:
            FieldCatalog for CTF data.
        """

        scalars = {
            name: FieldRef(
                name=name,
                path=f"ctf:{name}",
                kind="scalar",
                shape=(self._ny, self._nx),
                dtype=str(values.dtype),
            )
            for name, values in self._table.items()
        }
        patterns = {}
        if self._pattern_dir is not None:
            patterns["Pattern"] = FieldRef(
                name="Pattern",
                path=str(self._pattern_dir),
                kind="pattern",
                shape=(self._ny, self._nx),
                dtype="image",
            )
        return FieldCatalog(scalars=scalars, vectors={}, patterns=patterns)

    def _discover_pattern_files(self) -> list[Path]:
        """Discover pattern images in row-major order when no template is used.

        Returns:
            Sorted list of pattern image paths.
        """

        if self._pattern_dir is None or self._pattern_template:
            return []
        files = [
            path
            for path in self._pattern_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        files = sorted(files, key=lambda path: path.name.lower())
        expected = self._nx * self._ny
        if files and len(files) != expected:
            self._logger.warning(
                "Pattern folder %s has %d images; CTF grid expects %d.",
                self._pattern_dir,
                len(files),
                expected,
            )
        return files

    def _pattern_path_for_xy(self, x: int, y: int) -> Optional[Path]:
        """Resolve a pattern image path for a pixel coordinate.

        Parameters:
            x: Column index.
            y: Row index.

        Returns:
            Pattern image path, or None if unavailable.
        """

        if self._pattern_dir is None:
            return None
        index = y * self._nx + x
        if self._pattern_template:
            candidate = self._pattern_dir / self._pattern_template.format(
                x=x,
                y=y,
                col=x,
                row=y,
                index=index,
            )
            return candidate if candidate.exists() else None
        if index >= len(self._pattern_files):
            return None
        return self._pattern_files[index]

    def _validate_xy(self, x: int, y: int) -> None:
        """Validate scan coordinates.

        Parameters:
            x: Column index.
            y: Row index.

        Returns:
            None.
        """

        if x < 0 or y < 0 or x >= self._nx or y >= self._ny:
            raise IndexError(f"Pixel coordinate ({x}, {y}) outside scan bounds.")

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
            aliases: Mapping of canonical field names to alias names.

        Returns:
            Mapping of normalized alias to canonical field.
        """

        alias_map: Dict[str, str] = {}
        for canonical, alias_list in aliases.items():
            alias_map[self._normalize_field_name(canonical)] = canonical
            for alias in alias_list:
                alias_map[self._normalize_field_name(alias)] = canonical
        return alias_map

    def _resolve_scalar_field(self, field_name: str) -> FieldRef:
        """Resolve a CTF scalar field name.

        Parameters:
            field_name: Requested scalar field name.

        Returns:
            FieldRef for the scalar field.
        """

        if field_name in self._catalog.scalars:
            return self._catalog.scalars[field_name]
        canonical = self._alias_map.get(self._normalize_field_name(field_name))
        if canonical and canonical in self._catalog.scalars:
            return self._catalog.scalars[canonical]
        raise KeyError(f"Scalar field '{field_name}' not found.")

    def _resolve_pattern_field(self, field_name: str) -> Optional[FieldRef]:
        """Resolve a CTF pattern field name.

        Parameters:
            field_name: Requested pattern field name.

        Returns:
            FieldRef for the pattern field, or None if unavailable.
        """

        if field_name in self._catalog.patterns:
            return self._catalog.patterns[field_name]
        canonical = self._alias_map.get(self._normalize_field_name(field_name))
        if canonical and canonical in self._catalog.patterns:
            return self._catalog.patterns[canonical]
        return None
