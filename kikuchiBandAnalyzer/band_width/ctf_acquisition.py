"""CTF plus pattern-folder acquisition adapter for band-width workflows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.readers.ctf_reader import CtfPatternScanFileReader


@dataclass(frozen=True)
class PatternArrayDataset:
    """Small dataset wrapper exposing a ``data`` attribute like kikuchipy signals.

    Parameters:
        data: Pattern stack shaped ``(ny, nx, pattern_height, pattern_width)``.
    """

    data: np.ndarray


@dataclass(frozen=True)
class CtfAcquisitionResult:
    """Result of preparing a CTF acquisition for the band-width pipeline.

    Parameters:
        dataset: Pattern-array wrapper used by the batch detector.
        modified_h5_path: HDF5 output path created from the CTF source.
        scan_name: Scan group name written into the HDF5 output.
        euler_angles_deg: Euler angle table shaped ``(ny * nx, 3)``.
        pattern_shape: Pattern image shape as ``(height, width)``.
    """

    dataset: PatternArrayDataset
    modified_h5_path: Path
    scan_name: str
    euler_angles_deg: np.ndarray
    pattern_shape: tuple[int, int]


class CtfBandWidthAcquisition:
    """Prepare HKL/Oxford CTF metadata and external patterns for band detection."""

    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the acquisition adapter.

        Parameters:
            config: Band-width configuration mapping.
            output_dir: Directory for generated HDF5 outputs.
            logger: Optional logger instance.
        """

        self._config = config
        self._output_dir = Path(output_dir)
        self._logger = logger or logging.getLogger(__name__)

    def prepare(self) -> CtfAcquisitionResult:
        """Load CTF patterns and write the minimal HDF5 output container.

        Returns:
            CtfAcquisitionResult with pattern data and output path.
        """

        ctf_path = Path(str(self._config["ctf_file_path"]))
        pattern_dir = Path(str(self._config["pattern_folder"]))
        pattern_template = self._config.get("pattern_template")
        reader = CtfPatternScanFileReader(
            ctf_path=ctf_path,
            pattern_dir=pattern_dir,
            pattern_template=str(pattern_template) if pattern_template else None,
            field_aliases={
                "IQ": ["BC", "Band Contrast"],
                "CI": ["MAD", "Mean Angular Deviation"],
                "Fit": ["MAD", "Error"],
            },
            logger=self._logger,
        )
        try:
            patterns = self._load_pattern_stack(reader)
            modified_h5_path = self._output_dir / f"{ctf_path.stem}_modified.h5"
            prepared_h5 = self._config.get("prepared_h5_path")
            if prepared_h5 and Path(str(prepared_h5)).exists():
                shutil.copy2(Path(str(prepared_h5)), modified_h5_path)
                self._logger.info(
                    "Copied DA-compatible prepared HDF5 metadata to %s.",
                    modified_h5_path,
                )
            else:
                self._write_minimal_h5(reader, patterns, modified_h5_path)
            euler_angles = self._read_euler_angles(reader)
            return CtfAcquisitionResult(
                dataset=PatternArrayDataset(patterns),
                modified_h5_path=modified_h5_path,
                scan_name=ctf_path.stem,
                euler_angles_deg=euler_angles,
                pattern_shape=tuple(int(value) for value in patterns.shape[-2:]),
            )
        finally:
            reader.close()

    def _load_pattern_stack(self, reader: CtfPatternScanFileReader) -> np.ndarray:
        """Load all external patterns into a 4D row-major stack.

        Parameters:
            reader: CTF reader configured with a pattern folder.

        Returns:
            Pattern stack shaped ``(ny, nx, height, width)``.
        """

        rows: list[list[np.ndarray]] = []
        expected_shape: Optional[tuple[int, int]] = None
        for y in range(reader.ny):
            row: list[np.ndarray] = []
            for x in range(reader.nx):
                pattern = reader.get_pattern("Pattern", x, y)
                if pattern is None:
                    raise FileNotFoundError(
                        f"Missing pattern image for CTF pixel x={x}, y={y}."
                    )
                pattern = np.asarray(pattern, dtype=np.float32)
                if pattern.ndim != 2:
                    raise ValueError(
                        f"Pattern at x={x}, y={y} must be 2D; got shape {pattern.shape}."
                    )
                if expected_shape is None:
                    expected_shape = pattern.shape
                elif pattern.shape != expected_shape:
                    raise ValueError(
                        f"Pattern at x={x}, y={y} has shape {pattern.shape}; expected {expected_shape}."
                    )
                row.append(pattern)
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def _read_euler_angles(self, reader: CtfPatternScanFileReader) -> np.ndarray:
        """Read and validate Euler angles from a CTF table.

        Parameters:
            reader: Parsed CTF reader.

        Returns:
            Euler angle array shaped ``(n_pixels, 3)`` in degrees.
        """

        missing = [
            field
            for field in ("Euler1", "Euler2", "Euler3")
            if field not in reader.catalog().scalars
        ]
        if missing:
            raise ValueError(
                "CTF source is missing required Euler angle columns: "
                f"{', '.join(missing)}. Re-export the CTF with Euler1/Euler2/Euler3 "
                "columns enabled, or provide band_annotation_json_path to use "
                "precomputed band-line annotations."
            )
        euler = np.column_stack(
            [
                reader.get_map("Euler1").ravel(order="C"),
                reader.get_map("Euler2").ravel(order="C"),
                reader.get_map("Euler3").ravel(order="C"),
            ]
        ).astype(np.float64)
        expected = reader.nx * reader.ny
        if euler.shape != (expected, 3):
            raise ValueError(
                f"CTF Euler table has shape {euler.shape}; expected ({expected}, 3)."
            )
        if not np.all(np.isfinite(euler)):
            bad = int(np.count_nonzero(~np.isfinite(euler)))
            raise ValueError(
                f"CTF Euler table contains {bad} non-finite values. Clean the CTF "
                "or remove invalid pixels before running direct CTF simulation."
            )
        return euler

    def _write_minimal_h5(
        self,
        reader: CtfPatternScanFileReader,
        patterns: np.ndarray,
        output_path: Path,
    ) -> None:
        """Write a minimal OH5-like HDF5 output container.

        Parameters:
            reader: Parsed CTF reader.
            patterns: Pattern stack shaped ``(ny, nx, height, width)``.
            output_path: HDF5 output path.

        Returns:
            None.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        scan_name = reader._scan_name
        ny, nx, pattern_height, pattern_width = patterns.shape
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset("Manufacturer", data="HKL/Oxford CTF")
            handle.create_dataset("Version", data="1.0")
            scan = handle.create_group(scan_name)
            ebsd = scan.create_group("EBSD")
            header = ebsd.create_group("Header")
            data = ebsd.create_group("Data")
            header.create_dataset("nColumns", data=np.array([nx], dtype=np.int32))
            header.create_dataset("nRows", data=np.array([ny], dtype=np.int32))
            header.create_dataset("Pattern Height", data=np.array([pattern_height], dtype=np.int32))
            header.create_dataset("Pattern Width", data=np.array([pattern_width], dtype=np.int32))
            header.create_dataset("Source Format", data="CTF")
            data.create_dataset("Pattern", data=patterns.reshape(nx * ny, pattern_height, pattern_width))
            self._write_scalar_if_available(reader, data, "CI", fallback_field="MAD")
            self._write_scalar_if_available(reader, data, "IQ", fallback_field="BC")
            for field_name in reader.catalog().list_scalar_fields():
                if field_name in data:
                    continue
                try:
                    data.create_dataset(field_name, data=reader.get_map(field_name).ravel(order="C"))
                except Exception as exc:
                    self._logger.warning("Skipping CTF scalar field %s: %s", field_name, exc)
        self._logger.info("Prepared CTF-derived HDF5 output at %s", output_path)

    def _write_scalar_if_available(
        self,
        reader: CtfPatternScanFileReader,
        data_group: h5py.Group,
        output_name: str,
        fallback_field: str,
    ) -> None:
        """Write a canonical scalar dataset when a source field is available.

        Parameters:
            reader: Parsed CTF reader.
            data_group: HDF5 data group.
            output_name: Canonical output dataset name.
            fallback_field: CTF source field to use when canonical name is absent.

        Returns:
            None.
        """

        for candidate in (output_name, fallback_field):
            try:
                data_group.create_dataset(
                    output_name,
                    data=reader.get_map(candidate).ravel(order="C").astype(np.float32),
                )
                return
            except KeyError:
                continue
        n_pixels = reader.nx * reader.ny
        data_group.create_dataset(output_name, data=np.zeros(n_pixels, dtype=np.float32))
        self._logger.warning(
            "CTF source has no %s/%s field; wrote zeros for %s.",
            output_name,
            fallback_field,
            output_name,
        )


def export_ctf_with_prias_metrics(
    modified_h5_path: Path,
    output_ang_path: Optional[Path] = None,
) -> Path:
    """Export a CTF-derived ANG file with computed PRIAS metric columns.

    Parameters:
        modified_h5_path: CTF-derived HDF5 file written by the automator.
        output_ang_path: Optional ANG destination path. Defaults to
            ``<modified_h5_stem>.ang``.

    Returns:
        Path to the written ANG file.

    Raises:
        FileNotFoundError: If the HDF5 output does not exist.
        KeyError: If required metadata or output datasets are missing.
        ValueError: If dataset lengths are inconsistent.
    """

    source_h5 = Path(modified_h5_path)
    if not source_h5.exists():
        raise FileNotFoundError(f"CTF-derived HDF5 file not found: {source_h5}")
    destination = Path(output_ang_path) if output_ang_path else source_h5.with_suffix(".ang")

    with h5py.File(source_h5, "r") as handle:
        scan_name = _find_scan_group_name(handle)
        header = handle[f"/{scan_name}/EBSD/Header"]
        data = handle[f"/{scan_name}/EBSD/Data"]
        nx = int(np.ravel(header["nColumns"][()])[0])
        ny = int(np.ravel(header["nRows"][()])[0])
        n_pixels = nx * ny

        columns = {
            "phi1": _read_required_flat(data, "Euler1", n_pixels),
            "PHI": _read_required_flat(data, "Euler2", n_pixels),
            "phi2": _read_required_flat(data, "Euler3", n_pixels),
            "x": _read_required_flat(data, "X", n_pixels),
            "y": _read_required_flat(data, "Y", n_pixels),
            "IQ": _read_first_available_flat(data, ("IQ", "BC"), n_pixels, default=0.0),
            "CI": _read_first_available_flat(data, ("CI", "MAD"), n_pixels, default=0.0),
            "Phase index": _read_first_available_flat(data, ("Phase",), n_pixels, default=1.0),
            "Fit": _read_first_available_flat(data, ("Fit", "MAD", "Error"), n_pixels, default=0.0),
            "PRIAS Bottom Strip": _read_required_flat(data, "Band_Width", n_pixels),
            "PRIAS Center Square": _read_required_flat(data, "psnr", n_pixels),
            "PRIAS Top Strip": _read_required_flat(data, "band_intensity_ratio", n_pixels),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("# HEADER: Start\n")
        handle.write("# SourceFormat: HKL/Oxford CTF\n")
        handle.write("# GeneratedBy: kikuchiBandAnalyzer CTF band-width export\n")
        handle.write("# Phase 1: CTF-derived phase metadata; verify material metadata before TSL import\n")
        handle.write("# COLUMN_HEADERS: " + ", ".join(columns.keys()) + "\n")
        handle.write(f"# NCOLS_EVEN: {nx}\n")
        handle.write(f"# NROWS: {ny}\n")
        handle.write("# HEADER: End\n")
        for index in range(n_pixels):
            row = []
            for key, values in columns.items():
                value = values[index]
                if key == "Phase index":
                    row.append(str(int(round(float(value)))))
                else:
                    row.append(f"{float(value):.6f}")
            handle.write("  ".join(row) + "\n")
    return destination


def _find_scan_group_name(handle: h5py.File) -> str:
    """Find the scan group in a CTF-derived HDF5 file.

    Parameters:
        handle: Open HDF5 handle.

    Returns:
        Scan group name.
    """

    for key, item in handle.items():
        if key not in {"Manufacturer", "Version"} and isinstance(item, h5py.Group):
            return key
    raise KeyError("No scan group found in CTF-derived HDF5 file.")


def _read_required_flat(data_group: h5py.Group, name: str, n_pixels: int) -> np.ndarray:
    """Read a required one-value-per-pixel dataset.

    Parameters:
        data_group: HDF5 data group.
        name: Dataset name.
        n_pixels: Expected flattened length.

    Returns:
        Flattened float64 dataset values.
    """

    if name not in data_group:
        raise KeyError(
            f"Required dataset '{name}' is missing from CTF-derived HDF5 output. "
            "Re-run the automator and ensure CTF metadata contains this field."
        )
    values = np.asarray(data_group[name][()], dtype=np.float64).ravel(order="C")
    if values.size != n_pixels:
        raise ValueError(
            f"Dataset '{name}' has {values.size} values; expected {n_pixels}."
        )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _read_first_available_flat(
    data_group: h5py.Group,
    names: tuple[str, ...],
    n_pixels: int,
    *,
    default: float,
) -> np.ndarray:
    """Read the first available dataset or return a default vector.

    Parameters:
        data_group: HDF5 data group.
        names: Candidate dataset names.
        n_pixels: Expected flattened length.
        default: Default value when no candidates exist.

    Returns:
        Flattened float64 values.
    """

    for name in names:
        if name in data_group:
            return _read_required_flat(data_group, name, n_pixels)
    return np.full(n_pixels, float(default), dtype=np.float64)
