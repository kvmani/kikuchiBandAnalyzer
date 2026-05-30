"""Limited HKL/Oxford CTF to TSL-style ANG and OH5 conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Optional

import h5py
import numpy as np
from PIL import Image


_CTF_PHASE_LINE = re.compile(r"^\s*Phase\s+", re.IGNORECASE)
_ASCII = h5py.string_dtype(encoding="ascii")


@dataclass(frozen=True)
class CtfTable:
    """Parsed CTF table and grid metadata.

    Parameters:
        header: Mapping of CTF header keys to raw string values.
        columns: CTF data column names in file order.
        table: Mapping of CTF column names to one-dimensional numeric arrays.
        nx: Number of scan columns.
        ny: Number of scan rows.
        x_step: Scan step in the CTF X direction.
        y_step: Scan step in the CTF Y direction.
    """

    header: dict[str, str]
    columns: list[str]
    table: dict[str, np.ndarray]
    nx: int
    ny: int
    x_step: float
    y_step: float


@dataclass(frozen=True)
class HklToTslConversionResult:
    """Paths and summary metadata produced by an HKL-to-TSL conversion.

    Parameters:
        h5_path: Generated TSL-style HDF5 path.
        oh5_path: Generated OH5 alias path.
        ang_path: Generated ANG path.
        scan_name: Scan group name written to the HDF5 files.
        pattern_mapping: Filename template used to resolve HKL patterns.
        ignored_pattern_count: Number of pattern files not used by the CTF grid.
    """

    h5_path: Path
    oh5_path: Path
    ang_path: Path
    scan_name: str
    pattern_mapping: str
    ignored_pattern_count: int


def parse_ctf_file(ctf_path: Path | str) -> CtfTable:
    """Parse a CTF text file into numeric columns and grid metadata.

    Parameters:
        ctf_path: Path to the HKL/Oxford CTF file.

    Returns:
        Parsed CTF table and scan grid metadata.

    Raises:
        ValueError: If required columns or grid metadata are missing.
    """

    path = Path(ctf_path)
    header: dict[str, str] = {}
    columns: Optional[list[str]] = None
    rows: list[list[float]] = []

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if columns is None and _CTF_PHASE_LINE.match(line):
            columns = parts
            continue
        if columns is None:
            if len(parts) >= 2:
                header[parts[0]] = " ".join(parts[1:])
            continue
        if len(parts) < len(columns):
            continue
        try:
            rows.append([float(value) for value in parts[: len(columns)]])
        except ValueError:
            continue

    if columns is None:
        raise ValueError(f"CTF data header was not found in {path}.")
    if not rows:
        raise ValueError(f"CTF file contains no numeric data rows: {path}.")

    required = {"Phase", "X", "Y", "Euler1", "Euler2", "Euler3", "MAD", "BC", "BS"}
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"CTF file is missing required columns: {', '.join(missing)}.")

    array = np.asarray(rows, dtype=np.float64)
    table = {name: array[:, index] for index, name in enumerate(columns)}
    nx = _read_header_int(header, "XCells")
    ny = _read_header_int(header, "YCells")
    x_step = _read_header_float(header, "XStep")
    y_step = _read_header_float(header, "YStep")
    if nx <= 0 or ny <= 0:
        raise ValueError("CTF XCells and YCells must be positive.")
    if array.shape[0] != nx * ny:
        raise ValueError(
            f"CTF has {array.shape[0]} rows, but XCells*YCells is {nx * ny}."
        )
    return CtfTable(
        header=header,
        columns=columns,
        table=table,
        nx=nx,
        ny=ny,
        x_step=x_step,
        y_step=y_step,
    )


def convert_hkl_ctf_fixture_to_tsl(
    ctf_path: Path | str,
    pattern_dir: Path | str,
    reference_h5_path: Path | str,
    reference_ang_path: Path | str,
    output_dir: Path | str,
    *,
    scan_name: Optional[str] = None,
    phase_name: str = "Cr",
    phase_formula: str = "Cr",
    lattice_parameter: float = 2.91,
    pattern_template: str = "{x}_{y}.tiff",
    logger: Optional[logging.Logger] = None,
) -> HklToTslConversionResult:
    """Convert the HKL FCC fixture to DA-compatible ANG, H5, and OH5 files.

    Parameters:
        ctf_path: Path to ``Subset.ctf``.
        pattern_dir: Directory containing one TIFF pattern per HKL pixel.
        reference_h5_path: DA HDF5/OH5 file used as the TSL metadata template.
        reference_ang_path: DA ANG file used as the ANG header style template.
        output_dir: Directory where generated files are written.
        scan_name: Optional output scan group name. Defaults to the CTF stem.
        phase_name: FCC phase material name written into copied Ni metadata.
        phase_formula: FCC phase formula written into copied Ni metadata.
        lattice_parameter: Cubic lattice parameter written for a, b, and c.
        pattern_template: Filename template using ``x`` and ``y`` indices.
        logger: Optional logger instance.

    Returns:
        Conversion result containing generated paths and mapping details.
    """

    log = logger or logging.getLogger(__name__)
    ctf = parse_ctf_file(ctf_path)
    patterns = load_mapped_patterns(ctf, pattern_dir, pattern_template=pattern_template)
    fields = build_tsl_fields(ctf, patterns)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_scan_name = scan_name or Path(ctf_path).stem
    h5_path = destination / f"{resolved_scan_name}.h5"
    oh5_path = destination / f"{resolved_scan_name}.oh5"
    ang_path = destination / f"{resolved_scan_name}.ang"

    write_tsl_h5(
        reference_h5_path=Path(reference_h5_path),
        output_h5_path=h5_path,
        scan_name=resolved_scan_name,
        ctf=ctf,
        fields=fields,
        phase_name=phase_name,
        phase_formula=phase_formula,
        lattice_parameter=lattice_parameter,
    )
    write_tsl_h5(
        reference_h5_path=Path(reference_h5_path),
        output_h5_path=oh5_path,
        scan_name=resolved_scan_name,
        ctf=ctf,
        fields=fields,
        phase_name=phase_name,
        phase_formula=phase_formula,
        lattice_parameter=lattice_parameter,
    )
    write_ang_file(
        reference_ang_path=Path(reference_ang_path),
        output_ang_path=ang_path,
        ctf=ctf,
        fields=fields,
        phase_name=phase_name,
        phase_formula=phase_formula,
        lattice_parameter=lattice_parameter,
    )

    available = len([p for p in Path(pattern_dir).iterdir() if p.is_file()])
    ignored = max(0, available - ctf.nx * ctf.ny)
    log.info(
        "Converted %s using pattern mapping %s; ignored %d extra pattern files.",
        ctf_path,
        pattern_template,
        ignored,
    )
    return HklToTslConversionResult(
        h5_path=h5_path,
        oh5_path=oh5_path,
        ang_path=ang_path,
        scan_name=resolved_scan_name,
        pattern_mapping=pattern_template,
        ignored_pattern_count=ignored,
    )


def load_mapped_patterns(
    ctf: CtfTable,
    pattern_dir: Path | str,
    *,
    pattern_template: str = "{x}_{y}.tiff",
) -> np.ndarray:
    """Load CTF-mapped TIFF patterns using explicit filename coordinates.

    Parameters:
        ctf: Parsed CTF table.
        pattern_dir: Directory containing pattern image files.
        pattern_template: Filename template using zero-based ``x`` and ``y``.

    Returns:
        Pattern stack shaped ``(ny * nx, height, width)`` as ``uint16``.

    Raises:
        FileNotFoundError: If a required pattern image is missing.
        ValueError: If patterns have inconsistent dimensions.
    """

    root = Path(pattern_dir)
    stack: list[np.ndarray] = []
    expected_shape: Optional[tuple[int, int]] = None
    for y in range(ctf.ny):
        for x in range(ctf.nx):
            path = root / pattern_template.format(x=x, y=y, row=y, col=x, index=y * ctf.nx + x)
            if not path.exists():
                raise FileNotFoundError(f"Missing pattern for CTF pixel x={x}, y={y}: {path}")
            with Image.open(path) as image:
                arr8 = np.asarray(image.convert("L"), dtype=np.uint16)
            arr16 = arr8 * np.uint16(257)
            if expected_shape is None:
                expected_shape = arr16.shape
            elif arr16.shape != expected_shape:
                raise ValueError(
                    f"Pattern {path} has shape {arr16.shape}; expected {expected_shape}."
                )
            stack.append(arr16)
    return np.asarray(stack, dtype=np.uint16)


def build_tsl_fields(ctf: CtfTable, patterns: np.ndarray) -> dict[str, np.ndarray]:
    """Build DA-compatible TSL scalar fields from CTF columns and patterns.

    Parameters:
        ctf: Parsed CTF table.
        patterns: Pattern stack shaped ``(ny * nx, height, width)``.

    Returns:
        Mapping of TSL/EDAX dataset names to arrays.
    """

    n_pixels = ctf.nx * ctf.ny
    mad = np.asarray(ctf.table["MAD"], dtype=np.float32)
    max_mad = float(np.nanmax(mad)) if mad.size else 0.0
    ci = np.ones_like(mad, dtype=np.float32) if max_mad <= 0 else 1.0 - mad / max_mad
    ci = np.clip(np.nan_to_num(ci, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    valid_mask = np.asarray(ctf.table["Phase"] > 0, dtype=bool)
    height, width = patterns.shape[-2:]
    bottom = patterns[:, (height * 2) // 3 :, :].mean(axis=(1, 2)).astype(np.float32)
    top = patterns[:, : height // 3, :].mean(axis=(1, 2)).astype(np.float32)
    half = min(height, width) // 4
    cy, cx = height // 2, width // 2
    center = patterns[:, cy - half : cy + half, cx - half : cx + half].mean(axis=(1, 2)).astype(np.float32)

    phase = np.where(valid_mask, 0, -1).astype(np.int8)
    valid = np.where(valid_mask, 0, 2).astype(np.int8)
    fields = {
        "Phi1": np.deg2rad(ctf.table["Euler1"]).astype(np.float32),
        "Phi": np.deg2rad(ctf.table["Euler2"]).astype(np.float32),
        "Phi2": np.deg2rad(ctf.table["Euler3"]).astype(np.float32),
        "X Position": ctf.table["X"].astype(np.float32),
        "Y Position": ctf.table["Y"].astype(np.float32),
        "IQ": ctf.table["BC"].astype(np.float32),
        "CI": ci.astype(np.float32),
        "Phase": phase,
        "SEM Signal": ctf.table["BS"].astype(np.int32),
        "Fit": mad.astype(np.float32),
        "PRIAS Bottom Strip": bottom,
        "PRIAS Center Square": center,
        "PRIAS Top Strip": top,
        "Valid": valid,
        "Pattern": patterns,
    }
    for name, values in fields.items():
        if name != "Pattern" and values.shape[0] != n_pixels:
            raise ValueError(f"Field {name} has {values.shape[0]} values; expected {n_pixels}.")
    return fields


def write_tsl_h5(
    reference_h5_path: Path,
    output_h5_path: Path,
    scan_name: str,
    ctf: CtfTable,
    fields: dict[str, np.ndarray],
    phase_name: str,
    phase_formula: str,
    lattice_parameter: float,
) -> None:
    """Write a TSL-style HDF5/OH5 file using DA metadata as a template.

    Parameters:
        reference_h5_path: Source DA HDF5 file.
        output_h5_path: Destination HDF5/OH5 path.
        scan_name: Output top-level scan group name.
        ctf: Parsed CTF metadata.
        fields: TSL-compatible data arrays.
        phase_name: Material name to write under phase metadata.
        phase_formula: Formula to write under phase metadata.
        lattice_parameter: Cubic lattice parameter for a, b, and c.

    Returns:
        None.
    """

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(reference_h5_path, "r") as source, h5py.File(output_h5_path, "w") as target:
        source_scan = _discover_scan_group(source)
        source.copy(source[source_scan], target, name=scan_name)
        target.create_dataset("Manufacturer", data=np.asarray([b"EDAX"]))
        target.create_dataset("Version", data=np.asarray([b"OIM Analysis 8.6.107 x64 [17 Jan 2023]"]))
        data = target[f"{scan_name}/EBSD/Data"]
        for key in list(data.keys()):
            del data[key]
        for name, values in fields.items():
            data.create_dataset(name, data=values)
        header = target[f"{scan_name}/EBSD/Header"]
        _replace_dataset(header, "nColumns", np.array([ctf.nx], dtype=np.int32))
        _replace_dataset(header, "nRows", np.array([ctf.ny], dtype=np.int32))
        _replace_dataset(header, "Step X", np.array([ctf.x_step], dtype=np.float32))
        _replace_dataset(header, "Step Y", np.array([ctf.y_step], dtype=np.float32))
        _replace_dataset(header, "Pattern Height", np.array([fields["Pattern"].shape[1]], dtype=np.int32))
        _replace_dataset(header, "Pattern Width", np.array([fields["Pattern"].shape[2]], dtype=np.int32))
        _replace_dataset(header, "Pattern Bit Depth", np.array([16], dtype=np.int32))
        _replace_dataset(header, "Sample Tilt", np.array([70.0], dtype=np.float64))
        _replace_dataset(header, "Voltage[kV]", np.array([25.0], dtype=np.float64))
        _replace_dataset(header, "Working Distance", np.array([18.0], dtype=np.float64))
        pc_group = header.require_group("Pattern Center Calibration")
        _replace_dataset(pc_group, "x-star", np.array([0.457], dtype=np.float64))
        _replace_dataset(pc_group, "y-star", np.array([0.584], dtype=np.float64))
        phase_group = header.require_group("Phase").require_group("1")
        _replace_dataset(phase_group, "MaterialName", np.asarray([phase_name.encode("ascii")], dtype=_ASCII))
        _replace_dataset(phase_group, "Formula", np.asarray([phase_formula.encode("ascii")], dtype=_ASCII))
        for key in ("Lattice Constant a", "Lattice Constant b", "Lattice Constant c"):
            _replace_dataset(phase_group, key, np.array([lattice_parameter], dtype=np.float64))
        for key in (
            "Lattice Constant alpha",
            "Lattice Constant beta",
            "Lattice Constant gamma",
        ):
            _replace_dataset(phase_group, key, np.array([90.0], dtype=np.float64))


def write_ang_file(
    reference_ang_path: Path,
    output_ang_path: Path,
    ctf: CtfTable,
    fields: dict[str, np.ndarray],
    phase_name: str,
    phase_formula: str,
    lattice_parameter: float,
) -> None:
    """Write a DA-style ANG file from CTF-derived TSL fields.

    Parameters:
        reference_ang_path: DA ANG file providing header style and hkl families.
        output_ang_path: Destination ANG path.
        ctf: Parsed CTF metadata.
        fields: TSL-compatible data arrays.
        phase_name: Material name for the ANG phase block.
        phase_formula: Formula for the ANG phase block.
        lattice_parameter: Cubic lattice parameter for the ANG phase block.

    Returns:
        None.
    """

    output_ang_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = _rewrite_ang_header(
        reference_ang_path=reference_ang_path,
        ctf=ctf,
        phase_name=phase_name,
        phase_formula=phase_formula,
        lattice_parameter=lattice_parameter,
    )
    with output_ang_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(header_lines)
        for index in range(ctf.nx * ctf.ny):
            values = [
                fields["Phi1"][index],
                fields["Phi"][index],
                fields["Phi2"][index],
                fields["X Position"][index],
                fields["Y Position"][index],
                fields["IQ"][index],
                fields["CI"][index],
                fields["Phase"][index],
                fields["SEM Signal"][index],
                fields["Fit"][index],
                fields["PRIAS Bottom Strip"][index],
                fields["PRIAS Center Square"][index],
                fields["PRIAS Top Strip"][index],
            ]
            handle.write(
                f"{values[0]:10.5f} {values[1]:10.5f} {values[2]:10.5f} "
                f"{values[3]:12.5f} {values[4]:12.5f} {values[5]:.6g} "
                f"{values[6]:6.3f} {int(values[7]):2d} {int(values[8]):6d} "
                f"{values[9]:7.3f} {values[10]:.6f} {values[11]:.6f} {values[12]:.6f}\n"
            )


def _rewrite_ang_header(
    reference_ang_path: Path,
    ctf: CtfTable,
    phase_name: str,
    phase_formula: str,
    lattice_parameter: float,
) -> list[str]:
    """Rewrite DA ANG header lines for the HKL fixture.

    Parameters:
        reference_ang_path: DA ANG template path.
        ctf: Parsed CTF metadata.
        phase_name: Replacement material name.
        phase_formula: Replacement formula.
        lattice_parameter: Replacement cubic lattice parameter.

    Returns:
        Updated ANG header lines through ``# HEADER: End``.
    """

    lines = reference_ang_path.read_text(encoding="utf-8").splitlines(keepends=True)
    header: list[str] = []
    for line in lines:
        if line.startswith("# TEM_PIXperUM"):
            header.append("# TEM_PIXperUM          1.000000\n")
        elif line.startswith("# x-star"):
            header.append("# x-star                0.457000\n")
        elif line.startswith("# y-star"):
            header.append("# y-star                0.584000\n")
        elif line.startswith("# WorkingDistance"):
            header.append("# WorkingDistance       18.000000\n")
        elif line.startswith("# SampleTiltAngle"):
            header.append("# SampleTiltAngle       70.000000\n")
        elif line.startswith("# CameraElevationAngle"):
            header.append("# CameraElevationAngle  5.000000\n")
        elif line.startswith("# MaterialName"):
            header.append(f"# MaterialName  \t{phase_name}\n")
        elif line.startswith("# Formula"):
            header.append(f"# Formula     \t{phase_formula}\n")
        elif line.startswith("# LatticeConstants"):
            header.append(
                f"# LatticeConstants      {lattice_parameter:.3f} {lattice_parameter:.3f} "
                f"{lattice_parameter:.3f}  90.000  90.000  90.000\n"
            )
        elif line.startswith("# XSTEP:"):
            header.append(f"# XSTEP: {ctf.x_step:.6f}\n")
        elif line.startswith("# YSTEP:"):
            header.append(f"# YSTEP: {ctf.y_step:.6f}\n")
        elif line.startswith("# NCOLS_ODD:"):
            header.append(f"# NCOLS_ODD: {ctf.nx}\n")
        elif line.startswith("# NCOLS_EVEN:"):
            header.append(f"# NCOLS_EVEN: {ctf.nx}\n")
        elif line.startswith("# NROWS:"):
            header.append(f"# NROWS: {ctf.ny}\n")
        elif line.startswith("# COLUMN_COUNT:"):
            header.append("# COLUMN_COUNT: 13\n")
        elif line.startswith("# COLUMN_HEADERS:"):
            header.append(
                "# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase index, "
                "SEM, Fit, PRIAS Bottom Strip, PRIAS Center Square, PRIAS Top Strip\n"
            )
        elif line.startswith("# HEADER: End"):
            header.append("# SOURCE: HKL/Oxford CTF converted with kikuchiBandAnalyzer FCC fixture converter\n")
            header.append(line)
            break
        else:
            header.append(line)
    return header


def _read_header_int(header: dict[str, str], key: str) -> int:
    """Read a CTF integer header value.

    Parameters:
        header: CTF header mapping.
        key: Header key to read.

    Returns:
        Parsed integer value, or zero when missing.
    """

    return int(float(header.get(key, "0").split()[0]))


def _read_header_float(header: dict[str, str], key: str) -> float:
    """Read a CTF floating-point header value.

    Parameters:
        header: CTF header mapping.
        key: Header key to read.

    Returns:
        Parsed float value, or zero when missing.
    """

    return float(header.get(key, "0").split()[0])


def _discover_scan_group(handle: h5py.File) -> str:
    """Find the first non-metadata scan group in a DA-style HDF5 file.

    Parameters:
        handle: Open HDF5 file handle.

    Returns:
        Name of the discovered scan group.
    """

    for key in handle.keys():
        if key not in {"Manufacturer", "Version"} and isinstance(handle[key], h5py.Group):
            return key
    raise ValueError("No EBSD scan group found in reference HDF5 file.")


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Replace a dataset inside an HDF5 group.

    Parameters:
        group: HDF5 group containing the dataset.
        name: Dataset name to replace.
        data: New dataset payload.

    Returns:
        None.
    """

    if name in group:
        del group[name]
    group.create_dataset(name, data=data)
