"""Tests for OH5-to-ANG workflow mapping and export."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from kikuchiBandAnalyzer.oh5_to_ang_exporter.workflow import (
    export_oh5_to_ang,
    parse_ang_template,
    read_oh5_scalar_catalog,
    run_sanity_checks,
)



def _create_test_oh5(path: Path, scan_name: str, nrows: int, ncols: int) -> None:
    """Create a minimal OH5 file with scalar fields used in mapping tests.

    Parameters:
        path: Destination OH5 path.
        scan_name: Top-level scan group name.
        nrows: Number of rows in scan grid.
        ncols: Number of columns in scan grid.

    Returns:
        None.
    """

    n_pixels = nrows * ncols
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Manufacturer", data=np.array([b"EDAX"], dtype="S4"))
        handle.create_dataset("Version", data=np.array([b"1.0"], dtype="S3"))

        scan = handle.create_group(scan_name)
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        header.create_dataset("nRows", data=np.array([nrows], dtype=np.int32))
        header.create_dataset("nColumns", data=np.array([ncols], dtype=np.int32))

        data = ebsd.create_group("Data")
        data.create_dataset("Phi1", data=np.linspace(0.10, 0.60, n_pixels, dtype=np.float32))
        data.create_dataset("Phi", data=np.linspace(1.10, 1.60, n_pixels, dtype=np.float32))
        data.create_dataset("Phi2", data=np.linspace(2.10, 2.60, n_pixels, dtype=np.float32))
        data.create_dataset("Band_Width", data=np.linspace(10.0, 15.0, n_pixels, dtype=np.float32))
        data.create_dataset("psnr", data=np.linspace(20.0, 25.0, n_pixels, dtype=np.float32))
        data.create_dataset(
            "band_intensity_ratio",
            data=np.linspace(1.0, 1.5, n_pixels, dtype=np.float32),
        )
        data.create_dataset("CI", data=np.linspace(0.1, 0.6, n_pixels, dtype=np.float32))



def _create_test_ang(path: Path, nrows: int, ncols_even: int, headers: list[str]) -> None:
    """Create a minimal ANG template for export tests.

    Parameters:
        path: Destination ANG path.
        nrows: Number of scan rows.
        ncols_even: Number of scan columns in ANG header.
        headers: Column header names in ANG order.

    Returns:
        None.
    """

    total = nrows * ncols_even
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# HEADER: Start\n")
        handle.write("# NCOLS_EVEN: " + str(ncols_even) + "\n")
        handle.write("# NROWS: " + str(nrows) + "\n")
        handle.write("# COLUMN_HEADERS: " + ", ".join(headers) + "\n")
        handle.write("# HEADER: End\n")
        for idx in range(total):
            row_values = [
                f"{100.0 + idx:.3f}",
                f"{200.0 + idx:.3f}",
                f"{300.0 + idx:.3f}",
                f"{400.0 + idx:.3f}",
                f"{500.0 + idx:.3f}",
                f"{600.0 + idx:.3f}",
                f"{700.0 + idx:.3f}",
                f"{800.0 + idx:.3f}",
                f"{900.0 + idx:.3f}",
                f"{1000.0 + idx:.3f}",
                f"{1100.0 + idx:.3f}",
                f"{1200.0 + idx:.3f}",
                f"{1300.0 + idx:.3f}",
            ]
            handle.write("  ".join(row_values) + "\n")



def _read_data_rows(path: Path, ncols: int) -> list[list[str]]:
    """Read ANG data rows split into tokens.

    Parameters:
        path: ANG file path.
        ncols: Expected number of columns.

    Returns:
        Parsed data rows as token lists.
    """

    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tokens = line.split()
        if len(tokens) == ncols:
            rows.append(tokens)
    return rows



def test_parse_and_sanity_checks_match_counts(tmp_path) -> None:
    """ANG and OH5 pixel counts should pass sanity checks when aligned."""

    nrows = 2
    ncols = 3
    oh5_path = tmp_path / "scan_modified.oh5"
    ang_path = tmp_path / "scan.ang"

    headers = [
        "phi1",
        "PHI",
        "phi2",
        "x",
        "y",
        "IQ",
        "CI",
        "Phase index",
        "SEM",
        "Fit",
        "PRIAS Bottom Strip",
        "PRIAS Center Square",
        "PRIAS Top Strip",
    ]

    _create_test_oh5(oh5_path, scan_name="Scan", nrows=nrows, ncols=ncols)
    _create_test_ang(ang_path, nrows=nrows, ncols_even=ncols, headers=headers)

    template = parse_ang_template(ang_path)
    catalog = read_oh5_scalar_catalog(oh5_path)
    run_sanity_checks(template, catalog)

    assert template.expected_pixels == nrows * ncols
    assert catalog.n_pixels == nrows * ncols



def test_export_maps_locked_and_user_columns_with_fallback(tmp_path) -> None:
    """Exporter should enforce locked mappings and apply user mappings only where requested."""

    nrows = 2
    ncols = 3
    oh5_path = tmp_path / "scan_modified.oh5"
    ang_path = tmp_path / "scan.ang"
    output_path = tmp_path / "scan_exported.ang"

    headers = [
        "phi1",
        "PHI",
        "phi2",
        "x",
        "y",
        "IQ",
        "CI",
        "Phase index",
        "SEM",
        "Fit",
        "PRIAS Bottom Strip",
        "PRIAS Center Square",
        "PRIAS Top Strip",
    ]

    _create_test_oh5(oh5_path, scan_name="Scan", nrows=nrows, ncols=ncols)
    _create_test_ang(ang_path, nrows=nrows, ncols_even=ncols, headers=headers)

    export_oh5_to_ang(
        oh5_path=oh5_path,
        ang_path=ang_path,
        user_mappings=[
            ("Band_Width", "IQ"),
            ("band_intensity_ratio", "Fit"),
        ],
        output_ang_path=output_path,
        include_mapping_note=False,
    )

    rows = _read_data_rows(output_path, ncols=len(headers))
    assert len(rows) == nrows * ncols

    with h5py.File(oh5_path, "r") as handle:
        phi1 = handle["/Scan/EBSD/Data/Phi1"][()]
        phi = handle["/Scan/EBSD/Data/Phi"][()]
        phi2 = handle["/Scan/EBSD/Data/Phi2"][()]
        band_width = handle["/Scan/EBSD/Data/Band_Width"][()]
        ratio = handle["/Scan/EBSD/Data/band_intensity_ratio"][()]

    # locked columns
    assert rows[0][0] == f"{float(phi1[0]):.6f}"
    assert rows[0][1] == f"{float(phi[0]):.6f}"
    assert rows[0][2] == f"{float(phi2[0]):.6f}"

    # user mapped columns
    assert rows[0][5] == f"{float(band_width[0]):.6f}"
    assert rows[0][9] == f"{float(ratio[0]):.6f}"

    # unmapped columns should remain from original ANG template
    assert rows[0][6] == "700.000"  # CI column retained from template
    assert rows[0][10] == "1100.000"  # PRIAS Bottom Strip retained from template



def test_export_rejects_locked_column_override(tmp_path) -> None:
    """User mapping should not be allowed to override locked phi columns."""

    nrows = 2
    ncols = 3
    oh5_path = tmp_path / "scan_modified.oh5"
    ang_path = tmp_path / "scan.ang"

    headers = [
        "phi1",
        "PHI",
        "phi2",
        "x",
        "y",
        "IQ",
        "CI",
        "Phase index",
        "SEM",
        "Fit",
        "PRIAS Bottom Strip",
        "PRIAS Center Square",
        "PRIAS Top Strip",
    ]

    _create_test_oh5(oh5_path, scan_name="Scan", nrows=nrows, ncols=ncols)
    _create_test_ang(ang_path, nrows=nrows, ncols_even=ncols, headers=headers)

    with pytest.raises(ValueError, match="locked"):
        export_oh5_to_ang(
            oh5_path=oh5_path,
            ang_path=ang_path,
            user_mappings=[("Band_Width", "phi1")],
        )



def test_export_can_include_mapping_note_line(tmp_path) -> None:
    """Exporter should optionally write one ASCII mapping note line in the header."""

    nrows = 2
    ncols = 3
    oh5_path = tmp_path / "scan_modified.oh5"
    ang_path = tmp_path / "scan.ang"
    output_path = tmp_path / "scan_exported_note.ang"

    headers = [
        "phi1",
        "PHI",
        "phi2",
        "x",
        "y",
        "IQ",
        "CI",
        "Phase index",
        "SEM",
        "Fit",
        "PRIAS Bottom Strip",
        "PRIAS Center Square",
        "PRIAS Top Strip",
    ]

    _create_test_oh5(oh5_path, scan_name="Scan", nrows=nrows, ncols=ncols)
    _create_test_ang(ang_path, nrows=nrows, ncols_even=ncols, headers=headers)

    export_oh5_to_ang(
        oh5_path=oh5_path,
        ang_path=ang_path,
        user_mappings=[("Band_Width", "PRIAS Bottom Strip")],
        output_ang_path=output_path,
        include_mapping_note=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "# KBA_OH5_TO_ANG_MAPPING:" in text
    assert "Band_Width ---> PRIAS Bottom Strip" in text
