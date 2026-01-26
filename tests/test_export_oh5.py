"""Tests for exporting comparison results to OH5."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.export_oh5 import Oh5ComparisonExporter
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    build_alignment_from_parameters,
)


def _create_min_oh5(
    path: Path,
    scan_name: str,
    nx: int,
    ny: int,
    iq: np.ndarray,
    ci: np.ndarray,
    phase: np.ndarray,
    iq_as_1d: bool = True,
) -> None:
    """Create a minimal OH5-like HDF5 file for tests.

    Parameters:
        path: Destination path.
        scan_name: Scan group name.
        nx: Number of columns.
        ny: Number of rows.
        iq: IQ map shaped (ny, nx).
        ci: CI map shaped (ny, nx).
        phase: Phase map shaped (ny, nx).
        iq_as_1d: Whether to store IQ as a flattened 1D dataset.

    Returns:
        None.
    """

    with h5py.File(path, "w") as handle:
        handle.create_dataset("Manufacturer", data="Debug")
        handle.create_dataset("Version", data="1.0")
        scan = handle.create_group(scan_name)
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")
        header.create_dataset("nColumns", data=np.array([nx]))
        header.create_dataset("nRows", data=np.array([ny]))
        iq_payload = np.ravel(iq, order="C") if iq_as_1d else iq
        data.create_dataset("IQ", data=np.asarray(iq_payload, dtype=np.float32))
        data.create_dataset("CI", data=np.asarray(ci, dtype=np.float32))
        data.create_dataset("Phase", data=np.asarray(np.ravel(phase, order="C"), dtype=np.int32))


def test_exporter_writes_common_scalar_fields_and_metadata(tmp_path) -> None:
    """Export ratio comparisons into a scan-A template OH5 file."""

    scan_name = "Scan"
    nx, ny = 2, 2
    iq_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    iq_b = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    ci_a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ci_b = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float32)
    phase_a = np.array([[1, 1], [2, 2]], dtype=np.int32)
    phase_b = np.array([[1, 1], [2, 2]], dtype=np.int32)

    path_a = tmp_path / "scan_a.oh5"
    path_b = tmp_path / "scan_b.oh5"
    _create_min_oh5(path_a, scan_name, nx, ny, iq_a, ci_a, phase_a, iq_as_1d=True)
    _create_min_oh5(path_b, scan_name, nx, ny, iq_b, ci_b, phase_b, iq_as_1d=True)

    dataset_a = OH5ScanFileReader.from_path(path_a)
    dataset_b = OH5ScanFileReader.from_path(path_b)
    try:
        engine = ComparisonEngine(dataset_a, dataset_b, config={}, alignment=None)
        expected_iq = engine.map_triplet("IQ", "ratio")["D"]
        expected_ci = engine.map_triplet("CI", "ratio")["D"]
        exporter = Oh5ComparisonExporter(logger=logging.getLogger(__name__))
        output_path = tmp_path / "out.oh5"
        result = exporter.export(
            dataset_a,
            dataset_b,
            engine,
            output_path,
            mode="ratio",
            alignment=None,
            overwrite=True,
            excluded_fields=["Phase"],
        )
    finally:
        dataset_a.close()
        dataset_b.close()

    with h5py.File(output_path, "r") as handle:
        iq_out = handle[f"/{scan_name}/EBSD/Data/IQ"][()]
        ci_out = handle[f"/{scan_name}/EBSD/Data/CI"][()]
        phase_out = handle[f"/{scan_name}/EBSD/Data/Phase"][()]
        compare_group = handle[f"/{scan_name}/EBSD/Compare"]
        mode = compare_group["mode"][()].decode("utf-8")
        exported_fields = [val.decode("utf-8") for val in compare_group["exported_fields"][()]]
        skipped_fields = [val.decode("utf-8") for val in compare_group["skipped_fields"][()]]

    assert mode == "ratio"
    assert "IQ" in exported_fields
    assert "CI" in exported_fields
    assert "Phase" in skipped_fields
    assert set(result.exported_fields) == set(exported_fields)

    iq_out_map = np.reshape(iq_out, (ny, nx)) if iq_out.ndim == 1 else iq_out
    assert np.allclose(iq_out_map, expected_iq, equal_nan=True)
    assert np.allclose(ci_out, expected_ci, equal_nan=True)
    assert np.array_equal(phase_out, np.ravel(phase_a, order="C"))


def test_exporter_embeds_alignment_metadata(tmp_path) -> None:
    """Embed alignment metadata in the exported OH5 file."""

    scan_name = "Scan"
    nx, ny = 3, 3
    iq_a = np.zeros((ny, nx), dtype=np.float32)
    iq_b = np.zeros((ny, nx), dtype=np.float32)
    iq_a[1, 1] = 1.0
    iq_b[1, 2] = 1.0
    ci = np.zeros((ny, nx), dtype=np.float32)
    phase = np.ones((ny, nx), dtype=np.int32)
    path_a = tmp_path / "scan_a_align.oh5"
    path_b = tmp_path / "scan_b_align.oh5"
    _create_min_oh5(path_a, scan_name, nx, ny, iq_a, ci, phase, iq_as_1d=False)
    _create_min_oh5(path_b, scan_name, nx, ny, iq_b, ci, phase, iq_as_1d=False)

    alignment = build_alignment_from_parameters(0.0, [-1.0, 0.0])
    dataset_a = OH5ScanFileReader.from_path(path_a)
    dataset_b = OH5ScanFileReader.from_path(path_b)
    try:
        config = {"alignment": {"map_interpolation": "nearest", "warp": {"cval": 0.0}}}
        engine = ComparisonEngine(dataset_a, dataset_b, config=config, alignment=alignment)
        expected = engine.map_triplet("IQ", "delta")["D"]
        exporter = Oh5ComparisonExporter(logger=logging.getLogger(__name__))
        output_path = tmp_path / "out_align.oh5"
        exporter.export(
            dataset_a,
            dataset_b,
            engine,
            output_path,
            mode="delta",
            alignment=alignment,
            overwrite=True,
            excluded_fields=["Phase"],
        )
    finally:
        dataset_a.close()
        dataset_b.close()

    with h5py.File(output_path, "r") as handle:
        iq_out = handle[f"/{scan_name}/EBSD/Data/IQ"][()]
        alignment_group = handle[f"/{scan_name}/EBSD/Compare/alignment"]
        assert bool(alignment_group.attrs["enabled"]) is True
        assert "alignment_yaml" in alignment_group
        assert "matrix" in alignment_group
        assert alignment_group["matrix"][()].shape == (3, 3)
    assert np.allclose(iq_out, expected, equal_nan=True)
