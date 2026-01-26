"""Tests for derived field computations and HDF5 writing."""

from __future__ import annotations

import logging

import h5py
import numpy as np

from kikuchiBandAnalyzer.derived_fields import (
    build_default_registry,
    write_hdf5_dataset,
)


def test_band_intensity_diff_norm_handles_edge_cases() -> None:
    """Compute normalized intensity differences with zeros and NaNs."""

    inputs = {
        "efficientlineIntensity": np.array([2.0, 1.0, 0.0, np.nan], dtype=np.float32),
        "defficientlineIntensity": np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }
    registry = build_default_registry(logger=logging.getLogger(__name__))
    outputs = registry.compute(inputs)
    result = outputs["band_intensity_diff_norm"]
    assert np.isclose(result[0], 2.0 / 3.0, rtol=1e-5)
    assert result[1] == 0.0
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_registry_writes_hdf5_dataset(tmp_path) -> None:
    """Ensure derived fields are written with attributes."""

    inputs = {
        "efficientlineIntensity": np.array([2.0, 2.0], dtype=np.float32),
        "defficientlineIntensity": np.array([1.0, 3.0], dtype=np.float32),
    }
    registry = build_default_registry(logger=logging.getLogger(__name__))
    outputs = registry.compute(inputs)
    spec = registry.get_spec("band_intensity_diff_norm")
    assert spec is not None
    output_path = tmp_path / "derived.h5"
    dataset_path = f"/Scan/EBSD/Data/{spec.dataset_name}"
    with h5py.File(output_path, "w") as handle:
        handle.require_group("Scan/EBSD/Data")
        write_hdf5_dataset(
            handle, dataset_path, outputs[spec.name], attrs=spec.attrs
        )
    with h5py.File(output_path, "r") as handle:
        dataset = handle[dataset_path]
        assert dataset.shape == outputs[spec.name].shape
        assert dataset.attrs.get("formula") == spec.attrs["formula"]
