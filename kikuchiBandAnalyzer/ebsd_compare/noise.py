"""Noise generation utilities for OH5 files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


class NoisyOh5Generator:
    """Generate a noisy OH5 file by perturbing selected scalar datasets."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        sigma_map: Dict[str, float],
        seed: int,
        logger: logging.Logger,
    ) -> None:
        """Initialize the generator.

        Parameters:
            input_path: Path to the input OH5 file.
            output_path: Path for the noisy output OH5 file.
            sigma_map: Mapping of dataset names to noise sigma values.
            seed: Random seed for deterministic noise.
            logger: Logger instance.
        """

        self._input_path = Path(input_path)
        self._output_path = Path(output_path)
        self._sigma_map = sigma_map
        self._seed = seed
        self._logger = logger

    def run(self) -> None:
        """Generate the noisy OH5 output file."""

        self._logger.info("Generating noisy OH5: %s", self._output_path)
        rng = np.random.default_rng(self._seed)
        with h5py.File(self._input_path, "r") as source, h5py.File(
            self._output_path, "w"
        ) as target:
            self._copy_hdf5(source, target)
            for field, sigma in self._sigma_map.items():
                dataset_path = self._find_dataset_path(source, field)
                if dataset_path is None:
                    raise KeyError(
                        f"Required dataset '{field}' not found in {self._input_path}."
                    )
                dataset = target[dataset_path]
                data = dataset[()]
                noise = rng.normal(0.0, sigma, size=data.shape)
                noisy = data.astype(np.float64) + noise
                self._logger.info(
                    "Noising %s (sigma=%s) min/max before: %s/%s",
                    dataset_path,
                    sigma,
                    float(np.nanmin(data)),
                    float(np.nanmax(data)),
                )
                if np.issubdtype(dataset.dtype, np.integer):
                    info = np.iinfo(dataset.dtype)
                    noisy = np.clip(np.rint(noisy), info.min, info.max).astype(
                        dataset.dtype
                    )
                else:
                    noisy = noisy.astype(dataset.dtype, copy=False)
                dataset[...] = noisy
                self._logger.info(
                    "Noised %s min/max after: %s/%s",
                    dataset_path,
                    float(np.nanmin(noisy)),
                    float(np.nanmax(noisy)),
                )

    def _copy_hdf5(self, source: h5py.File, target: h5py.File) -> None:
        """Copy the entire HDF5 structure from source to target.

        Parameters:
            source: Source HDF5 file.
            target: Target HDF5 file.
        """

        for key in source.keys():
            source.copy(key, target)
        for key, value in source.attrs.items():
            target.attrs[key] = value

    def _find_dataset_path(self, source: h5py.File, field: str) -> str | None:
        """Locate a dataset path by name within the EBSD data group.

        Parameters:
            source: Source HDF5 file.
            field: Dataset name to locate.

        Returns:
            Dataset path or None if not found.
        """

        for key in source.keys():
            if key in {"Manufacturer", "Version"}:
                continue
            group = source[key]
            if not isinstance(group, h5py.Group):
                continue
            data_group = group.get("EBSD/Data")
            if data_group is None:
                continue
            if field in data_group:
                return data_group[field].name
        return None
