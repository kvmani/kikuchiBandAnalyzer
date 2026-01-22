"""Synthetic scan generation for debug workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage.transform import EuclideanTransform, warp

from kikuchiBandAnalyzer.ebsd_compare.model import FieldCatalog, FieldRef, ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.readers.base import ScanFileReader


class InMemoryScanReader(ScanFileReader):
    """In-memory reader for synthetic scan data."""

    def __init__(
        self,
        maps: Dict[str, np.ndarray],
        patterns: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Initialize the in-memory reader.

        Parameters:
            maps: Mapping of scalar field names to 2D arrays.
            patterns: Mapping of pattern field names to arrays.
        """

        self._maps = maps
        self._patterns = patterns or {}
        self._catalog = self._build_catalog()

    def catalog(self) -> FieldCatalog:
        """Return the in-memory field catalog.

        Returns:
            FieldCatalog describing available fields.
        """

        return self._catalog

    def get_map(self, field_name: str) -> np.ndarray:
        """Return the map for the requested scalar field.

        Parameters:
            field_name: Name of the scalar field.

        Returns:
            2D NumPy array.
        """

        return np.array(self._maps[field_name], copy=False)

    def get_scalar(self, field_name: str, x: int, y: int) -> float:
        """Return a scalar value at the specified coordinate.

        Parameters:
            field_name: Name of the scalar field.
            x: Column index.
            y: Row index.

        Returns:
            Scalar value at the specified location.
        """

        return float(self._maps[field_name][y, x])

    def get_pattern(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return a pattern image at the specified coordinate.

        Parameters:
            field_name: Name of the pattern field.
            x: Column index.
            y: Row index.

        Returns:
            Pattern array or None if unavailable.
        """

        if field_name not in self._patterns:
            return None
        patterns = self._patterns[field_name]
        if patterns.ndim == 4:
            return np.array(patterns[y, x], copy=False)
        if patterns.ndim == 3:
            ny, nx = self._maps[next(iter(self._maps))].shape
            index = y * nx + x
            return np.array(patterns[index], copy=False)
        raise ValueError(f"Unsupported pattern array shape {patterns.shape}.")

    def close(self) -> None:
        """Release any held resources.

        Returns:
            None.
        """

        return None

    def _build_catalog(self) -> FieldCatalog:
        """Build the field catalog from in-memory arrays.

        Returns:
            FieldCatalog with scalar and pattern references.
        """

        scalars = {}
        for name, array in self._maps.items():
            scalars[name] = FieldRef(
                name=name,
                path=f"memory:{name}",
                kind="scalar",
                shape=tuple(array.shape),
                dtype=str(array.dtype),
            )
        patterns = {}
        for name, array in self._patterns.items():
            patterns[name] = FieldRef(
                name=name,
                path=f"memory:{name}",
                kind="pattern",
                shape=tuple(array.shape),
                dtype=str(array.dtype),
            )
        return FieldCatalog(scalars=scalars, patterns=patterns)


@dataclass(frozen=True)
class SimulatedScanConfig:
    """Configuration for simulated scan generation.

    Parameters:
        nx: Number of columns for scan A.
        ny: Number of rows for scan A.
        nx_b: Number of columns for scan B.
        ny_b: Number of rows for scan B.
        rotation_deg: Rotation applied to scan B relative to scan A.
        translation: Translation applied to scan B relative to scan A.
        noise_sigma: Noise standard deviation per scalar field.
        seed: Random seed for reproducible noise.
        include_patterns: Whether to generate pattern stacks.
        pattern_height: Height of synthetic patterns.
        pattern_width: Width of synthetic patterns.
    """

    nx: int
    ny: int
    nx_b: int
    ny_b: int
    rotation_deg: float
    translation: Tuple[float, float]
    noise_sigma: Dict[str, float]
    seed: int
    include_patterns: bool
    pattern_height: int
    pattern_width: int


class SimulatedScanFactory:
    """Factory for generating synthetic scan pairs."""

    def __init__(self, config: SimulatedScanConfig, logger: logging.Logger) -> None:
        """Initialize the factory.

        Parameters:
            config: Simulation configuration.
            logger: Logger instance.
        """

        self._config = config
        self._logger = logger

    @classmethod
    def from_config(cls, config: Dict[str, Any], logger: logging.Logger) -> "SimulatedScanFactory":
        """Create a factory from configuration values.

        Parameters:
            config: Configuration dictionary.
            logger: Logger instance.

        Returns:
            SimulatedScanFactory instance.
        """

        nx = int(config.get("nx", 24))
        ny = int(config.get("ny", 24))
        nx_b = int(config.get("nx_b", nx))
        ny_b = int(config.get("ny_b", ny))
        rotation_deg = float(config.get("rotation_deg", 5.0))
        translation = config.get("translation", [2.0, -1.0])
        noise_sigma = config.get("noise_sigma", {"IQ": 0.02, "CI": 0.01})
        seed = int(config.get("seed", 123))
        include_patterns = bool(config.get("include_patterns", False))
        pattern_height = int(config.get("pattern_height", 32))
        pattern_width = int(config.get("pattern_width", 32))
        sim_config = SimulatedScanConfig(
            nx=nx,
            ny=ny,
            nx_b=nx_b,
            ny_b=ny_b,
            rotation_deg=rotation_deg,
            translation=(float(translation[0]), float(translation[1])),
            noise_sigma=noise_sigma,
            seed=seed,
            include_patterns=include_patterns,
            pattern_height=pattern_height,
            pattern_width=pattern_width,
        )
        return cls(sim_config, logger)

    def create_pair(self) -> Tuple[ScanDataset, ScanDataset]:
        """Create a synthetic scan pair.

        Returns:
            Tuple of (scan_a, scan_b) datasets.
        """

        rng = np.random.default_rng(self._config.seed)
        self._logger.info(
            "Generating simulated scans nx=%s ny=%s -> nx_b=%s ny_b=%s",
            self._config.nx,
            self._config.ny,
            self._config.nx_b,
            self._config.ny_b,
        )
        grid_x = np.linspace(-1.0, 1.0, self._config.nx)
        grid_y = np.linspace(-1.0, 1.0, self._config.ny)
        xx, yy = np.meshgrid(grid_x, grid_y)
        iq_map = np.exp(-(xx**2 + yy**2) / 0.4) + 0.2 * xx
        ci_map = 0.5 + 0.5 * np.sin(3.0 * xx) * np.cos(3.0 * yy)
        maps_a = {"IQ": iq_map.astype(np.float32), "CI": ci_map.astype(np.float32)}
        transform = EuclideanTransform(
            rotation=np.deg2rad(self._config.rotation_deg),
            translation=self._config.translation,
        )
        maps_b = {}
        for name, data in maps_a.items():
            warped = warp(
                data,
                inverse_map=transform.inverse,
                output_shape=(self._config.ny_b, self._config.nx_b),
                order=1,
                mode="constant",
                cval=np.nan,
                preserve_range=True,
            )
            sigma = float(self._config.noise_sigma.get(name, 0.0))
            if sigma > 0:
                warped = warped + rng.normal(0.0, sigma, size=warped.shape)
            maps_b[name] = warped.astype(np.float32)
        patterns_a: Optional[Dict[str, np.ndarray]] = None
        patterns_b: Optional[Dict[str, np.ndarray]] = None
        if self._config.include_patterns:
            pattern = self._create_pattern(
                self._config.pattern_height, self._config.pattern_width
            )
            patterns_a = {
                "Pattern": self._tile_pattern(
                    pattern, self._config.ny, self._config.nx, rng
                )
            }
            patterns_b = {
                "Pattern": self._tile_pattern(
                    pattern, self._config.ny_b, self._config.nx_b, rng
                )
            }
        scan_a = self._build_dataset("SimulatedA", maps_a, patterns_a)
        scan_b = self._build_dataset("SimulatedB", maps_b, patterns_b)
        return scan_a, scan_b

    def _build_dataset(
        self,
        scan_name: str,
        maps: Dict[str, np.ndarray],
        patterns: Optional[Dict[str, np.ndarray]],
    ) -> ScanDataset:
        """Build a ScanDataset from in-memory arrays.

        Parameters:
            scan_name: Name of the simulated scan.
            maps: Scalar maps for the scan.
            patterns: Optional pattern stacks.

        Returns:
            ScanDataset instance.
        """

        reader = InMemoryScanReader(maps=maps, patterns=patterns)
        catalog = reader.catalog()
        ny, nx = next(iter(maps.values())).shape
        return ScanDataset(
            file_path=Path("memory"),
            scan_name=scan_name,
            nx=nx,
            ny=ny,
            catalog=catalog,
            reader=reader,
        )

    def _create_pattern(self, height: int, width: int) -> np.ndarray:
        """Create a synthetic diffraction pattern image.

        Parameters:
            height: Pattern height.
            width: Pattern width.

        Returns:
            2D NumPy array for the pattern.
        """

        grid_x = np.linspace(-1.0, 1.0, width)
        grid_y = np.linspace(-1.0, 1.0, height)
        xx, yy = np.meshgrid(grid_x, grid_y)
        pattern = np.exp(-(xx**2 + yy**2) / 0.2)
        return pattern.astype(np.float32)

    def _tile_pattern(
        self, pattern: np.ndarray, ny: int, nx: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Tile a pattern across the scan grid with mild noise.

        Parameters:
            pattern: Base pattern image.
            ny: Number of rows.
            nx: Number of columns.
            rng: Random generator for noise.

        Returns:
            4D pattern stack with shape (ny, nx, height, width).
        """

        stack = np.tile(pattern[None, None, :, :], (ny, nx, 1, 1))
        noise = rng.normal(0.0, 0.01, size=stack.shape)
        return (stack + noise).astype(np.float32)
