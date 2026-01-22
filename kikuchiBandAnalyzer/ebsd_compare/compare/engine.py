"""Comparison engine for EBSD scan datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.compare import ops
from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    AlignmentResult,
    alignment_settings_from_config,
    align_map,
)


@dataclass
class ProbeResult:
    """Container for probe results at a single pixel.

    Parameters:
        fields: Mapping of field name to comparison values.
        x: Column index.
        y: Row index.
    """

    fields: Dict[str, Dict[str, float]]
    x: int
    y: int


class ComparisonEngine:
    """Engine for comparing two EBSD scans.

    Parameters:
        scan_a: ScanDataset for scan A.
        scan_b: ScanDataset for scan B.
        config: Configuration dictionary.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        scan_a: ScanDataset,
        scan_b: ScanDataset,
        config: Dict,
        logger: Optional[logging.Logger] = None,
        alignment: Optional[AlignmentResult] = None,
    ) -> None:
        """Initialize the comparison engine.

        Parameters:
            scan_a: ScanDataset for scan A.
            scan_b: ScanDataset for scan B.
            config: Configuration dictionary.
            logger: Optional logger instance.
        """

        self._logger = logger or logging.getLogger(__name__)
        self._scan_a = scan_a
        self._scan_b = scan_b
        self._config = config
        self._alignment = alignment
        self._alignment_settings = alignment_settings_from_config(
            config.get("alignment", {})
        )
        self._aligned_map_cache: Dict[str, np.ndarray] = {}
        self._validate_shapes()

    def available_scalar_fields(self) -> list[str]:
        """Return the scalar fields available in both scans.

        Returns:
            List of scalar field names.
        """

        fields_a = set(self._scan_a.catalog.list_scalar_fields())
        fields_b = set(self._scan_b.catalog.list_scalar_fields())
        return sorted(fields_a & fields_b)

    def set_alignment(self, alignment: Optional[AlignmentResult]) -> None:
        """Set or clear the alignment transform.

        Parameters:
            alignment: AlignmentResult to apply, or None to disable.

        Returns:
            None.
        """

        self._alignment = alignment
        self._aligned_map_cache.clear()

    def default_probe_xy(self) -> Tuple[int, int]:
        """Return the default probe coordinate (middle pixel).

        Returns:
            Tuple of (x, y) indices.
        """

        return self._scan_a.nx // 2, self._scan_a.ny // 2

    def default_map_field(self) -> str:
        """Return the default map field based on configuration.

        Returns:
            Field name for the default map.
        """

        preferred = self._config.get("default_map_field")
        if preferred and preferred in self.available_scalar_fields():
            return preferred
        fields = self.available_scalar_fields()
        if not fields:
            raise ValueError("No common scalar fields available for comparison.")
        return fields[0]

    def map_triplet(self, field_name: str, mode: str) -> Dict[str, np.ndarray]:
        """Return map triplet (A, B, diff) for a scalar field.

        Parameters:
            field_name: Scalar field name.
            mode: Difference mode ("delta", "abs_delta", "ratio").

        Returns:
            Dictionary with keys "A", "B", "D".
        """

        map_a = self._scan_a.get_map(field_name)
        map_b = self._aligned_map(field_name)
        diff = self._diff_array(map_a, map_b, mode)
        return {"A": map_a, "B": map_b, "D": diff}

    def probe_scalars(
        self, x: int, y: int, fields: Iterable[str]
    ) -> ProbeResult:
        """Probe scalar values at a coordinate.

        Parameters:
            x: Column index.
            y: Row index.
            fields: Scalar fields to probe.

        Returns:
            ProbeResult with values for each field.
        """

        results: Dict[str, Dict[str, float]] = {}
        for field in fields:
            value_a = float(self._scan_a.get_scalar(field, x, y))
            map_b = self._aligned_map(field)
            value_b = float(map_b[y, x])
            delta_value = value_a - value_b
            ratio_value = value_a / value_b if value_b != 0 else np.nan
            results[field] = {
                "A": value_a,
                "B": value_b,
                "Delta": delta_value,
                "Ratio": ratio_value,
            }
        return ProbeResult(fields=results, x=x, y=y)

    def probe_patterns(
        self, x: int, y: int, fields: Iterable[str], mode: str
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """Probe pattern images at a coordinate.

        Parameters:
            x: Column index.
            y: Row index.
            fields: Pattern field names to probe.
            mode: Difference mode ("delta", "abs_delta", "ratio").

        Returns:
            Mapping of field name to pattern triplets.
        """

        results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for field in fields:
            pattern_a = self._scan_a.get_pattern(field, x, y)
            pattern_b = self._aligned_pattern(field, x, y)
            if pattern_a is None or pattern_b is None:
                results[field] = {"A": pattern_a, "B": pattern_b, "D": None}
                continue
            diff = self._diff_array(pattern_a, pattern_b, mode)
            results[field] = {"A": pattern_a, "B": pattern_b, "D": diff}
        return results

    def _validate_shapes(self) -> None:
        """Validate that both scans have identical grid shapes."""

        if self._scan_a.nx != self._scan_b.nx or self._scan_a.ny != self._scan_b.ny:
            if self._alignment is None:
                raise ValueError(
                    "Scan grids do not match; alignment required before comparison."
                )
            self._logger.info(
                "Scan grids differ (A=%s x %s, B=%s x %s); using alignment.",
                self._scan_a.nx,
                self._scan_a.ny,
                self._scan_b.nx,
                self._scan_b.ny,
            )

    def _diff_array(self, a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
        """Compute the diff array for two inputs.

        Parameters:
            a: First array.
            b: Second array.
            mode: Difference mode ("delta", "abs_delta", "ratio").

        Returns:
            Diff array.
        """

        if mode == "delta":
            return ops.delta(a, b)
        if mode == "abs_delta":
            return ops.abs_delta(a, b)
        if mode == "ratio":
            return ops.ratio(a, b)
        raise ValueError(f"Unsupported diff mode '{mode}'.")

    def _aligned_map(self, field_name: str) -> np.ndarray:
        """Return the aligned map for scan B (or raw map if no alignment).

        Parameters:
            field_name: Scalar field name.

        Returns:
            2D NumPy array aligned to scan A coordinates.
        """

        if self._alignment is None:
            return self._scan_b.get_map(field_name)
        if field_name in self._aligned_map_cache:
            return self._aligned_map_cache[field_name]
        data = self._scan_b.get_map(field_name)
        aligned = align_map(
            data,
            self._alignment.transform,
            output_shape=(self._scan_a.ny, self._scan_a.nx),
            settings=self._alignment_settings,
        )
        self._aligned_map_cache[field_name] = aligned
        return aligned

    def _aligned_pattern(self, field_name: str, x: int, y: int) -> Optional[np.ndarray]:
        """Return the aligned pattern for scan B at the given scan A coordinate.

        Parameters:
            field_name: Pattern field name.
            x: Column index in scan A.
            y: Row index in scan A.

        Returns:
            Pattern array or None if outside scan B bounds.
        """

        if self._alignment is None:
            return self._scan_b.get_pattern(field_name, x, y)
        inverse_point = self._alignment.transform.inverse(
            np.array([[float(x), float(y)]])
        )[0]
        bx = float(inverse_point[0])
        by = float(inverse_point[1])
        if (
            bx < 0
            or by < 0
            or bx > self._scan_b.nx - 1
            or by > self._scan_b.ny - 1
        ):
            return None
        if self._alignment_settings.pattern_sampling != "nearest":
            raise ValueError(
                f"Unsupported pattern sampling '{self._alignment_settings.pattern_sampling}'."
            )
        bx_idx = int(round(bx))
        by_idx = int(round(by))
        if (
            bx_idx < 0
            or by_idx < 0
            or bx_idx >= self._scan_b.nx
            or by_idx >= self._scan_b.ny
        ):
            return None
        return self._scan_b.get_pattern(field_name, bx_idx, by_idx)
