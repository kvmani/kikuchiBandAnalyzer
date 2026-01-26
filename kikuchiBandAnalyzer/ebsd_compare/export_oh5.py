"""Export comparison results into a TSL-compatible OH5 file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
from typing import Iterable, Optional

import h5py
import numpy as np
import yaml

from kikuchiBandAnalyzer import __version__ as _repo_version
from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    AlignmentResult,
    alignment_result_to_dict,
)


@dataclass(frozen=True)
class Oh5ComparisonExportResult:
    """Summary of an OH5 comparison export operation.

    Parameters:
        output_path: Path to the written OH5 file.
        mode: Comparison mode ("delta", "abs_delta", or "ratio").
        exported_fields: Scalar field names that were overwritten with comparison data.
        skipped_fields: Scalar field names that were skipped (e.g., Phase or missing).
    """

    output_path: Path
    mode: str
    exported_fields: tuple[str, ...]
    skipped_fields: tuple[str, ...]


class Oh5ComparisonExporter:
    """Export aligned A/B comparisons into an OH5 file.

    The exporter copies scan A to a new output file and overwrites discovered scalar
    maps (excluding phase-like fields) with the comparison result computed in scan A
    coordinates. Alignment metadata and export context are stored under:

    `/<scan_name>/EBSD/Compare/`
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the exporter.

        Parameters:
            logger: Optional logger instance.
        """

        self._logger = logger or logging.getLogger(__name__)

    def export(
        self,
        scan_a: ScanDataset,
        scan_b: ScanDataset,
        engine: ComparisonEngine,
        output_path: Path,
        mode: str,
        alignment: Optional[AlignmentResult] = None,
        overwrite: bool = False,
        excluded_fields: Optional[Iterable[str]] = None,
    ) -> Oh5ComparisonExportResult:
        """Export the comparison OH5 file.

        Parameters:
            scan_a: Scan A dataset (used as the template OH5).
            scan_b: Scan B dataset (used for comparisons; alignment handled by engine).
            engine: Comparison engine configured with optional alignment.
            output_path: Destination OH5 path.
            mode: Comparison mode ("delta", "abs_delta", or "ratio").
            alignment: Optional alignment result to embed as metadata.
            overwrite: Whether to overwrite an existing output file.
            excluded_fields: Optional iterable of field names to skip.

        Returns:
            Oh5ComparisonExportResult with exported and skipped fields.
        """

        mode = str(mode).strip()
        if mode not in {"delta", "abs_delta", "ratio"}:
            raise ValueError(f"Unsupported compare mode '{mode}'.")
        output_path = Path(output_path)
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(output_path)
            output_path.unlink()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scan_a.file_path, output_path)
        self._logger.info("Copied scan A OH5 template to %s", output_path)

        excluded = {field.strip().lower() for field in (excluded_fields or [])}
        common_fields = sorted(
            set(scan_a.catalog.scalars.keys()) & set(scan_b.catalog.scalars.keys())
        )
        scan_name = scan_a.scan_name
        exported: list[str] = []
        skipped: list[str] = []

        with h5py.File(output_path, "r+") as handle:
            data_group_path = f"/{scan_name}/EBSD/Data"
            if data_group_path not in handle:
                raise KeyError(f"Missing data group '{data_group_path}' in output file.")
            for field in common_fields:
                normalized = field.strip().lower()
                if normalized in excluded or _is_phase_like_field(normalized):
                    skipped.append(field)
                    continue
                dataset_path = f"{data_group_path}/{field}"
                dataset = handle.get(dataset_path)
                if dataset is None or not isinstance(dataset, h5py.Dataset):
                    self._logger.warning(
                        "Field '%s' missing from scan-A template at %s; skipping.",
                        field,
                        dataset_path,
                    )
                    skipped.append(field)
                    continue
                if mode == "ratio" and dataset.dtype.kind not in {"f"}:
                    self._logger.warning(
                        "Field '%s' stored as %s cannot represent A/B ratios; skipping.",
                        field,
                        dataset.dtype,
                    )
                    skipped.append(field)
                    continue
                diff_map = engine.map_triplet(field, mode)["D"]
                payload = self._reshape_to_dataset(diff_map, dataset.shape)
                if payload is None:
                    self._logger.warning(
                        "Field '%s' comparison shape %s does not match dataset shape %s; skipping.",
                        field,
                        diff_map.shape,
                        dataset.shape,
                    )
                    skipped.append(field)
                    continue
                dataset[...] = payload.astype(dataset.dtype, copy=False)
                exported.append(field)

            self._write_metadata(
                handle,
                scan_name=scan_name,
                mode=mode,
                scan_a_path=scan_a.file_path,
                scan_b_path=scan_b.file_path,
                exported_fields=exported,
                skipped_fields=skipped,
                alignment=alignment,
            )

        self._logger.info(
            "Exported comparison OH5 with %d fields (%d skipped): %s",
            len(exported),
            len(skipped),
            output_path,
        )
        return Oh5ComparisonExportResult(
            output_path=output_path,
            mode=mode,
            exported_fields=tuple(exported),
            skipped_fields=tuple(skipped),
        )

    def _reshape_to_dataset(
        self, map_data: np.ndarray, target_shape: tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Reshape a 2D map into the target dataset shape.

        Parameters:
            map_data: Map array shaped (ny, nx).
            target_shape: Target dataset shape in the OH5 file.

        Returns:
            Array matching target_shape, or None if unsupported.
        """

        map_data = np.asarray(map_data)
        if len(target_shape) == 1:
            if map_data.ndim != 2:
                return None
            payload = np.ravel(map_data, order="C")
            if payload.shape != target_shape:
                return None
            return payload
        if len(target_shape) == 2:
            if map_data.shape != target_shape:
                return None
            return map_data
        return None

    def _write_metadata(
        self,
        handle: h5py.File,
        scan_name: str,
        mode: str,
        scan_a_path: Path,
        scan_b_path: Path,
        exported_fields: list[str],
        skipped_fields: list[str],
        alignment: Optional[AlignmentResult],
    ) -> None:
        """Write compare/alignment metadata into the output OH5.

        Parameters:
            handle: Open HDF5 output handle.
            scan_name: Scan group name.
            mode: Comparison mode.
            scan_a_path: Source scan A path.
            scan_b_path: Source scan B path.
            exported_fields: Exported scalar field names.
            skipped_fields: Skipped scalar field names.
            alignment: Optional alignment result.

        Returns:
            None.
        """

        compare_group = handle.require_group(f"/{scan_name}/EBSD/Compare")
        compare_group.attrs["kikuchiBandAnalyzer_version"] = _repo_version
        compare_group.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_string(compare_group, "mode", mode)
        self._write_string(compare_group, "scan_a_source", str(scan_a_path))
        self._write_string(compare_group, "scan_b_source", str(scan_b_path))
        self._write_string_list(compare_group, "exported_fields", exported_fields)
        self._write_string_list(compare_group, "skipped_fields", skipped_fields)

        alignment_group = compare_group.require_group("alignment")
        alignment_group.attrs["enabled"] = bool(alignment is not None)
        if alignment is None:
            return
        alignment_dict = alignment_result_to_dict(alignment)
        self._write_string(alignment_group, "alignment_yaml", yaml.safe_dump(alignment_dict, sort_keys=False))
        self._write_numeric(alignment_group, "rotation_deg", float(alignment.rotation_deg))
        self._write_numeric(alignment_group, "translation", np.asarray(alignment.translation, dtype=np.float64))
        self._write_numeric(alignment_group, "matrix", np.asarray(alignment.matrix, dtype=np.float64))
        rms_error = alignment.rms_error
        if rms_error is not None:
            self._write_numeric(alignment_group, "rms_error", float(rms_error))

    def _write_string(self, group: h5py.Group, name: str, value: str) -> None:
        """Write or replace a UTF-8 string dataset.

        Parameters:
            group: Destination HDF5 group.
            name: Dataset name.
            value: String value.

        Returns:
            None.
        """

        if name in group:
            del group[name]
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=np.asarray(value, dtype=dtype))

    def _write_string_list(self, group: h5py.Group, name: str, values: list[str]) -> None:
        """Write or replace a UTF-8 string list dataset.

        Parameters:
            group: Destination HDF5 group.
            name: Dataset name.
            values: List of string values.

        Returns:
            None.
        """

        if name in group:
            del group[name]
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=np.asarray(values, dtype=dtype))

    def _write_numeric(self, group: h5py.Group, name: str, value: np.ndarray | float) -> None:
        """Write or replace a numeric dataset.

        Parameters:
            group: Destination HDF5 group.
            name: Dataset name.
            value: Numeric payload.

        Returns:
            None.
        """

        if name in group:
            del group[name]
        group.create_dataset(name, data=value)


def _is_phase_like_field(normalized_name: str) -> bool:
    """Return True for phase/phase-id scalar fields.

    Parameters:
        normalized_name: Field name normalized to lowercase and stripped.

    Returns:
        True if field should be treated as phase-like.
    """

    if normalized_name == "phase":
        return True
    if normalized_name in {"phase_id", "phaseid", "phase id", "phases"}:
        return True
    return False
