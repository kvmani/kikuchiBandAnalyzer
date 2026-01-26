"""Derived output field registry for Kikuchi band analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, Dict, Iterable, Optional

import h5py
import numpy as np

ArrayMap = Dict[str, np.ndarray]
ComputeFunc = Callable[[ArrayMap], np.ndarray]
PostprocessFunc = Callable[[ArrayMap, np.ndarray, logging.Logger], None]


@dataclass(frozen=True)
class DerivedFieldSpec:
    """Definition for a derived output field.

    Parameters:
        name: Canonical field key for registry lookups.
        dataset_name: Dataset name relative to the EBSD/Data group.
        dtype: NumPy dtype for the output array.
        dependencies: Names of upstream fields required for computation.
        compute: Pure function that computes the derived array.
        attrs: Optional HDF5 attributes to attach when writing.
        postprocess: Optional callback invoked after computation.
    """

    name: str
    dataset_name: str
    dtype: np.dtype
    dependencies: tuple[str, ...]
    compute: ComputeFunc
    attrs: Dict[str, str] = field(default_factory=dict)
    postprocess: Optional[PostprocessFunc] = None


class DerivedFieldRegistry:
    """Registry for derived output field specifications."""

    def __init__(
        self,
        fields: Iterable[DerivedFieldSpec],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the registry.

        Parameters:
            fields: Iterable of derived field specifications.
            logger: Optional logger for warnings.
        """

        self._fields = tuple(fields)
        self._logger = logger or logging.getLogger(__name__)
        self._specs = {field.name: field for field in self._fields}

    def fields(self) -> tuple[DerivedFieldSpec, ...]:
        """Return the registered field specifications.

        Returns:
            Tuple of derived field specifications.
        """

        return self._fields

    def get_spec(self, name: str) -> Optional[DerivedFieldSpec]:
        """Return a field specification by name.

        Parameters:
            name: Canonical field name.

        Returns:
            DerivedFieldSpec if present, otherwise None.
        """

        return self._specs.get(name)

    def compute(self, inputs: ArrayMap) -> ArrayMap:
        """Compute derived fields from the input arrays.

        Parameters:
            inputs: Mapping of available base fields to arrays.

        Returns:
            Mapping of derived field names to computed arrays.
        """

        available: ArrayMap = dict(inputs)
        outputs: ArrayMap = {}
        for spec in self._fields:
            missing = [dep for dep in spec.dependencies if dep not in available]
            if missing:
                self._logger.warning(
                    "Derived field '%s' skipped; missing dependencies: %s.",
                    spec.name,
                    ", ".join(missing),
                )
                continue
            result = spec.compute(available)
            result = np.asarray(result, dtype=spec.dtype)
            outputs[spec.name] = result
            available[spec.name] = result
            if spec.postprocess is not None:
                spec.postprocess(available, result, self._logger)
        return outputs


def write_hdf5_dataset(
    h5file: h5py.File,
    dataset_path: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, str]] = None,
) -> h5py.Dataset:
    """Create an HDF5 dataset and attach optional attributes.

    Parameters:
        h5file: Open HDF5 file handle.
        dataset_path: Full dataset path to create.
        data: Array data to write.
        attrs: Optional attributes to attach to the dataset.

    Returns:
        The created HDF5 dataset.
    """

    dataset = h5file.create_dataset(dataset_path, data=data)
    if attrs:
        for key, value in attrs.items():
            dataset.attrs[key] = value
    return dataset


def build_default_registry(
    logger: Optional[logging.Logger] = None,
    eps: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> DerivedFieldRegistry:
    """Build the default registry of derived fields.

    Parameters:
        logger: Optional logger for warnings.
        eps: Denominator threshold for normalized differences.
        dtype: Output dtype for derived arrays.

    Returns:
        DerivedFieldRegistry populated with default specs.
    """

    dtype = np.dtype(dtype)

    def compute_band_intensity_diff_norm(inputs: ArrayMap) -> np.ndarray:
        """Compute normalized intensity difference for Kikuchi bands.

        Parameters:
            inputs: Mapping with efficient and deficient intensity arrays.

        Returns:
            Normalized intensity difference array.
        """

        efficient = np.asarray(inputs["efficientlineIntensity"], dtype=np.float64)
        deficient = np.asarray(inputs["defficientlineIntensity"], dtype=np.float64)
        denom = efficient + deficient
        numerator = efficient - deficient
        result = np.full_like(denom, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(2.0 * numerator, denom, out=result, where=np.abs(denom) > eps)
        return result.astype(dtype, copy=False)

    def warn_small_denominator(
        inputs: ArrayMap, _: np.ndarray, logger: logging.Logger
    ) -> None:
        """Warn when near-zero denominators are encountered.

        Parameters:
            inputs: Mapping with intensity arrays.
            _: Computed output array (unused).
            logger: Logger for warnings.

        Returns:
            None.
        """

        efficient = np.asarray(inputs["efficientlineIntensity"], dtype=np.float64)
        deficient = np.asarray(inputs["defficientlineIntensity"], dtype=np.float64)
        denom = efficient + deficient
        finite = np.isfinite(denom)
        small = finite & (np.abs(denom) <= eps)
        count = int(np.count_nonzero(small))
        if count:
            logger.warning(
                "Derived field 'band_intensity_diff_norm' set %d entries to NaN "
                "because |I_eff + I_def| <= %s.",
                count,
                eps,
            )

    diff_norm_spec = DerivedFieldSpec(
        name="band_intensity_diff_norm",
        dataset_name="band_intensity_diff_norm",
        dtype=dtype,
        dependencies=("efficientlineIntensity", "defficientlineIntensity"),
        compute=compute_band_intensity_diff_norm,
        attrs={
            "description": "Normalized intensity difference between efficient and deficient bands.",
            "formula": "2*(I_eff - I_def)/(I_eff + I_def)",
        },
        postprocess=warn_small_denominator,
    )
    return DerivedFieldRegistry([diff_norm_spec], logger=logger)
