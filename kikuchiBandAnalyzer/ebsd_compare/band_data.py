"""Helpers for loading and validating exported band profile data.

The Kikuchi bandwidth pipeline stores per-pixel band profile vectors and
associated metadata in OH5/HDF5 outputs. This module provides a small,
testable extraction layer used by downstream GUIs such as the EBSD Comparator
and the Automator GUI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset


@dataclass(frozen=True)
class BandProfilePayload:
    """Container for band profile data at a single pixel.

    Parameters:
        profile: 1D band intensity profile vector, or None when unavailable.
        central_line: Optional 4-vector [x1, y1, x2, y2] in pattern pixel coordinates.
        band_start_idx: Optional left minima index into ``profile``.
        central_peak_idx: Optional peak index into ``profile``.
        band_end_idx: Optional right minima index into ``profile``.
        profile_length: Optional expected profile length (for validation).
        band_valid: True when profile and indices describe a valid band.
    """

    profile: Optional[np.ndarray]
    central_line: Optional[np.ndarray]
    band_start_idx: Optional[int]
    central_peak_idx: Optional[int]
    band_end_idx: Optional[int]
    profile_length: Optional[int]
    band_valid: bool


def _safe_int(value: float, *, field: str, logger: logging.Logger) -> Optional[int]:
    """Convert a scalar value to int if finite, otherwise return None.

    Parameters:
        value: Scalar value (often returned as float from ScanDataset).
        field: Field name for logging context.
        logger: Logger instance.

    Returns:
        Integer value when finite, otherwise None.
    """

    if value is None or not np.isfinite(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Failed to convert field %s=%r to int.", field, value)
        return None


def _read_scalar_int(
    scan: ScanDataset,
    field_name: str,
    x: int,
    y: int,
    *,
    logger: logging.Logger,
) -> Optional[int]:
    """Read an integer-like scalar field from a scan.

    Parameters:
        scan: ScanDataset to read from.
        field_name: Scalar dataset name.
        x: Column index.
        y: Row index.
        logger: Logger instance.

    Returns:
        Integer value, or None when field is missing/unreadable.
    """

    try:
        value = scan.get_scalar(field_name, x, y)
    except Exception as exc:
        logger.debug("Scalar field %s not available: %s", field_name, exc)
        return None
    return _safe_int(value, field=field_name, logger=logger)


def _read_vector(
    scan: ScanDataset,
    field_name: str,
    x: int,
    y: int,
    *,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """Read a vector field from a scan.

    Parameters:
        scan: ScanDataset to read from.
        field_name: Vector dataset name.
        x: Column index.
        y: Row index.
        logger: Logger instance.

    Returns:
        1D NumPy array, or None when unavailable.
    """

    try:
        vec = scan.get_vector(field_name, x, y)
    except Exception as exc:
        logger.debug("Vector field %s read failed: %s", field_name, exc)
        return None
    if vec is None:
        return None
    return np.asarray(vec)


def normalize_profile(
    profile: np.ndarray,
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "",
) -> np.ndarray:
    """Normalize a band profile by its maximum value.

    Parameters:
        profile: 1D profile array.
        logger: Optional logger for warnings.
        context: Additional context string for log messages.

    Returns:
        Normalized profile array when possible, otherwise the original profile.
    """

    logger = logger or logging.getLogger(__name__)
    if profile.size == 0:
        return profile
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        logger.warning("Profile normalization skipped (no finite values). %s", context)
        return profile
    max_val = float(np.max(finite))
    if not np.isfinite(max_val) or max_val == 0.0:
        logger.warning("Profile normalization skipped (max=%s). %s", max_val, context)
        return profile
    return profile / max_val


def extract_band_profile_payload(
    scan: ScanDataset,
    x: int,
    y: int,
    *,
    logger: Optional[logging.Logger] = None,
) -> BandProfilePayload:
    """Extract band profile data for a given scan and pixel.

    Parameters:
        scan: ScanDataset to extract from.
        x: Column index.
        y: Row index.
        logger: Optional logger instance.

    Returns:
        BandProfilePayload describing the profile availability and metadata.
    """

    logger = logger or logging.getLogger(__name__)
    profile = _read_vector(scan, "band_profile", x, y, logger=logger)
    central_line = _read_vector(scan, "central_line", x, y, logger=logger)
    band_start_idx = _read_scalar_int(scan, "band_start_idx", x, y, logger=logger)
    band_end_idx = _read_scalar_int(scan, "band_end_idx", x, y, logger=logger)
    central_peak_idx = _read_scalar_int(scan, "central_peak_idx", x, y, logger=logger)
    profile_length = _read_scalar_int(scan, "profile_length", x, y, logger=logger)
    band_valid_flag = _read_scalar_int(scan, "band_valid", x, y, logger=logger)

    if profile is None:
        return BandProfilePayload(
            profile=None,
            central_line=central_line,
            band_start_idx=band_start_idx,
            central_peak_idx=central_peak_idx,
            band_end_idx=band_end_idx,
            profile_length=profile_length,
            band_valid=False,
        )

    profile_arr = np.asarray(profile, dtype=np.float32).ravel()
    has_finite = bool(np.isfinite(profile_arr).any())
    indices_valid = (
        band_start_idx is not None
        and band_end_idx is not None
        and band_start_idx >= 0
        and band_end_idx >= 0
        and band_end_idx > band_start_idx
    )
    if band_valid_flag is not None:
        band_valid = bool(band_valid_flag)
    else:
        band_valid = has_finite and indices_valid

    if profile_length is not None and profile_length != int(profile_arr.size):
        logger.warning(
            "profile_length mismatch at X=%s Y=%s: expected=%s actual=%s",
            x,
            y,
            profile_length,
            profile_arr.size,
        )

    return BandProfilePayload(
        profile=profile_arr,
        central_line=np.asarray(central_line, dtype=np.float32).ravel() if central_line is not None else None,
        band_start_idx=band_start_idx if band_start_idx is not None and band_start_idx >= 0 else None,
        central_peak_idx=central_peak_idx if central_peak_idx is not None and central_peak_idx >= 0 else None,
        band_end_idx=band_end_idx if band_end_idx is not None and band_end_idx >= 0 else None,
        profile_length=profile_length,
        band_valid=band_valid,
    )

