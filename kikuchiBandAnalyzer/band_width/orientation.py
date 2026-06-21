"""Runtime orientation selection for Kikuchi-line simulation.

This module deliberately separates orientations used for simulation from the
acquisition Euler arrays stored in EBSD files. Indexed orientations are kept in
memory and failed pixels may fall back to acquisition orientations without
modifying source metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import kikuchipy as kp
import numpy as np
from orix.crystal_map import PhaseList
from orix.quaternion import Rotation


@dataclass(frozen=True)
class OrientationSelection:
    """Runtime rotations and per-pixel indexing diagnostics.

    Parameters:
        source: Normalized requested source, ``indexed`` or ``acquisition``.
        rotations: Rotations selected for line simulation.
        indexing_success: ``1`` for successful indexing, ``0`` for failure,
            and ``-1`` when indexing was not requested.
        orientation_fallback: ``1`` where indexed mode fell back to acquisition.
        source_used: ``1`` where indexed rotations are used and ``0`` where
            acquisition rotations are used.
        fit: PyEBSDIndex fit values, or NaN when unavailable.
        confidence: PyEBSDIndex confidence metric, or NaN when unavailable.
    """

    source: str
    rotations: Rotation
    indexing_success: np.ndarray
    orientation_fallback: np.ndarray
    source_used: np.ndarray
    fit: np.ndarray
    confidence: np.ndarray


def normalize_orientation_source(value: object) -> str:
    """Normalize a configured orientation source.

    Parameters:
        value: User/configuration value.

    Returns:
        ``indexed`` or ``acquisition``.

    Raises:
        ValueError: If the value is unsupported.
    """

    normalized = str(value or "indexed").strip().lower()
    aliases = {
        "indexed": "indexed",
        "live": "indexed",
        "hough": "indexed",
        "acquisition": "acquisition",
        "original": "acquisition",
        "input": "acquisition",
    }
    if normalized not in aliases:
        raise ValueError(
            "orientation_source must be 'indexed' or 'acquisition' "
            f"(legacy 'original' is accepted); got {value!r}."
        )
    return aliases[normalized]


def select_runtime_orientations(
    pattern_data: np.ndarray,
    acquisition_eulers_rad: np.ndarray,
    detector: Any,
    phase_list: PhaseList,
    hkl_list: list[list[int]],
    *,
    source: object = "indexed",
    direction: str = "lab2crystal",
    hough: dict[str, Any] | None = None,
    signal: Any | None = None,
    logger: logging.Logger | None = None,
) -> OrientationSelection:
    """Select acquisition or indexed rotations for pattern simulation.

    Parameters:
        pattern_data: Pattern stack shaped ``(ny, nx, height, width)``.
        acquisition_eulers_rad: Acquisition Euler array shaped ``(ny*nx, 3)``.
        detector: Configured kikuchipy EBSD detector.
        phase_list: Phase list supplied to PyEBSDIndex.
        hkl_list: Reflector families supplied to PyEBSDIndex.
        source: Requested source, ``indexed`` or ``acquisition``.
        direction: Euler direction used for acquisition rotations.
        hough: Optional Hough indexing settings.
        signal: Optional existing kikuchipy EBSD signal.
        logger: Optional logger.

    Returns:
        Runtime orientation selection with diagnostic arrays.
    """

    log = logger or logging.getLogger(__name__)
    data = np.asarray(pattern_data)
    if data.ndim != 4:
        raise ValueError(f"Pattern data must have shape (ny, nx, h, w); got {data.shape}.")
    navigation_shape = data.shape[:2]
    n_pixels = int(np.prod(navigation_shape))
    eulers = np.asarray(acquisition_eulers_rad, dtype=np.float64)
    if eulers.shape != (n_pixels, 3):
        raise ValueError(
            "Acquisition Euler count does not match pattern grid: "
            f"Euler shape={eulers.shape}, expected=({n_pixels}, 3)."
        )
    requested = normalize_orientation_source(source)
    acquisition = Rotation.from_euler(eulers, direction=direction, degrees=False)
    unavailable = np.full(n_pixels, np.nan, dtype=np.float32)
    if requested == "acquisition":
        return OrientationSelection(
            source=requested,
            rotations=acquisition.reshape(*navigation_shape),
            indexing_success=np.full(n_pixels, -1, dtype=np.int8),
            orientation_fallback=np.zeros(n_pixels, dtype=np.int8),
            source_used=np.zeros(n_pixels, dtype=np.int8),
            fit=unavailable.copy(),
            confidence=unavailable.copy(),
        )

    hough_cfg = dict(hough or {})
    try:
        ebsd_signal = (
            signal
            if signal is not None and hasattr(signal, "hough_indexing")
            else kp.signals.EBSD(data)
        )
        indexer = detector.get_indexer(
            phase_list,
            hkl_list,
            nBands=int(hough_cfg.get("n_bands", hough_cfg.get("nBands", 10))),
            tSigma=float(hough_cfg.get("t_sigma", hough_cfg.get("tSigma", 2.0))),
            rSigma=float(hough_cfg.get("r_sigma", hough_cfg.get("rSigma", 2.0))),
        )
        xmap, index_data, _ = ebsd_signal.hough_indexing(
            phase_list=phase_list,
            indexer=indexer,
            return_index_data=True,
            return_band_data=True,
            verbose=int(hough_cfg.get("verbose", 1)),
        )
    except Exception:
        log.exception(
            "Live Hough indexing failed for the scan; using acquisition-Euler "
            "fallback for all %d pixels.",
            n_pixels,
        )
        return OrientationSelection(
            source=requested,
            rotations=acquisition.reshape(*navigation_shape),
            indexing_success=np.zeros(n_pixels, dtype=np.int8),
            orientation_fallback=np.ones(n_pixels, dtype=np.int8),
            source_used=np.zeros(n_pixels, dtype=np.int8),
            fit=unavailable.copy(),
            confidence=unavailable.copy(),
        )
    success = np.asarray(xmap.is_indexed, dtype=bool).reshape(-1)
    indexed_data = np.asarray(xmap.rotations.data, dtype=np.float64).reshape(n_pixels, 4)
    acquisition_data = np.asarray(acquisition.data, dtype=np.float64).reshape(n_pixels, 4)
    finite_rotation = np.isfinite(indexed_data).all(axis=1) & (np.linalg.norm(indexed_data, axis=1) > 0)
    success &= finite_rotation
    selected_data = indexed_data.copy()
    selected_data[~success] = acquisition_data[~success]

    fit = unavailable.copy()
    confidence = unavailable.copy()
    index_array = np.asarray(index_data)
    if index_array.dtype.names and index_array.ndim >= 2 and index_array.shape[-1] == n_pixels:
        phase_result = index_array[0]
        if "fit" in index_array.dtype.names:
            fit = np.asarray(phase_result["fit"], dtype=np.float32).reshape(-1)
        if "cm" in index_array.dtype.names:
            confidence = np.asarray(phase_result["cm"], dtype=np.float32).reshape(-1)
    fit[~success] = np.nan
    confidence[~success] = np.nan
    fallback = (~success).astype(np.int8)
    log.info(
        "Live Hough indexing selected %d/%d pixels; %d pixels use acquisition-Euler fallback.",
        int(success.sum()),
        n_pixels,
        int(fallback.sum()),
    )
    return OrientationSelection(
        source=requested,
        rotations=Rotation(selected_data).reshape(*navigation_shape),
        indexing_success=success.astype(np.int8),
        orientation_fallback=fallback,
        source_used=success.astype(np.int8),
        fit=fit,
        confidence=confidence,
    )
