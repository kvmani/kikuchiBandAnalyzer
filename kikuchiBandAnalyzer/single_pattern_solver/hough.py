"""Reusable Hough diagnostics for single-pattern EBSD indexing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

import kikuchipy as kp
import numpy as np
from orix.crystal_map import Phase, PhaseList


@dataclass(frozen=True)
class HoughPeak:
    """One Hough peak selected by the indexing backend.

    Parameters:
        rank: One-based rank sorted by decreasing peak intensity.
        intensity: Raw peak intensity from PyEBSDIndex.
        norm_intensity: Normalized peak intensity.
        theta: Band angle in radians.
        rho: Band distance in detector pixels.
        display_theta_deg: Hough display theta in degrees using PyEBSDIndex's
            own ``aveloc`` to Radon-axis mapping.
        display_rho: Hough display rho in pixels using PyEBSDIndex's own
            ``aveloc`` to Radon-axis mapping.
        maxloc: Integer-like peak location in the Hough image.
        avemax: Averaged peak intensity after local refinement.
        aveloc: Refined peak location in the Hough image.
        valid: Whether the backend marked this band as valid.
        line: Optional image-space line segment ``[x1, y1, x2, y2]`` fitted
            from the backend Radon ``indexPlan`` pixels.
    """

    rank: int
    intensity: float
    norm_intensity: float
    theta: float
    rho: float
    display_theta_deg: float
    display_rho: float
    maxloc: tuple[float, float]
    avemax: float
    aveloc: tuple[float, float]
    valid: bool
    line: Optional[list[float]]


@dataclass(frozen=True)
class HoughDiagnostic:
    """Hough transform diagnostic output for one EBSP pattern.

    Parameters:
        hough_image: Convolved Hough/Radon image used for peak detection.
        theta_axis: Hough theta axis in degrees.
        rho_axis: Hough rho axis in pixels.
        peaks: Ranked selected peaks.
        band_data: Raw structured band data returned by PyEBSDIndex.
    """

    hough_image: np.ndarray
    theta_axis: np.ndarray
    rho_axis: np.ndarray
    peaks: list[HoughPeak]
    band_data: np.ndarray


def extract_hough_diagnostic(
    pattern: np.ndarray,
    detector: kp.detectors.EBSDDetector,
    phase: Phase,
    hkl_list: list[list[int]],
    *,
    n_bands: int = 10,
    t_sigma: float = 2.0,
    r_sigma: float = 2.0,
    band_data_override: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> HoughDiagnostic:
    """Extract Hough image, selected peaks, and image-space lines.

    Parameters:
        pattern: Two-dimensional EBSP image.
        detector: Kikuchipy detector configured with PC and geometry.
        phase: Phase used to create the PyEBSDIndex indexer.
        hkl_list: Reflector families passed to the indexer.
        n_bands: Number of selected Hough peaks.
        t_sigma: Theta smoothing sigma used by the backend.
        r_sigma: Rho smoothing sigma used by the backend.
        band_data_override: Optional structured band data already returned by
            kikuchipy/PyEBSDIndex. When supplied, these selected peaks are used
            instead of running the lower-level band finder independently.
        logger: Optional logger.

    Returns:
        Hough diagnostic payload.
    """

    log = logger or logging.getLogger(__name__)
    image = np.asarray(pattern, dtype=np.float32)
    phase_list = PhaseList(phase)
    indexer = detector.get_indexer(
        phase_list,
        hkl_list,
        nBands=int(n_bands),
        tSigma=float(t_sigma),
        rSigma=float(r_sigma),
    )
    band_plan = indexer.bandDetectPlan
    radon_plan = band_plan.radonPlan
    patterns = image.reshape((1,) + image.shape)
    radon_norm = band_plan.radonPlan.radon_faster(
        patterns,
        band_plan.padding,
        fixArtifacts=False,
        background=band_plan.backgroundsub,
    )
    radon_conv, _ = band_plan.rdn_conv(radon_norm)
    if band_data_override is None:
        band_data = band_plan.find_bands(patterns, verbose=0)[0]
    else:
        band_data = np.asarray(band_data_override).reshape(-1)
    hough_image = _squeeze_hough_image(radon_conv, band_plan)
    peaks = _rank_hough_peaks(band_data, image.shape, radon_plan, log)
    return HoughDiagnostic(
        hough_image=hough_image,
        theta_axis=np.asarray(radon_plan.theta, dtype=np.float32),
        rho_axis=np.asarray(radon_plan.rho, dtype=np.float32),
        peaks=peaks,
        band_data=band_data,
    )


def hough_peak_to_line(
    theta: float,
    rho: float,
    shape: tuple[int, int],
) -> Optional[list[float]]:
    """Invert one Hough peak into an image-space line segment.

    The PyEBSDIndex ``theta`` and ``rho`` values are interpreted as a normal-form
    image line around the pattern center. The resulting long line is clipped to
    the image bounds for robust plotting.

    Parameters:
        theta: Hough display theta value in degrees.
        rho: Hough display rho value in pixels.
        shape: Pattern image shape ``(height, width)``.

    Returns:
        Clipped ``[x1, y1, x2, y2]`` segment or None if it does not cross image.
    """

    height, width = shape
    center = np.array([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float64)
    theta_rad = np.deg2rad(float(theta))
    direction = np.array([np.cos(theta_rad), -np.sin(theta_rad)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    point = center + float(rho) * normal
    length = float(max(width, height) * 3)
    start = point - length * direction
    end = point + length * direction
    return _clip_segment_to_bounds(
        [float(start[0]), float(start[1]), float(end[0]), float(end[1])],
        width=width,
        height=height,
    )


def _clip_segment_to_bounds(
    coords: list[float],
    *,
    width: int,
    height: int,
) -> Optional[list[float]]:
    """Clip a segment to image bounds with Liang-Barsky clipping.

    Parameters:
        coords: Segment coordinates ``[x1, y1, x2, y2]``.
        width: Image width.
        height: Image height.

    Returns:
        Clipped segment or None.
    """

    x1, y1, x2, y2 = [float(value) for value in coords]
    dx = x2 - x1
    dy = y2 - y1
    p_values = [-dx, dx, -dy, dy]
    q_values = [x1, float(width - 1) - x1, y1, float(height - 1) - y1]
    u1, u2 = 0.0, 1.0
    for p_value, q_value in zip(p_values, q_values):
        if p_value == 0:
            if q_value < 0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0:
            u1 = max(u1, ratio)
        else:
            u2 = min(u2, ratio)
        if u1 > u2:
            return None
    return [x1 + u1 * dx, y1 + u1 * dy, x1 + u2 * dx, y1 + u2 * dy]


def hough_diagnostic_to_json(diagnostic: Optional[HoughDiagnostic]) -> Optional[dict[str, Any]]:
    """Convert Hough diagnostics into a compact JSON-ready payload.

    Parameters:
        diagnostic: Optional Hough diagnostic object.

    Returns:
        JSON-ready mapping or None.
    """

    if diagnostic is None:
        return None
    return {
        "hough_shape": list(diagnostic.hough_image.shape),
        "theta_axis_deg": diagnostic.theta_axis.astype(float).tolist(),
        "rho_axis": diagnostic.rho_axis.astype(float).tolist(),
        "peaks": [
            {
                "rank": peak.rank,
                "intensity": peak.intensity,
                "norm_intensity": peak.norm_intensity,
                "theta": peak.theta,
                "rho": peak.rho,
                "display_theta_deg": peak.display_theta_deg,
                "display_rho": peak.display_rho,
                "maxloc": list(peak.maxloc),
                "aveloc": list(peak.aveloc),
                "valid": peak.valid,
                "line": peak.line,
            }
            for peak in diagnostic.peaks
        ],
    }


def _squeeze_hough_image(radon_conv: np.ndarray, band_plan: Any) -> np.ndarray:
    """Return a displayable 2D Hough image from backend output.

    Parameters:
        radon_conv: Convolved Radon image from PyEBSDIndex.
        band_plan: PyEBSDIndex band detection plan containing padding.

    Returns:
        Trimmed two-dimensional Hough image.
    """

    image = np.asarray(radon_conv, dtype=np.float32)
    if image.ndim == 3:
        image = image[:, :, 0]
    rho_pad, theta_pad = [int(value) for value in band_plan.padding]
    if rho_pad > 0:
        image = image[rho_pad:-rho_pad, :]
    if theta_pad > 0:
        image = image[:, theta_pad:-theta_pad]
    return np.asarray(image, dtype=np.float32)


def _rank_hough_peaks(
    band_data: np.ndarray,
    shape: tuple[int, int],
    radon_plan: Any,
    logger: logging.Logger,
) -> list[HoughPeak]:
    """Rank backend-selected Hough peaks by descending intensity.

    Parameters:
        band_data: Structured PyEBSDIndex band array for one pattern.
        shape: Pattern image shape.
        radon_plan: PyEBSDIndex Radon plan with theta/rho axes.
        logger: Logger for non-fatal inversion issues.

    Returns:
        Ranked peak descriptors.
    """

    order = np.argsort(np.asarray(band_data["max"], dtype=np.float64))[::-1]
    peaks: list[HoughPeak] = []
    for rank, index in enumerate(order, start=1):
        row = band_data[index]
        theta = float(row["theta"])
        rho = float(row["rho"])
        display_theta_deg = float(
            180.0
            - np.interp(
                float(row["aveloc"][1]) + 0.5,
                np.arange(radon_plan.nTheta),
                radon_plan.theta,
            )
        )
        display_rho = float(
            -np.interp(
                float(row["aveloc"][0]) - 0.5,
                np.arange(radon_plan.nRho),
                radon_plan.rho,
            )
        )
        try:
            line = _line_from_radon_index_plan(row, shape, radon_plan)
            if line is None:
                line = hough_peak_to_line(display_theta_deg, display_rho, shape)
        except Exception as exc:
            logger.debug("Failed to invert Hough peak rank %d: %s", rank, exc)
            line = None
        peaks.append(
            HoughPeak(
                rank=rank,
                intensity=float(row["max"]),
                norm_intensity=float(row["normmax"]),
                theta=theta,
                rho=rho,
                display_theta_deg=display_theta_deg,
                display_rho=display_rho,
                maxloc=tuple(float(value) for value in row["maxloc"]),
                avemax=float(row["avemax"]),
                aveloc=tuple(float(value) for value in row["aveloc"]),
                valid=bool(row["valid"]),
                line=line,
            )
        )
    return peaks


def _line_from_radon_index_plan(
    band_row: np.void,
    shape: tuple[int, int],
    radon_plan: Any,
) -> Optional[list[float]]:
    """Fit an EBSP line from PyEBSDIndex's exact Radon pixel trace.

    PyEBSDIndex builds the Radon transform through an ``indexPlan`` lookup. For
    diagnostic overlays, using the same lookup avoids ambiguity in theta/rho
    sign conventions and display transforms.

    Parameters:
        band_row: One structured selected-band row.
        shape: Pattern image shape.
        radon_plan: PyEBSDIndex Radon plan containing ``indexPlan``.

    Returns:
        Clipped line segment or None when the lookup is unavailable.
    """

    if not hasattr(radon_plan, "indexPlan"):
        return None
    height, width = shape
    rho_index = int(round(float(band_row["aveloc"][0])))
    theta_index = int(round(float(band_row["aveloc"][1])))
    rho_index = min(max(rho_index, 0), int(radon_plan.nRho) - 1)
    theta_index = min(max(theta_index, 0), int(radon_plan.nTheta) - 1)
    flat_indices = np.asarray(radon_plan.indexPlan[rho_index, theta_index], dtype=np.int64)
    flat_indices = flat_indices[(flat_indices >= 0) & (flat_indices < height * width)]
    if flat_indices.size < 2:
        return None
    y_values = (flat_indices // width).astype(np.float64)
    x_values = (flat_indices % width).astype(np.float64)
    points = np.column_stack([x_values, y_values])
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    start = centroid + projections.min() * direction
    end = centroid + projections.max() * direction
    return _clip_segment_to_bounds(
        [float(start[0]), float(start[1]), float(end[0]), float(end[1])],
        width=width,
        height=height,
    )
