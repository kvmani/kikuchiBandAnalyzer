"""Alignment utilities for EBSD scan registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, warp


@dataclass(frozen=True)
class AlignmentSettings:
    """Configuration values for alignment and resampling.

    Parameters:
        min_point_pairs: Minimum number of point pairs required.
        ransac_min_samples: Minimum samples for RANSAC estimation.
        ransac_residual_threshold: Residual threshold for inlier detection.
        ransac_max_trials: Maximum RANSAC iterations.
        warp_order: Interpolation order for map warping.
        warp_mode: Padding mode for map warping.
        warp_cval: Fill value for constant padding.
        preserve_range: Preserve original data range when warping.
        pattern_sampling: Sampling strategy for pattern lookups.
    """

    min_point_pairs: int
    ransac_min_samples: int
    ransac_residual_threshold: float
    ransac_max_trials: int
    warp_order: int
    warp_mode: str
    warp_cval: float
    preserve_range: bool
    pattern_sampling: str


@dataclass
class AlignmentResult:
    """Result of an alignment estimation.

    Parameters:
        transform: Euclidean transform mapping scan B to scan A.
        inliers: Boolean mask for inlier point pairs.
        rms_error: RMS error for inlier pairs.
        points_a: Point coordinates in scan A.
        points_b: Point coordinates in scan B.
        residuals: Per-point residual distances in scan A coordinates.
    """

    transform: EuclideanTransform
    inliers: Optional[np.ndarray]
    rms_error: Optional[float]
    points_a: Optional[np.ndarray]
    points_b: Optional[np.ndarray]
    residuals: Optional[np.ndarray]

    @property
    def rotation_deg(self) -> float:
        """Return the rotation in degrees.

        Returns:
            Rotation in degrees.
        """

        return float(np.rad2deg(self.transform.rotation))

    @property
    def translation(self) -> Tuple[float, float]:
        """Return the translation as (tx, ty).

        Returns:
            Translation tuple (tx, ty).
        """

        translation = self.transform.translation
        return float(translation[0]), float(translation[1])

    @property
    def matrix(self) -> np.ndarray:
        """Return the homogeneous transform matrix.

        Returns:
            3x3 homogeneous transform matrix.
        """

        return self.transform.params.copy()


def alignment_settings_from_config(
    alignment_config: Optional[Dict[str, Any]]
) -> AlignmentSettings:
    """Build AlignmentSettings from configuration.

    Parameters:
        alignment_config: Alignment configuration dictionary.

    Returns:
        AlignmentSettings with defaults applied.
    """

    alignment_config = alignment_config or {}
    ransac_config = alignment_config.get("ransac", {})
    warp_config = alignment_config.get("warp", {})
    map_interpolation = alignment_config.get("map_interpolation")
    warp_order = int(warp_config.get("order", 1))
    if isinstance(map_interpolation, str):
        interpolation_map = {
            "nearest": 0,
            "bilinear": 1,
            "bicubic": 3,
        }
        warp_order = interpolation_map.get(map_interpolation.lower(), warp_order)
    cval = warp_config.get("cval", np.nan)
    warp_cval = float(cval) if cval is not None else float("nan")
    return AlignmentSettings(
        min_point_pairs=int(alignment_config.get("point_pairs_min", 3)),
        ransac_min_samples=int(ransac_config.get("min_samples", 3)),
        ransac_residual_threshold=float(ransac_config.get("residual_threshold", 2.0)),
        ransac_max_trials=int(ransac_config.get("max_trials", 2000)),
        warp_order=warp_order,
        warp_mode=str(warp_config.get("mode", "constant")),
        warp_cval=warp_cval,
        preserve_range=bool(warp_config.get("preserve_range", True)),
        pattern_sampling=str(alignment_config.get("pattern_sampling", "nearest")),
    )


def compute_residuals(
    points_a: np.ndarray, points_b: np.ndarray, transform: EuclideanTransform
) -> np.ndarray:
    """Compute residual distances between aligned points.

    Parameters:
        points_a: Point coordinates in scan A.
        points_b: Point coordinates in scan B.
        transform: Transform mapping scan B to scan A.

    Returns:
        1D array of residual distances per point.
    """

    predicted = transform(points_b)
    return np.linalg.norm(predicted - points_a, axis=1)


def estimate_alignment(
    points_a: np.ndarray,
    points_b: np.ndarray,
    settings: AlignmentSettings,
    logger: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """Estimate alignment between two point sets using RANSAC.

    Parameters:
        points_a: Point coordinates in scan A.
        points_b: Point coordinates in scan B.
        settings: Alignment settings for RANSAC.
        logger: Optional logger instance.

    Returns:
        AlignmentResult with transform and diagnostics.
    """

    logger = logger or logging.getLogger(__name__)
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    if points_a.shape != points_b.shape:
        raise ValueError("Point arrays must share the same shape.")
    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError("Point arrays must have shape (N, 2).")
    if len(points_a) < settings.min_point_pairs:
        raise ValueError(
            f"Need at least {settings.min_point_pairs} point pairs for alignment."
        )
    min_samples = min(settings.ransac_min_samples, len(points_a))
    logger.info(
        "Estimating alignment with %s point pairs (min_samples=%s).",
        len(points_a),
        min_samples,
    )
    model_robust, inliers = ransac(
        (points_b, points_a),
        EuclideanTransform,
        min_samples=min_samples,
        residual_threshold=settings.ransac_residual_threshold,
        max_trials=settings.ransac_max_trials,
    )
    if inliers is None or not np.any(inliers):
        raise ValueError("RANSAC failed to find inliers for alignment.")
    residuals = compute_residuals(points_a, points_b, model_robust)
    rms_error = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    logger.info(
        "Alignment estimated: rotation=%.3f deg, translation=(%.3f, %.3f), rms=%.4f",
        float(np.rad2deg(model_robust.rotation)),
        float(model_robust.translation[0]),
        float(model_robust.translation[1]),
        rms_error,
    )
    return AlignmentResult(
        transform=model_robust,
        inliers=inliers,
        rms_error=rms_error,
        points_a=points_a,
        points_b=points_b,
        residuals=residuals,
    )


def build_alignment_from_parameters(
    rotation_deg: float, translation: Sequence[float]
) -> AlignmentResult:
    """Build an AlignmentResult from explicit rotation and translation.

    Parameters:
        rotation_deg: Rotation in degrees.
        translation: Translation vector (tx, ty).

    Returns:
        AlignmentResult with the configured transform.
    """

    transform = EuclideanTransform(
        rotation=np.deg2rad(float(rotation_deg)),
        translation=(float(translation[0]), float(translation[1])),
    )
    return AlignmentResult(
        transform=transform,
        inliers=None,
        rms_error=None,
        points_a=None,
        points_b=None,
        residuals=None,
    )


def alignment_result_to_dict(result: AlignmentResult) -> Dict[str, Any]:
    """Serialize an AlignmentResult to a dictionary.

    Parameters:
        result: AlignmentResult instance.

    Returns:
        Dictionary representation of the alignment.
    """

    return {
        "rotation_deg": result.rotation_deg,
        "translation": list(result.translation),
        "matrix": result.matrix.tolist(),
        "points_a": result.points_a.tolist() if result.points_a is not None else None,
        "points_b": result.points_b.tolist() if result.points_b is not None else None,
        "inliers": result.inliers.tolist() if result.inliers is not None else None,
        "rms_error": result.rms_error,
        "residuals": result.residuals.tolist() if result.residuals is not None else None,
    }


def alignment_result_from_dict(data: Dict[str, Any]) -> AlignmentResult:
    """Deserialize an AlignmentResult from a dictionary.

    Parameters:
        data: Serialized alignment dictionary.

    Returns:
        AlignmentResult instance.
    """

    if "rotation_deg" in data and "translation" in data:
        result = build_alignment_from_parameters(
            float(data["rotation_deg"]), data["translation"]
        )
    elif "matrix" in data:
        matrix = np.array(data["matrix"], dtype=float)
        transform = EuclideanTransform(matrix=matrix)
        result = AlignmentResult(
            transform=transform,
            inliers=None,
            rms_error=None,
            points_a=None,
            points_b=None,
            residuals=None,
        )
    else:
        raise ValueError("Alignment data must include rotation/translation or matrix.")
    if data.get("points_a") is not None:
        result.points_a = np.asarray(data["points_a"], dtype=float)
    if data.get("points_b") is not None:
        result.points_b = np.asarray(data["points_b"], dtype=float)
    if data.get("inliers") is not None:
        result.inliers = np.asarray(data["inliers"], dtype=bool)
    if data.get("rms_error") is not None:
        result.rms_error = float(data["rms_error"])
    if data.get("residuals") is not None:
        result.residuals = np.asarray(data["residuals"], dtype=float)
    return result


def save_alignment_to_yaml(output_path: Path, result: AlignmentResult) -> None:
    """Save alignment results to a YAML file.

    Parameters:
        output_path: Output YAML path.
        result: AlignmentResult to serialize.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(alignment_result_to_dict(result), handle, sort_keys=False)


def load_alignment_from_yaml(path: Path) -> AlignmentResult:
    """Load an alignment result from a YAML file.

    Parameters:
        path: Path to the YAML file.

    Returns:
        AlignmentResult instance.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return alignment_result_from_dict(data)


def alignment_from_config(
    alignment_config: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Optional[AlignmentResult]:
    """Build an AlignmentResult from configuration.

    Parameters:
        alignment_config: Alignment configuration dictionary.
        logger: Optional logger instance.

    Returns:
        AlignmentResult if configured, otherwise None.
    """

    if not alignment_config:
        return None
    logger = logger or logging.getLogger(__name__)
    enabled = bool(alignment_config.get("enabled", True))
    if not enabled:
        return None
    precomputed_path = alignment_config.get("precomputed_alignment_path")
    if precomputed_path:
        logger.info("Loading precomputed alignment from %s", precomputed_path)
        return load_alignment_from_yaml(Path(precomputed_path))
    control_points = alignment_config.get("control_points", {})
    points_a = control_points.get("points_a")
    points_b = control_points.get("points_b")
    if points_a and points_b:
        settings = alignment_settings_from_config(alignment_config)
        return estimate_alignment(
            np.asarray(points_a, dtype=float),
            np.asarray(points_b, dtype=float),
            settings,
            logger=logger,
        )
    transform_config = alignment_config.get("transform", {})
    if transform_config:
        rotation_deg = float(transform_config.get("rotation_deg", 0.0))
        translation = transform_config.get("translation", [0.0, 0.0])
        logger.info(
            "Using configured alignment rotation=%.3f deg translation=%s",
            rotation_deg,
            translation,
        )
        return build_alignment_from_parameters(rotation_deg, translation)
    return None


def align_map(
    data: np.ndarray,
    transform: EuclideanTransform,
    output_shape: Tuple[int, int],
    settings: AlignmentSettings,
) -> np.ndarray:
    """Warp a map from scan B into scan A coordinates.

    Parameters:
        data: Map array from scan B.
        transform: Transform mapping scan B to scan A.
        output_shape: Output shape (ny, nx).
        settings: Alignment settings for warping.

    Returns:
        Warped map aligned to scan A coordinates.
    """

    return warp(
        data,
        inverse_map=transform.inverse,
        output_shape=output_shape,
        order=settings.warp_order,
        mode=settings.warp_mode,
        cval=settings.warp_cval,
        preserve_range=settings.preserve_range,
    )
