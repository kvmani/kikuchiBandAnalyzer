"""YAML-driven single-pattern EBSP simulation and band-profile solving."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Optional

import kikuchipy as kp
import matplotlib
import numpy as np
import yaml
from diffsims.crystallography import ReciprocalLatticeVector
from diffpy.structure import Atom, Lattice, Structure
from matplotlib.figure import Figure
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Rotation
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates

from kikuchiBandAnalyzer.ebsd_compare.band_data import BandProfilePayload, normalize_profile
from kikuchiBandAnalyzer.ebsd_compare.readers.ctf_reader import CtfPatternScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader
from kikuchiBandAnalyzer.single_pattern_solver.hough import (
    HoughDiagnostic,
    extract_hough_diagnostic,
    hough_diagnostic_to_json,
)
from simulators import CustomKikuchiPatternSimulator
from strategies import RectangularAreaBandDetector
import utilities as ut

matplotlib.use("Agg")


@dataclass(frozen=True)
class SinglePatternConfig:
    """Configuration for solving one EBSP pattern.

    Parameters:
        path: Source YAML path.
        raw: Parsed YAML mapping.
    """

    path: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class SinglePatternSolution:
    """Result of a single-pattern EBSP solve.

    Parameters:
        config: Source configuration.
        pattern: Experimental pattern image.
        x: Scan column index.
        y: Scan row index.
        eulers_rad: Euler angles in radians.
        lines: Simulated principal-family Kikuchi lines.
        detector_summary: Detector values passed to kikuchipy.
        profile_payload: Optional detected band-profile payload.
        selected_band: Optional raw detector result for the chosen line.
        hough_summary: Optional Hough/indexing diagnostic summary.
        hough_diagnostic: Optional Hough transform diagnostic.
    """

    config: SinglePatternConfig
    pattern: np.ndarray
    x: int
    y: int
    eulers_rad: np.ndarray
    lines: list[dict[str, Any]]
    detector_summary: dict[str, Any]
    profile_payload: Optional[BandProfilePayload]
    selected_band: Optional[dict[str, Any]]
    hough_summary: Optional[dict[str, Any]]
    hough_diagnostic: Optional[HoughDiagnostic]


def load_single_pattern_config(path: Path | str) -> SinglePatternConfig:
    """Load a single-pattern YAML configuration.

    Parameters:
        path: YAML configuration path.

    Returns:
        Parsed configuration container.
    """

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Single-pattern config must be a mapping: {config_path}")
    return SinglePatternConfig(path=config_path, raw=raw)


def solve_single_pattern(
    config: SinglePatternConfig,
    *,
    logger: Optional[logging.Logger] = None,
) -> SinglePatternSolution:
    """Load one pattern, simulate Kikuchi lines, and detect one target profile.

    Parameters:
        config: Single-pattern configuration.
        logger: Optional logger.

    Returns:
        Single-pattern solution.
    """

    log = logger or logging.getLogger(__name__)
    pattern, x, y, eulers_rad = _load_pattern_and_eulers(config)
    phase = _build_phase(config.raw["phase"])
    detector, detector_summary = _build_detector(config, pattern.shape)
    hough_summary, hough_diagnostic = _try_hough_summary(pattern, detector, phase, config, log)
    if hough_summary and hough_summary.get("success") and hough_summary.get("use_indexed_orientation"):
        eulers_rad = np.asarray(hough_summary["indexed_eulers_rad"], dtype=np.float64)
        log.info("Using kikuchipy Hough-indexed orientation for simulated overlay.")
    rotations = Rotation.from_euler(
        eulers_rad.reshape(1, 3),
        direction=str(config.raw.get("orientation", {}).get("direction", "lab2crystal")),
        degrees=False,
    ).reshape(1, 1)
    reflectors = ReciprocalLatticeVector(
        phase=phase,
        hkl=_hkl_list(config),
    ).symmetrise()
    simulation = CustomKikuchiPatternSimulator(reflectors).on_detector(detector, rotations)
    simulation.phase = phase
    lines = _extract_principal_lines(simulation, phase, config)
    selected_band, profile_payload = _detect_one_band(pattern, lines, phase, config, log)
    log.info(
        "Solved single pattern x=%d y=%d with %d simulated principal-family lines.",
        x,
        y,
        len(lines),
    )
    return SinglePatternSolution(
        config=config,
        pattern=pattern,
        x=x,
        y=y,
        eulers_rad=eulers_rad,
        lines=lines,
        detector_summary=detector_summary,
        profile_payload=profile_payload,
        selected_band=selected_band,
        hough_summary=hough_summary,
        hough_diagnostic=hough_diagnostic,
    )


def write_solution_json(solution: SinglePatternSolution, path: Path | str) -> None:
    """Write the single-pattern solution metadata to JSON.

    Parameters:
        solution: Solution to serialize.
        path: Destination JSON path.

    Returns:
        None.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_config": str(solution.config.path),
        "pixel": {"x": solution.x, "y": solution.y},
        "pattern_shape": list(solution.pattern.shape),
        "eulers_rad": solution.eulers_rad.astype(float).tolist(),
        "eulers_deg": np.rad2deg(solution.eulers_rad).astype(float).tolist(),
        "detector": solution.detector_summary,
        "simulated_lines": solution.lines,
        "selected_band": _json_ready(solution.selected_band),
        "hough_summary": _json_ready(solution.hough_summary),
        "hough_diagnostic": hough_diagnostic_to_json(solution.hough_diagnostic),
    }
    if solution.profile_payload is not None:
        payload["band_profile"] = {
            "profile": solution.profile_payload.profile.astype(float).tolist()
            if solution.profile_payload.profile is not None
            else None,
            "central_line": solution.profile_payload.central_line.astype(float).tolist()
            if solution.profile_payload.central_line is not None
            else None,
            "band_start_idx": solution.profile_payload.band_start_idx,
            "central_peak_idx": solution.profile_payload.central_peak_idx,
            "band_end_idx": solution.profile_payload.band_end_idx,
            "profile_length": solution.profile_payload.profile_length,
            "band_valid": solution.profile_payload.band_valid,
        }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_solution(
    solution: SinglePatternSolution,
    path: Path | str,
    *,
    show_hough: bool = False,
) -> None:
    """Render an experimental EBSP overlay and profile proof image.

    Parameters:
        solution: Solution to render.
        path: Destination PNG path.
        show_hough: Reserved for future Hough overlays.

    Returns:
        None.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(10.5, 5.0), dpi=140)
    axes_pattern = figure.add_subplot(1, 2, 1)
    axes_profile = figure.add_subplot(1, 2, 2)
    axes_pattern.imshow(solution.pattern, cmap="gray")
    axes_pattern.set_title("Experimental EBSP + simulated Kikuchi lines")
    axes_pattern.set_xticks([])
    axes_pattern.set_yticks([])
    _draw_lines_on_axes(axes_pattern, solution.pattern.shape, solution.lines)
    _draw_profile_on_axes(axes_profile, solution.profile_payload)
    figure.tight_layout()
    figure.savefig(destination)


def _load_pattern_and_eulers(config: SinglePatternConfig) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Load the configured source pattern and Euler angles.

    Parameters:
        config: Single-pattern configuration.

    Returns:
        Pattern image, x, y, and Euler angles in radians.
    """

    source = dict(config.raw.get("input", {}))
    source_type = str(source.get("type", "")).lower()
    x = int(source.get("x", 0))
    y = int(source.get("y", 0))
    if source_type == "ctf":
        reader = CtfPatternScanFileReader(
            ctf_path=Path(source["ctf_path"]),
            pattern_dir=Path(source["pattern_dir"]),
            pattern_template=source.get("pattern_template"),
        )
        pattern = reader.get_pattern("Pattern", x, y)
        eulers = np.array(
            [
                reader.get_scalar("Euler1", x, y),
                reader.get_scalar("Euler2", x, y),
                reader.get_scalar("Euler3", x, y),
            ],
            dtype=np.float64,
        )
        reader.close()
        return _require_pattern(pattern), x, y, np.deg2rad(eulers)
    if source_type in {"oh5", "h5", "hdf5"}:
        dataset = OH5ScanFileReader.from_path(Path(source["path"]))
        pattern_field = str(source.get("pattern_field", "Pattern"))
        pattern = dataset.get_pattern(pattern_field, x, y)
        if all(field in dataset.catalog.scalars for field in ("Phi1", "Phi", "Phi2")):
            eulers = np.array(
                [
                    dataset.get_scalar("Phi1", x, y),
                    dataset.get_scalar("Phi", x, y),
                    dataset.get_scalar("Phi2", x, y),
                ],
                dtype=np.float64,
            )
        else:
            eulers = np.deg2rad(
                np.array(
                    [
                        dataset.get_scalar("Euler1", x, y),
                        dataset.get_scalar("Euler2", x, y),
                        dataset.get_scalar("Euler3", x, y),
                    ],
                    dtype=np.float64,
                )
            )
        dataset.close()
        return _require_pattern(pattern), x, y, eulers
    if source_type in {"image", "pattern", "single"}:
        with Image.open(Path(source["path"])) as image:
            pattern = np.asarray(image.convert("L"), dtype=np.float32)
        eulers_deg = source.get("eulers_deg", [0.0, 0.0, 0.0])
        eulers = np.deg2rad(np.asarray(eulers_deg, dtype=np.float64))
        if eulers.shape != (3,):
            raise ValueError("input.eulers_deg must contain exactly three Euler angles.")
        return _require_pattern(pattern), x, y, eulers
    raise ValueError("input.type must be 'ctf', 'oh5', or 'image'.")


def _require_pattern(pattern: Optional[np.ndarray]) -> np.ndarray:
    """Validate and return a pattern image.

    Parameters:
        pattern: Pattern image or None.

    Returns:
        2D float32 pattern image.
    """

    if pattern is None:
        raise ValueError("Pattern image is unavailable for the configured pixel.")
    image = np.asarray(pattern, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Pattern must be 2D, got shape {image.shape}.")
    return image


def _build_phase(phase_cfg: dict[str, Any]) -> Phase:
    """Build an orix phase from YAML phase settings.

    Parameters:
        phase_cfg: Phase configuration mapping.

    Returns:
        Orix phase.
    """

    atoms_cfg = phase_cfg.get("atoms") or [
        {"element": phase_cfg.get("name", "Ni"), "position": [0, 0, 0]}
    ]
    atoms = [Atom(str(atom["element"]), atom.get("position", [0, 0, 0])) for atom in atoms_cfg]
    return Phase(
        name=str(phase_cfg.get("name", "Ni")),
        space_group=int(phase_cfg.get("space_group", 225)),
        structure=Structure(
            lattice=Lattice(*[float(value) for value in phase_cfg["lattice"]]),
            atoms=atoms,
        ),
    )


def _build_detector(
    config: SinglePatternConfig,
    pattern_shape: tuple[int, int],
) -> tuple[kp.detectors.EBSDDetector, dict[str, Any]]:
    """Build a kikuchipy detector from YAML settings.

    Parameters:
        config: Single-pattern configuration.
        pattern_shape: Experimental pattern shape.

    Returns:
        Detector and serializable summary.
    """

    detector_cfg = dict(config.raw.get("detector", {}))
    pc = [float(value) for value in detector_cfg.get("pc", [0.5, 0.5, 0.5])]
    convention = str(detector_cfg.get("convention", "edax"))
    detector = kp.detectors.EBSDDetector(
        shape=tuple(int(value) for value in pattern_shape),
        px_size=float(detector_cfg.get("px_size", 1.0)),
        binning=int(detector_cfg.get("binning", 1)),
        sample_tilt=float(detector_cfg.get("sample_tilt", 70.0)),
        tilt=float(detector_cfg.get("tilt", 0.0)),
        azimuthal=float(detector_cfg.get("azimuthal", 0.0)),
        convention=convention,
        pc=tuple(pc),
    )
    summary = {
        "shape": list(pattern_shape),
        "pc": pc,
        "convention": convention,
        "sample_tilt": float(detector_cfg.get("sample_tilt", 70.0)),
        "tilt": float(detector_cfg.get("tilt", 0.0)),
        "azimuthal": float(detector_cfg.get("azimuthal", 0.0)),
        "px_size": float(detector_cfg.get("px_size", 1.0)),
        "binning": int(detector_cfg.get("binning", 1)),
    }
    return detector, summary


def _hkl_list(config: SinglePatternConfig) -> list[list[int]]:
    """Return configured HKL reflectors.

    Parameters:
        config: Single-pattern configuration.

    Returns:
        List of HKL triplets.
    """

    return [[int(value) for value in row] for row in config.raw.get("simulation", {}).get("hkl_list", [])]


def _extract_principal_lines(
    simulation: Any,
    phase: Phase,
    config: SinglePatternConfig,
) -> list[dict[str, Any]]:
    """Extract principal-family simulated line segments for one pattern.

    Parameters:
        simulation: kikuchipy geometrical simulation.
        phase: Crystal phase.
        config: Solver config.

    Returns:
        List of line dictionaries.
    """

    coords = np.asarray(simulation.lines_coordinates(index=(), exclude_nan=False), dtype=np.float64)
    coords = np.around(coords.reshape(-1, 4), 3)
    reflectors = simulation._reflectors.coordinates.round().astype(int)
    families = _principal_families(config, phase)
    height, width = [float(value) for value in simulation.detector.shape]
    center = np.array([0.5 * width, 0.5 * height])
    lines: list[dict[str, Any]] = []
    for index, line in enumerate(coords):
        if not np.isfinite(line).all():
            continue
        hkl = " ".join(str(int(value)) for value in reflectors[index])
        family = _family_for_hkl(hkl, families)
        if family is None:
            continue
        clipped = clip_segment_to_bounds(line, width=int(width), height=int(height))
        if clipped is None:
            continue
        midpoint = np.array([0.5 * (clipped[0] + clipped[2]), 0.5 * (clipped[1] + clipped[3])])
        lines.append(
            {
                "hkl": family["label"],
                "reflector": hkl,
                "family_hkl": family["hkl"],
                "color": family["color"],
                "central_line": [float(value) for value in clipped],
                "raw_line": [float(value) for value in line],
                "line_mid_xy": midpoint.astype(float).tolist(),
                "line_dist": float(np.linalg.norm(midpoint - center)),
            }
        )
    return sorted(lines, key=lambda item: float(item["line_dist"]))


def _principal_families(config: SinglePatternConfig, phase: Phase) -> list[dict[str, Any]]:
    """Build principal HKL family display settings.

    Parameters:
        config: Solver config.
        phase: Crystal phase.

    Returns:
        Family dictionaries.
    """

    colors = config.raw.get("display", {}).get(
        "family_colors",
        ["#00e676", "#ffdd33", "#40c4ff", "#ff6d00", "#e040fb", "#ffffff"],
    )
    families = []
    for index, hkl in enumerate(_hkl_list(config)):
        families.append(
            {
                "hkl": ",".join(str(int(value)) for value in hkl),
                "label": "{" + "".join(str(abs(int(value))) for value in hkl) + "}",
                "color": str(colors[index % len(colors)]),
                "phase": phase,
            }
        )
    return families


def _family_for_hkl(hkl: str, families: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the configured family matching one reflector.

    Parameters:
        hkl: Reflector string.
        families: Family dictionaries.

    Returns:
        Matching family or None.
    """

    for family in families:
        try:
            belongs, _ = ut.belongs_to_group(hkl, family["hkl"], phase=family["phase"])
        except Exception:
            belongs = False
        if belongs:
            return family
    return None


def _detect_one_band(
    pattern: np.ndarray,
    lines: list[dict[str, Any]],
    phase: Phase,
    config: SinglePatternConfig,
    logger: logging.Logger,
) -> tuple[Optional[dict[str, Any]], Optional[BandProfilePayload]]:
    """Detect a band profile for one configured desired family.

    Parameters:
        pattern: Experimental pattern.
        lines: Simulated lines sorted by center distance.
        phase: Crystal phase.
        config: Solver config.
        logger: Logger instance.

    Returns:
        Raw selected band result and profile payload.
    """

    profile_cfg = dict(config.raw.get("band_profile", {}))
    desired_hkl = str(profile_cfg.get("desired_hkl", config.raw.get("desired_hkl", "1,1,1")))
    detector_config = {
        "phase_list": config.raw["phase"],
        "desired_hkl": desired_hkl,
        "rectWidth": int(profile_cfg.get("rectWidth", 20)),
        "smoothing_sigma": float(profile_cfg.get("smoothing_sigma", 2.0)),
        "min_psnr": float(profile_cfg.get("min_psnr", 1.01)),
        "debug": bool(profile_cfg.get("debug", False)),
        "plot_band_detection": False,
        "plot_band_detection_condition": "False",
    }
    for line in lines:
        try:
            belongs, _ = ut.belongs_to_group(line["reflector"], desired_hkl, phase=phase)
        except Exception:
            belongs = False
        if not belongs:
            continue
        try:
            detector = RectangularAreaBandDetector(
                pattern,
                line["central_line"],
                detector_config,
                line["reflector"],
            )
            result = detector.detect()
        except Exception as exc:
            logger.info(
                "OpenCV band detector failed for %s; using SciPy single-pattern profile fallback: %s",
                line["reflector"],
                exc,
            )
            result = detect_profile_without_opencv(pattern, line, detector_config)
        result.update(
            {
                "hkl": line["reflector"],
                "hkl_group": line["hkl"],
                "line_mid_xy": line["line_mid_xy"],
                "line_dist": line["line_dist"],
            }
        )
        profile = np.asarray(result.get("band_profile"), dtype=np.float32)
        central_line = np.asarray(result.get("central_line"), dtype=np.float32)
        payload = BandProfilePayload(
            profile=profile,
            central_line=central_line,
            band_start_idx=int(result.get("band_start_idx", -1))
            if int(result.get("band_start_idx", -1)) >= 0
            else None,
            central_peak_idx=int(result.get("central_peak_idx", -1))
            if int(result.get("central_peak_idx", -1)) >= 0
            else None,
            band_end_idx=int(result.get("band_end_idx", -1))
            if int(result.get("band_end_idx", -1)) >= 0
            else None,
            profile_length=int(result.get("profile_length", profile.size)),
            band_valid=bool(result.get("band_valid", False)),
        )
        return result, payload
    return None, None


def detect_profile_without_opencv(
    pattern: np.ndarray,
    line: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Sample a band-width profile without OpenCV.

    The fallback samples a strip centered on the simulated line and sums
    intensity along the line direction, producing a profile across the band
    normal. This is intended for single-pattern geometry debugging when the
    production OpenCV detector is unavailable.

    Parameters:
        pattern: Experimental pattern.
        line: Simulated line dictionary.
        config: Detection configuration.

    Returns:
        Band detection result dictionary.
    """

    x1, y1, x2, y2 = [float(value) for value in line["central_line"]]
    direction = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 0:
        raise ValueError("Cannot sample profile for zero-length line.")
    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    midpoint = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float64)
    rect_width = int(config.get("rectWidth", 20))
    profile_length = int(rect_width * 4)
    offsets = np.linspace(-2.0 * rect_width, 2.0 * rect_width, profile_length)
    along = np.linspace(-0.45 * length, 0.45 * length, max(32, int(length // 2)))
    samples = []
    for offset in offsets:
        points = midpoint + offset * normal + along[:, None] * direction
        values = map_coordinates(
            pattern,
            [points[:, 1], points[:, 0]],
            order=1,
            mode="nearest",
        )
        samples.append(float(np.nanmean(values)))
    profile = np.asarray(samples, dtype=np.float32)
    smoothed = gaussian_filter1d(profile, sigma=float(config.get("smoothing_sigma", 2.0)))
    central_peak_idx = int(np.argmax(smoothed))
    if 0 < central_peak_idx < smoothed.size - 1:
        band_start_idx = int(np.argmin(smoothed[:central_peak_idx]))
        band_end_idx = int(np.argmin(smoothed[central_peak_idx:]) + central_peak_idx)
    else:
        band_start_idx = -1
        band_end_idx = -1
    if band_start_idx >= 0 and band_end_idx > band_start_idx:
        background = 0.5 * (smoothed[band_start_idx] + smoothed[band_end_idx])
        psnr = float(smoothed[central_peak_idx] / background) if background != 0 else 0.0
        band_width = float(band_end_idx - band_start_idx)
        band_valid = bool(psnr > float(config.get("min_psnr", 1.01)))
    else:
        psnr = 0.0
        band_width = 0.0
        band_valid = False
    return {
        "central_line": [x1, y1, x2, y2],
        "band_profile": profile.astype(float).tolist(),
        "band_start_idx": band_start_idx,
        "central_peak_idx": central_peak_idx,
        "band_end_idx": band_end_idx,
        "profile_length": profile_length,
        "bandStart": band_start_idx,
        "centralPeak": central_peak_idx,
        "bandEnd": band_end_idx,
        "bandWidth": band_width,
        "band_valid": band_valid,
        "psnr": psnr,
        "efficientlineIntensity": float(np.nanmax(profile)),
        "defficientlineIntensity": float(np.nanmin(profile)),
    }


def _try_hough_summary(
    pattern: np.ndarray,
    detector: kp.detectors.EBSDDetector,
    phase: Phase,
    config: SinglePatternConfig,
    logger: logging.Logger,
) -> tuple[Optional[dict[str, Any]], Optional[HoughDiagnostic]]:
    """Try kikuchipy Hough indexing for one pattern and summarize diagnostics.

    Parameters:
        pattern: Experimental pattern.
        detector: Detector.
        phase: Crystal phase.
        config: Solver config.
        logger: Logger.

    Returns:
        Hough diagnostic summary and optional transform details.
    """

    hough_cfg = dict(config.raw.get("hough", {}))
    if not bool(hough_cfg.get("enabled", False)):
        return None, None
    try:
        signal = kp.signals.EBSD(pattern.reshape((1, 1) + pattern.shape))
        phase_list = PhaseList(phase)
        indexer = detector.get_indexer(
            phase_list,
            _hkl_list(config),
            nBands=int(hough_cfg.get("n_bands", 10)),
            tSigma=float(hough_cfg.get("t_sigma", 2)),
            rSigma=float(hough_cfg.get("r_sigma", 2)),
        )
        xmap, _, band_data = signal.hough_indexing(
            phase_list=phase_list,
            indexer=indexer,
            return_index_data=True,
            return_band_data=True,
            verbose=0,
        )
        diagnostic = extract_hough_diagnostic(
            pattern,
            detector,
            phase,
            _hkl_list(config),
            n_bands=int(hough_cfg.get("n_bands", 10)),
            t_sigma=float(hough_cfg.get("t_sigma", 2)),
            r_sigma=float(hough_cfg.get("r_sigma", 2)),
            band_data_override=band_data,
            logger=logger,
        )
        return {
            "success": True,
            "xmap_shape": list(xmap.shape),
            "band_data_type": type(band_data).__name__,
            "fit": _json_ready(getattr(xmap, "fit", None)),
            "phase_id": _json_ready(getattr(xmap, "phase_id", None)),
            "indexed_eulers_rad": np.asarray(xmap.rotations.to_euler()).reshape(-1, 3)[0].tolist(),
            "indexed_eulers_deg": np.rad2deg(
                np.asarray(xmap.rotations.to_euler()).reshape(-1, 3)[0]
            ).tolist(),
            "use_indexed_orientation": bool(hough_cfg.get("use_indexed_orientation", False)),
        }, diagnostic
    except Exception as exc:
        logger.warning("Single-pattern Hough indexing failed: %s", exc)
        return {"success": False, "error": str(exc)}, None


def clip_segment_to_bounds(
    coords: np.ndarray | list[float],
    *,
    width: int,
    height: int,
) -> Optional[list[float]]:
    """Clip a segment to image bounds with Liang-Barsky clipping.

    Parameters:
        coords: ``x1, y1, x2, y2`` segment.
        width: Image width.
        height: Image height.

    Returns:
        Clipped segment or None if outside.
    """

    x1, y1, x2, y2 = [float(value) for value in coords]
    dx = x2 - x1
    dy = y2 - y1
    bounds = (0.0, float(width - 1), 0.0, float(height - 1))
    p_values = [-dx, dx, -dy, dy]
    q_values = [x1 - bounds[0], bounds[1] - x1, y1 - bounds[2], bounds[3] - y1]
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


def image_line_text_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return an upright annotation angle parallel to an image line.

    Parameters:
        x1: Line start X coordinate.
        y1: Line start Y coordinate.
        x2: Line end X coordinate.
        y2: Line end Y coordinate.

    Returns:
        Display rotation angle in degrees within ``[-90, 90]``.
    """

    angle = -float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _draw_lines_on_axes(axes: Any, shape: tuple[int, int], lines: list[dict[str, Any]]) -> None:
    """Draw clipped lines and sparse angle-matched labels on axes.

    Parameters:
        axes: Matplotlib axes.
        shape: Pattern shape.
        lines: Line dictionaries.

    Returns:
        None.
    """

    height, width = shape
    counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        coords = clip_segment_to_bounds(line["central_line"], width=width, height=height)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        color = str(line.get("color", "#00e676"))
        axes.plot([x1, x2], [y1, y2], color=color, linewidth=1.4, alpha=0.92, clip_on=True)
        family = str(line.get("hkl", ""))
        count = counts.get(family, 0)
        if count < 2:
            fraction = 0.16 if (count + index) % 2 == 0 else 0.84
            label_x = x1 + fraction * (x2 - x1)
            label_y = y1 + fraction * (y2 - y1)
            angle = image_line_text_angle(x1, y1, x2, y2)
            axes.text(
                label_x,
                label_y,
                family,
                color=color,
                rotation=angle,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.0},
                clip_on=True,
            )
        counts[family] = count + 1
    axes.set_xlim(-0.5, width - 0.5)
    axes.set_ylim(height - 0.5, -0.5)


def _draw_profile_on_axes(axes: Any, payload: Optional[BandProfilePayload]) -> None:
    """Draw a single detected band profile on axes.

    Parameters:
        axes: Matplotlib axes.
        payload: Band profile payload.

    Returns:
        None.
    """

    axes.set_title("Chosen {111} band profile")
    axes.set_xlabel("Profile index")
    axes.set_ylabel("Intensity (normalized)")
    if payload is None or payload.profile is None:
        axes.text(0.5, 0.5, "No valid profile", transform=axes.transAxes, ha="center", va="center")
        return
    profile = normalize_profile(np.asarray(payload.profile, dtype=np.float32))
    x_vals = np.arange(profile.size)
    axes.plot(x_vals, profile, color="#1f77b4", linewidth=1.8)
    for idx, color, label in (
        (payload.band_start_idx, "#d62728", "start"),
        (payload.central_peak_idx, "#2ca02c", "peak"),
        (payload.band_end_idx, "#d62728", "end"),
    ):
        if idx is not None and 0 <= idx < profile.size:
            axes.axvline(float(idx), color=color, linewidth=1.1, alpha=0.75, label=label)
    axes.grid(True, alpha=0.18)
    axes.legend(fontsize=8, loc="best")


def _json_ready(value: Any) -> Any:
    """Convert NumPy values into JSON-serializable objects.

    Parameters:
        value: Arbitrary value.

    Returns:
        JSON-ready value.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the single-pattern solver CLI parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Solve and render one EBSP pattern.")
    parser.add_argument("--config", required=True, type=Path, help="Single-pattern YAML file.")
    parser.add_argument("--json", type=Path, help="Optional output JSON path.")
    parser.add_argument("--png", type=Path, help="Optional output PNG path.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


def main() -> None:
    """Run the single-pattern solver CLI."""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    config = load_single_pattern_config(args.config)
    solution = solve_single_pattern(config)
    if args.json:
        write_solution_json(solution, args.json)
    if args.png:
        render_solution(solution, args.png)


if __name__ == "__main__":
    main()
