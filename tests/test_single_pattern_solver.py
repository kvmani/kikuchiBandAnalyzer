"""Tests for the single-pattern EBSP solver."""

from pathlib import Path
import copy

import numpy as np
from PIL import Image

from kikuchiBandAnalyzer.single_pattern_solver.solver import (
    SinglePatternConfig,
    clip_segment_to_bounds,
    load_single_pattern_config,
    solve_single_pattern,
)


def test_clip_segment_to_bounds_clips_to_image() -> None:
    """Line clipping should constrain output coordinates to image bounds."""

    clipped = clip_segment_to_bounds([-10, 5, 20, 5], width=12, height=10)

    assert clipped == [0.0, 5.0, 11.0, 5.0]


def test_single_pattern_ctf_config_solves() -> None:
    """The bundled CTF single-pattern config should load and produce a profile."""

    config = load_single_pattern_config(Path("configs/single_pattern_ctf.yml"))
    solution = solve_single_pattern(config)

    assert solution.pattern.shape == (240, 320)
    assert solution.detector_summary["convention"] == "oxford"
    assert solution.detector_summary["pc"] == [0.457, 0.584, 0.696374]
    assert any(line["hkl"] == "{111}" for line in solution.lines)
    assert solution.profile_payload is not None
    assert solution.profile_payload.profile is not None
    assert np.isfinite(solution.profile_payload.profile).all()


def test_single_pattern_da_config_solves() -> None:
    """The bundled DA single-pattern config should load with EDAX PC convention."""

    config = load_single_pattern_config(Path("configs/single_pattern_da.yml"))
    solution = solve_single_pattern(config)

    assert solution.pattern.shape == (230, 230)
    assert solution.detector_summary["convention"] == "edax"
    assert any(line["hkl"] == "{111}" for line in solution.lines)


def test_single_pattern_can_overlay_hough_indexed_orientation() -> None:
    """The solver should optionally use kikuchipy's indexed orientation for overlays."""

    config = load_single_pattern_config(Path("configs/single_pattern_da.yml"))
    raw = copy.deepcopy(config.raw)
    raw["hough"]["enabled"] = True
    raw["hough"]["use_indexed_orientation"] = True
    solution = solve_single_pattern(SinglePatternConfig(path=config.path, raw=raw))

    assert solution.hough_summary is not None
    assert solution.hough_summary["success"] is True
    assert solution.hough_summary["use_indexed_orientation"] is True
    assert len(solution.hough_summary["indexed_eulers_deg"]) == 3


def test_single_pattern_image_input_solves(tmp_path: Path) -> None:
    """A standalone EBSP image should solve using manually supplied Euler angles."""

    image_path = tmp_path / "single_pattern.png"
    image = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (80, 1))
    Image.fromarray(image).save(image_path)
    config = SinglePatternConfig(
        path=tmp_path / "single_pattern.yml",
        raw={
            "input": {
                "type": "image",
                "path": str(image_path),
                "eulers_deg": [0.0, 45.0, 0.0],
                "x": 0,
                "y": 0,
            },
            "phase": {
                "name": "Ni",
                "space_group": 225,
                "lattice": [3.5236, 3.5236, 3.5236, 90, 90, 90],
                "atoms": [{"element": "Ni", "position": [0, 0, 0]}],
            },
            "detector": {
                "convention": "edax",
                "pc": [0.5, 0.5, 0.6],
                "sample_tilt": 70.0,
                "tilt": 0.0,
                "azimuthal": 0.0,
                "px_size": 1.0,
                "binning": 1,
            },
            "orientation": {"direction": "lab2crystal"},
            "simulation": {"hkl_list": [[1, 1, 1], [2, 0, 0]]},
            "band_profile": {"desired_hkl": "1,1,1", "rectWidth": 10},
            "hough": {"enabled": False},
        },
    )

    solution = solve_single_pattern(config)

    assert solution.pattern.shape == (80, 96)
    assert np.allclose(np.rad2deg(solution.eulers_rad), [0.0, 45.0, 0.0])
    assert solution.detector_summary["pc"] == [0.5, 0.5, 0.6]
