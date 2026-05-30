"""Tests for the single-pattern EBSP solver."""

from pathlib import Path

import numpy as np

from kikuchiBandAnalyzer.single_pattern_solver.solver import (
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
