"""Tests for alignment utilities and engine behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.transform import EuclideanTransform

from kikuchiBandAnalyzer.ebsd_compare.compare.engine import ComparisonEngine
from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.registration.alignment import (
    AlignmentSettings,
    build_alignment_from_parameters,
    estimate_alignment,
)
from kikuchiBandAnalyzer.ebsd_compare.simulated import InMemoryScanReader


def test_alignment_estimate_rotation_translation() -> None:
    """Estimate rotation and translation with RANSAC."""

    points_b = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]],
        dtype=float,
    )
    transform = EuclideanTransform(rotation=np.deg2rad(10.0), translation=(2.0, -1.0))
    points_a = transform(points_b)
    settings = AlignmentSettings(
        min_point_pairs=3,
        ransac_min_samples=3,
        ransac_residual_threshold=0.2,
        ransac_max_trials=2000,
        warp_order=1,
        warp_mode="constant",
        warp_cval=0.0,
        preserve_range=True,
        pattern_sampling="nearest",
    )
    result = estimate_alignment(points_a, points_b, settings)
    assert abs(result.rotation_deg - 10.0) < 0.5
    assert np.allclose(result.translation, (2.0, -1.0), atol=0.5)


def test_engine_alignment_map_triplet() -> None:
    """Ensure aligned maps line up under a translation."""

    map_a = np.zeros((5, 5), dtype=np.float32)
    map_b = np.zeros((5, 5), dtype=np.float32)
    map_a[2, 2] = 1.0
    map_b[2, 3] = 1.0
    reader_a = InMemoryScanReader({"IQ": map_a})
    reader_b = InMemoryScanReader({"IQ": map_b})
    scan_a = ScanDataset(
        file_path=Path("memory"),
        scan_name="A",
        nx=5,
        ny=5,
        catalog=reader_a.catalog(),
        reader=reader_a,
    )
    scan_b = ScanDataset(
        file_path=Path("memory"),
        scan_name="B",
        nx=5,
        ny=5,
        catalog=reader_b.catalog(),
        reader=reader_b,
    )
    alignment = build_alignment_from_parameters(0.0, [-1.0, 0.0])
    config = {"alignment": {"map_interpolation": "nearest", "warp": {"cval": 0.0}}}
    engine = ComparisonEngine(scan_a, scan_b, config, alignment=alignment)
    maps = engine.map_triplet("IQ", "delta")
    assert maps["B"][2, 2] == 1.0
    assert maps["D"][2, 2] == 0.0
