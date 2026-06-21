"""Tests for runtime indexed/acquisition orientation selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from orix.quaternion import Rotation

from kikuchiBandAnalyzer.band_width.orientation import (
    normalize_orientation_source,
    select_runtime_orientations,
)


class _FakeDetector:
    """Minimal detector returning a sentinel indexer."""

    def get_indexer(self, phase_list, hkl_list, **kwargs):
        """Return arguments for assertion by the fake signal.

        Parameters:
            phase_list: Configured phase list.
            hkl_list: Configured reflector families.
            kwargs: Hough options.

        Returns:
            Sentinel indexer dictionary.
        """

        return {"phase_list": phase_list, "hkl_list": hkl_list, **kwargs}


class _FakeSignal:
    """Minimal signal returning deterministic indexed rotations."""

    def __init__(self, indexed: Rotation, success: np.ndarray) -> None:
        """Initialize the fake signal.

        Parameters:
            indexed: Indexed rotations.
            success: Per-pixel indexing mask.
        """

        self._indexed = indexed
        self._success = success

    def hough_indexing(self, **kwargs):
        """Return a kikuchipy-like xmap and index arrays.

        Parameters:
            kwargs: Hough indexing options.

        Returns:
            Crystal-map stand-in, structured index data, and empty band data.
        """

        assert kwargs["return_index_data"] is True
        dtype = np.dtype([("fit", "f4"), ("cm", "f4")])
        index_data = np.zeros((1, self._success.size), dtype=dtype)
        index_data[0]["fit"] = [1.0, 2.0, 3.0]
        index_data[0]["cm"] = [0.8, 0.1, 0.9]
        xmap = SimpleNamespace(is_indexed=self._success, rotations=self._indexed)
        return xmap, index_data, np.empty((self._success.size, 0))


def test_normalize_orientation_source_accepts_legacy_original() -> None:
    """Map legacy and user-facing aliases to canonical source names."""

    assert normalize_orientation_source(None) == "indexed"
    assert normalize_orientation_source("live") == "indexed"
    assert normalize_orientation_source("original") == "acquisition"


def test_indexed_selection_falls_back_without_exporting_eulers() -> None:
    """Replace failed runtime rotations with acquisition rotations and mark them."""

    patterns = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    acquisition_eulers = np.deg2rad(
        np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    )
    indexed = Rotation.from_euler(
        np.deg2rad(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    )
    success = np.array([True, False, True])

    result = select_runtime_orientations(
        patterns,
        acquisition_eulers,
        _FakeDetector(),
        object(),
        [[1, 1, 1], [2, 0, 0]],
        source="indexed",
        signal=_FakeSignal(indexed, success),
    )

    selected = np.asarray(result.rotations.data).reshape(3, 4)
    acquisition = np.asarray(Rotation.from_euler(acquisition_eulers).data).reshape(3, 4)
    assert result.indexing_success.tolist() == [1, 0, 1]
    assert result.orientation_fallback.tolist() == [0, 1, 0]
    assert result.source_used.tolist() == [1, 0, 1]
    assert np.allclose(selected[1], acquisition[1])
    assert np.isnan(result.fit[1])
    assert np.isnan(result.confidence[1])


def test_acquisition_selection_does_not_run_indexing() -> None:
    """Use acquisition rotations directly when indexing is disabled."""

    patterns = np.zeros((1, 2, 4, 4), dtype=np.uint8)
    eulers = np.zeros((2, 3), dtype=np.float64)
    result = select_runtime_orientations(
        patterns,
        eulers,
        _FakeDetector(),
        object(),
        [[1, 1, 1]],
        source="acquisition",
    )
    assert result.indexing_success.tolist() == [-1, -1]
    assert result.orientation_fallback.tolist() == [0, 0]
    assert result.source_used.tolist() == [0, 0]


def test_scan_level_indexing_error_falls_back_all_pixels() -> None:
    """Continue with explicit acquisition fallback if the backend cannot initialize."""

    class _FailingDetector:
        """Detector stand-in that cannot create a PyEBSDIndex indexer."""

        def get_indexer(self, phase_list, hkl_list, **kwargs):
            """Raise the backend initialization failure.

            Parameters:
                phase_list: Configured phase list.
                hkl_list: Configured reflector families.
                kwargs: Hough options.

            Raises:
                RuntimeError: Always.
            """

            raise RuntimeError("backend unavailable")

    result = select_runtime_orientations(
        np.zeros((1, 2, 4, 4), dtype=np.uint8),
        np.zeros((2, 3), dtype=np.float64),
        _FailingDetector(),
        object(),
        [[1, 1, 1]],
        source="indexed",
    )
    assert result.indexing_success.tolist() == [0, 0]
    assert result.orientation_fallback.tolist() == [1, 1]
    assert result.source_used.tolist() == [0, 0]
