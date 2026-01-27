"""Run the Automator GUI demo and capture a screenshot."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PySide6 import QtCore, QtWidgets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kikuchiBandAnalyzer.automator_gui.main_window import AutomatorGuiMainWindow
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging


def create_simulated_band_output(output_path: Path, logger: logging.Logger) -> None:
    """Create a small OH5 file that already contains band-profile outputs.

    Parameters:
        output_path: Path for the simulated OH5 file.
        logger: Logger instance.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx = 2
    ny = 2
    pattern_h = 64
    pattern_w = 64
    profile_len = 64
    n_pixels = nx * ny
    rng = np.random.default_rng(123)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("Manufacturer", data="Debug")
        handle.create_dataset("Version", data="1.0")
        scan = handle.create_group("DebugScan")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        header.create_dataset("nColumns", data=np.array([nx]))
        header.create_dataset("nRows", data=np.array([ny]))
        header.create_dataset("Pattern Height", data=np.array([pattern_h]))
        header.create_dataset("Pattern Width", data=np.array([pattern_w]))
        data_group = ebsd.create_group("Data")
        data_group.create_dataset("IQ", data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        data_group.create_dataset("CI", data=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        yy, xx = np.mgrid[0:pattern_h, 0:pattern_w]
        base_pattern = (
            np.exp(-((xx - pattern_w / 2) ** 2 + (yy - pattern_h / 2) ** 2) / (2 * (pattern_w / 6) ** 2))
            + 0.2 * np.sin(xx / 5.0)
        ).astype(np.float32)
        patterns = np.stack(
            [base_pattern + rng.normal(0.0, 0.02, size=base_pattern.shape).astype(np.float32) for _ in range(n_pixels)],
            axis=0,
        )
        data_group.create_dataset("Pattern", data=patterns)

        x_prof = np.arange(profile_len, dtype=np.float32)
        peak_idx = profile_len // 2
        profile = np.exp(-((x_prof - peak_idx) ** 2) / (2 * (profile_len / 10) ** 2)).astype(np.float32)
        profile = np.clip(profile + 0.03 * rng.normal(0.0, 1.0, size=profile.shape).astype(np.float32), 0.0, None)
        band_profiles = np.tile(profile[None, :], (n_pixels, 1)).astype(np.float32)
        data_group.create_dataset("band_profile", data=band_profiles)
        data_group.create_dataset(
            "central_line",
            data=np.tile(np.array([10.0, 10.0, 52.0, 52.0], dtype=np.float32)[None, :], (n_pixels, 1)),
        )
        data_group.create_dataset("band_start_idx", data=np.full(n_pixels, int(profile_len * 0.25), dtype=np.int32))
        data_group.create_dataset("central_peak_idx", data=np.full(n_pixels, int(peak_idx), dtype=np.int32))
        data_group.create_dataset("band_end_idx", data=np.full(n_pixels, int(profile_len * 0.75), dtype=np.int32))
        data_group.create_dataset("profile_length", data=np.full(n_pixels, int(profile_len), dtype=np.int32))
        data_group.create_dataset("band_valid", data=np.ones(n_pixels, dtype=np.int8))
        data_group.create_dataset("Band_Width", data=np.full(n_pixels, 12.3, dtype=np.float32))
        data_group.create_dataset("psnr", data=np.full(n_pixels, 5.4, dtype=np.float32))
    logger.info("Created simulated automator input at %s", output_path)


def write_demo_yaml(output_path: Path, oh5_path: Path) -> None:
    """Write a minimal YAML config for the GUI loader.

    Parameters:
        output_path: YAML destination path.
        oh5_path: Path to the simulated OH5 file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"h5_file_path: {oh5_path}",
                "debug: true",
                "rectWidth: 16",
                "desired_hkl: 110",
                "desired_hkl_ref_width: 1.0",
                "elastic_modulus: 1.0",
                "phase_list:",
                "  name: Ni",
                "  space_group: 225",
                "  lattice: [1, 1, 1, 90, 90, 90]",
                "  atoms:",
                "    - element: Ni",
                "      position: [0, 0, 0]",
                "hkl_list:",
                "  - [1, 1, 0]",
            ]
        ),
        encoding="utf-8",
    )


def _ensure_offscreen(logger: logging.Logger) -> None:
    """Prefer Qt offscreen platform if available."""

    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        logger.info("QT_QPA_PLATFORM set to offscreen for screenshot capture.")


def _require_qt_dependencies(logger: logging.Logger) -> None:
    """Ensure Qt runtime dependencies are present or raise a clear error."""

    if shutil.which("ldconfig") is None:
        logger.warning("ldconfig not found; skipping dependency check.")
        return
    missing = [
        lib
        for lib in ("libGL.so.1", "libxkbcommon.so.0", "libEGL.so.1")
        if os.system(f"ldconfig -p | grep -q {lib}") != 0
    ]
    if missing:
        logger.error("Missing Qt runtime libraries: %s", ", ".join(missing))
        raise RuntimeError("Missing Qt runtime dependencies.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""

    parser = argparse.ArgumentParser(description="Run Automator GUI demo.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a small simulated dataset (recommended for proof screenshots).",
    )
    return parser


def main() -> None:
    """Run the demo, capture a screenshot, and exit."""

    args = build_arg_parser().parse_args()
    configure_logging(args.debug, None)
    logger = logging.getLogger(__name__)
    _ensure_offscreen(logger)
    _require_qt_dependencies(logger)

    screenshot_path = Path("docs/screenshots/automator_gui_proof.png")
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    oh5_path = Path("tmp/automator_gui_demo.oh5")
    yaml_path = Path("tmp/automator_gui_demo.yml")
    create_simulated_band_output(oh5_path, logger)
    write_demo_yaml(yaml_path, oh5_path)

    app = QtWidgets.QApplication([])
    window = AutomatorGuiMainWindow(config_path=yaml_path)
    window.show()

    def _capture_and_exit() -> None:
        pixmap = window.grab()
        pixmap.save(str(screenshot_path), "PNG")
        logger.info("Saved screenshot to %s", screenshot_path)
        app.quit()

    QtCore.QTimer.singleShot(500, _capture_and_exit)
    app.exec()


if __name__ == "__main__":
    main()

