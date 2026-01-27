"""Run the EBSD compare GUI demo and capture a screenshot."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from PySide6 import QtCore, QtWidgets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kikuchiBandAnalyzer.ebsd_compare.gui.main_window import EbsdCompareMainWindow
from kikuchiBandAnalyzer.ebsd_compare.noise import NoisyOh5Generator
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging, load_yaml_config


def create_simulated_oh5(output_path: Path, logger: logging.Logger) -> None:
    """Create a small simulated OH5 file for debug mode.

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
        data_group.create_dataset(
            "IQ", data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )
        data_group.create_dataset(
            "CI", data=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        )
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
        profile += 0.05 * rng.normal(0.0, 1.0, size=profile.shape).astype(np.float32)
        profile = np.clip(profile, 0.0, None)
        band_profiles = np.tile(profile[None, :], (n_pixels, 1)).astype(np.float32)
        data_group.create_dataset("band_profile", data=band_profiles)
        central_lines = np.tile(
            np.array([10.0, 10.0, 52.0, 52.0], dtype=np.float32)[None, :],
            (n_pixels, 1),
        )
        data_group.create_dataset("central_line", data=central_lines)
        start_idx = np.full(n_pixels, int(profile_len * 0.25), dtype=np.int32)
        end_idx = np.full(n_pixels, int(profile_len * 0.75), dtype=np.int32)
        peak_idx_arr = np.full(n_pixels, int(peak_idx), dtype=np.int32)
        profile_len_arr = np.full(n_pixels, int(profile_len), dtype=np.int32)
        band_valid = np.ones(n_pixels, dtype=np.int8)
        data_group.create_dataset("band_start_idx", data=start_idx)
        data_group.create_dataset("band_end_idx", data=end_idx)
        data_group.create_dataset("central_peak_idx", data=peak_idx_arr)
        data_group.create_dataset("profile_length", data=profile_len_arr)
        data_group.create_dataset("band_valid", data=band_valid)
    logger.debug("Created simulated OH5 at %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description="Run EBSD compare GUI demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ebsd_compare_config.yml"),
        help="Path to EBSD compare config.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (uses a small simulated dataset).",
    )
    return parser


def _load_demo_config(config_path: Path) -> Dict:
    """Load the demo configuration from YAML.

    Parameters:
        config_path: Path to the YAML config.

    Returns:
        Demo configuration dictionary.
    """

    config = load_yaml_config(config_path)
    return config.get("demo", {})


def _prepare_paths(
    config_path: Path, debug: bool, logger: logging.Logger
) -> Tuple[Path, Path, Path]:
    """Prepare scan and screenshot paths for the demo.

    Parameters:
        config_path: Path to the YAML config.
        debug: Whether debug mode is active.
        logger: Logger instance.

    Returns:
        Tuple of (scan_a, scan_b, screenshot_path).
    """

    if debug:
        scan_a = Path("tmp/debug_input.oh5")
        scan_b = Path("tmp/debug_noisy.oh5")
        create_simulated_oh5(scan_a, logger)
        generator = NoisyOh5Generator(
            input_path=scan_a,
            output_path=scan_b,
            sigma_map={"IQ": 0.05, "CI": 0.02, "band_profile": 0.01},
            seed=123,
            logger=logger,
        )
        generator.run()
        screenshot_path = Path("docs/screenshots/ebsd_compare_band_profile_proof.png")
        return scan_a, scan_b, screenshot_path
    demo = _load_demo_config(config_path)
    scan_a = Path(demo.get("scan_a", "testData/Test_Ti.oh5"))
    scan_b = Path(demo.get("scan_b", "testData/Test_Ti_noisy.oh5"))
    screenshot_path = Path(
        demo.get("output_screenshot", "tmp/ebsd_compare_gui_proof.png")
    )
    noise_config = load_yaml_config(config_path).get("noisy_generation", {})
    generator = NoisyOh5Generator(
        input_path=Path(noise_config.get("input_path", scan_a)),
        output_path=Path(noise_config.get("output_path", scan_b)),
        sigma_map=noise_config.get("sigma", {"IQ": 0.05, "CI": 0.02}),
        seed=int(noise_config.get("seed", 123)),
        logger=logger,
    )
    generator.run()
    return scan_a, scan_b, screenshot_path


def _ensure_offscreen(logger: logging.Logger) -> None:
    """Prefer Qt offscreen platform if available.

    Parameters:
        logger: Logger instance.
    """

    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        logger.info("QT_QPA_PLATFORM set to offscreen for screenshot capture.")


def _require_qt_dependencies(logger: logging.Logger) -> None:
    """Ensure Qt runtime dependencies are present or raise a clear error.

    Parameters:
        logger: Logger instance.
    """

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


def main() -> None:
    """Run the demo, capture a screenshot, and exit."""

    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    configure_logging(args.debug, config.get("ebsd_compare", {}).get("logging"))
    logger = logging.getLogger(__name__)
    _ensure_offscreen(logger)
    _require_qt_dependencies(logger)
    scan_a, scan_b, screenshot_path = _prepare_paths(args.config, args.debug, logger)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication([])
    window = EbsdCompareMainWindow(args.config)
    window.show()
    window.load_scans(scan_a, scan_b)

    def _capture_and_exit() -> None:
        window.save_screenshot(screenshot_path)
        app.quit()

    QtCore.QTimer.singleShot(500, _capture_and_exit)
    app.exec()


if __name__ == "__main__":
    main()
