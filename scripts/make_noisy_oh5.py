"""Generate a noisy OH5 file by perturbing IQ and CI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kikuchiBandAnalyzer.ebsd_compare.noise import NoisyOh5Generator
from kikuchiBandAnalyzer.ebsd_compare.utils import configure_logging, load_yaml_config


def create_simulated_oh5(output_path: Path, logger: logging.Logger) -> None:
    """Create a small simulated OH5 file for debug mode.

    Parameters:
        output_path: Path for the simulated OH5 file.
        logger: Logger instance.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("Manufacturer", data="Debug")
        handle.create_dataset("Version", data="1.0")
        scan = handle.create_group("DebugScan")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        header.create_dataset("nColumns", data=np.array([2]))
        header.create_dataset("nRows", data=np.array([2]))
        data_group = ebsd.create_group("Data")
        data_group.create_dataset("IQ", data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        data_group.create_dataset("CI", data=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    logger.debug("Created simulated OH5 at %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description="Generate a noisy OH5 file.")
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


def _load_noise_config(config_path: Path) -> Dict:
    """Load the noise generation configuration.

    Parameters:
        config_path: Path to the YAML config.

    Returns:
        Noise generation config dictionary.
    """

    config = load_yaml_config(config_path)
    return config.get("noisy_generation", {})


def main() -> None:
    """Run the noise generation script."""

    args = build_arg_parser().parse_args()
    log_config = load_yaml_config(args.config).get("ebsd_compare", {}).get("logging")
    configure_logging(args.debug, log_config)
    logger = logging.getLogger(__name__)
    if args.debug:
        input_path = Path("tmp/debug_input.oh5")
        output_path = Path("tmp/debug_noisy.oh5")
        create_simulated_oh5(input_path, logger)
        sigma_map = {"IQ": 0.05, "CI": 0.02}
        seed = 123
    else:
        config = _load_noise_config(args.config)
        input_path = Path(config.get("input_path", "testData/Test_Ti.oh5"))
        output_path = Path(config.get("output_path", "testData/Test_Ti_noisy.oh5"))
        sigma_map = config.get("sigma", {"IQ": 0.05, "CI": 0.02})
        seed = int(config.get("seed", 123))
    generator = NoisyOh5Generator(
        input_path=input_path,
        output_path=output_path,
        sigma_map=sigma_map,
        seed=seed,
        logger=logger,
    )
    generator.run()


if __name__ == "__main__":
    main()
