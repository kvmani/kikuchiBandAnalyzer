
"""Automation entry point for batch Kikuchi band width detection.

This script wraps the lower level detection utilities and provides a
configuration driven pipeline.  The :class:`BandWidthAutomator` class loads
an EBSD data set, simulates Kikuchi patterns, performs band width detection
and finally exports all results.  It is intended to be run either in a normal
mode, reading all options from a YAML configuration file, or in a debug mode
where the data set is cropped and detailed logging is enabled.
"""

import argparse
import time
import os
from pathlib import Path
from configLoader import load_config
import logging

import matplotlib.pyplot as plt
import kikuchipy as kp
from orix import plot
from diffsims.crystallography import ReciprocalLatticeVector
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Rotation
from orix.vector import Vector3d
from typing import Optional
import numpy as np

import pandas as pd
import json
from kikuchiBandWidthDetector import KikuchiBatchProcessor
from kikuchiBandWidthDetector import prepare_json_input
import shutil
import h5py
import utilities as ut
from kikuchiBandAnalyzer.band_width.ctf_acquisition import (
    CtfBandWidthAcquisition,
    export_ctf_with_prias_metrics,
)
from kikuchiBandAnalyzer.derived_fields import build_default_registry, write_hdf5_dataset
from simulators import (
    make_text_marker,
    CustomGeometricalKikuchiPatternSimulation,
    CustomKikuchiPatternSimulator,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BandWidthAutomator:
    """Automate band width detection for an EBSD data set.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file controlling the pipeline.  The
        configuration describes the input HDF5/ANG files, material information
        and all detection parameters.  ``bandDetectorOptionsDebug.yml`` is used
        by default.
    """

    def __init__(self, config_path: str = "bandDetectorOptionsHcp.yml"):
        """Instantiate the automator and load the configuration."""

        self.config = load_config(config_path)
        self.source_format = "ctf" if self.config.get("ctf_file_path") else "h5"
        self.data_path = Path(
            self.config.get(
                "ctf_file_path",
                self.config.get("h5_file_path", "path_to_default_file.h5"),
            )
        )
        configured_output_dir = self.config.get("output_dir")
        self.output_dir = Path(configured_output_dir) if configured_output_dir else self.data_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = self.data_path.stem
        self.modified_data_path = self.output_dir / f"{self.base_name}_modified.h5"
        self.in_ang_path = self.output_dir / f"{self.base_name}.ang"
        self.dataset = None
        self.grouped_dict_list = None
        self.ctf_acquisition_result = None
        logging.info('Justcompleted the object initiation')

    # ------------------------------------------------------------------
    def prepare_dataset(self):
        """Load the EBSD data set and optionally crop for debug mode."""

        if self.source_format == "ctf":
            acquisition = CtfBandWidthAcquisition(
                self.config,
                output_dir=self.output_dir,
                logger=logging.getLogger(__name__),
            )
            result = acquisition.prepare()
            self.dataset = result.dataset
            self.modified_data_path = result.modified_h5_path
            self.base_name = self.data_path.stem
            self.ctf_acquisition_result = result
            logging.info("Prepared CTF pattern-folder dataset from %s.", self.data_path)
            return

        path = self.data_path
        if path.suffix == ".oh5":
            new_data_path = self.output_dir / f"{path.stem}.h5"
            shutil.copy(path, new_data_path)
            logging.info(f"Copied .oh5 file to new .h5 file: {new_data_path}")
            path = new_data_path
            self.data_path = new_data_path

        if Path(path).resolve() != self.modified_data_path.resolve():
            shutil.copy(path, self.modified_data_path)
            logging.info(f"Copied HDF5 file to: {self.modified_data_path}")
        else:
            logging.info("Using existing HDF5 working file: %s", self.modified_data_path)

        logging.info(f"Loading dataset from: {path}")
        self.dataset = kp.load(path, lazy=False)

        if self.config.get("debug", False):
            crop_start = self.config.get("crop_start", 5)
            crop_end = self.config.get("crop_end", 25)
            logging.info("Debug mode enabled: Cropping data for faster processing.")
            self.dataset.crop(1, start=crop_start, end=crop_end + 10)
            self.dataset.crop(0, start=crop_start, end=crop_end)

    # ------------------------------------------------------------------
    def simulate_and_index(self):
        """Simulate production band locations from acquisition orientations.

        Hough indexing is intentionally excluded from the batch production path.
        It is available in the single-pattern diagnostic solver, but exported
        HDF5/OH5/ANG orientations and production band locations must continue to
        use the Euler angles supplied by the acquisition data.
        """
        annotation_path = self.config.get("band_annotation_json_path") or self.config.get(
            "line_annotation_json_path"
        )
        if annotation_path:
            n_patterns = int(np.prod(self.dataset.data.shape[:2]))
            self.grouped_dict_list = prepare_json_input(
                str(annotation_path),
                n_patterns=n_patterns,
                tile_from_single=bool(self.config.get("tile_annotations_from_single", False)),
            )
            logging.info(
                "Loaded %d precomputed band-line annotation entries from %s.",
                len(self.grouped_dict_list),
                annotation_path,
            )
            return
        if self.source_format == "ctf":
            self.grouped_dict_list = self._simulate_ctf_band_lines()
            return
        if str(self.config.get("orientation_source", "original")).lower() == "original":
            self.grouped_dict_list = self._simulate_h5_original_band_lines()
            return
        phase_cfg = self.config["phase_list"]
        phase_list = PhaseList(
            Phase(
                name=phase_cfg["name"],
                space_group=phase_cfg["space_group"],
                structure=Structure(
                    lattice=Lattice(*phase_cfg["lattice"]),
                    atoms=[Atom(at["element"], at["position"]) for at in phase_cfg["atoms"]],
                ),
            ),
        )
        hkl_list = self.config["hkl_list"]
        header_data = ut.extract_header_data(str(self.modified_data_path))

        sig_shape = self.dataset.axes_manager.signal_shape[::-1]
        detector_cfg = dict(self.config.get("detector", {}) or {})
        convention = str(detector_cfg.get("convention", self.config.get("detector_convention", "edax")))
        pc = detector_cfg.get("pc", self.config.get("pc", header_data.get("pc", (0.0, 0.0, 0.0))))
        det = kp.detectors.EBSDDetector(
            sig_shape,
            sample_tilt=float(detector_cfg.get("sample_tilt", header_data.get("Sample Tilt", 0.0))),
            tilt=float(detector_cfg.get("tilt", header_data.get("Camera Elevation Angle", 0.0))),
            azimuthal=float(detector_cfg.get("azimuthal", header_data.get("Camera Azimuthal Angle", 0.0))),
            convention=convention,
            pc=tuple(pc),
        )
        logging.info(
            "Built EBSD detector for HDF5/TSL route with convention=%s, pc=%s.",
            convention,
            tuple(pc),
        )

        indexer = det.get_indexer(phase_list, hkl_list, nBands=10, tSigma=2, rSigma=2)
        xmap, index_data, indexed_band_data = self.dataset.hough_indexing(
            phase_list=phase_list,
            indexer=indexer,
            return_index_data=True,
            return_band_data=True,
            verbose=1,
        )

        phase = phase_list[0]
        ref = ReciprocalLatticeVector(phase=xmap.phases[0], hkl=hkl_list).symmetrise()
        simulator = CustomKikuchiPatternSimulator(ref)
        sim = simulator.on_detector(det, xmap.rotations.reshape(*xmap.shape))
        sim.phase = phase

        desired_hkl = str(self.config.get("desired_hkl", "1,1,1"))
        markers, grouped_dict_list = sim.as_markers(
            kikuchi_line_labels=True, desired_hkl=desired_hkl
        )
        self.dataset.add_marker(markers, plot_marker=False, permanent=True)
        self.grouped_dict_list = grouped_dict_list

        if not self.config.get("skip_display_EBSDmap", False):
            v_ipf = Vector3d.xvector()
            sym = xmap.phases[0].point_group
            rgb = plot.IPFColorKeyTSL(sym, v_ipf).orientation2color(xmap.rotations)
            maps_nav_rgb = kp.draw.get_rgb_navigator(rgb.reshape(xmap.shape + (3,)))
            self.dataset.plot(maps_nav_rgb)
            plt.show()

    def _simulate_h5_original_band_lines(self):
        """Simulate HDF5/OH5 band annotations from stored Euler datasets.

        Returns:
            Grouped line-annotation dictionaries consumable by
            :class:`KikuchiBatchProcessor`.

        Raises:
            KeyError: If no supported Euler triplet exists in the HDF5 data.
            ValueError: If the Euler count does not match the pattern grid.
        """

        phase_list = self._build_phase_list()
        phase = phase_list[0]
        euler = self._read_h5_original_eulers()
        expected = int(np.prod(self.dataset.data.shape[:2]))
        if euler.shape != (expected, 3):
            raise ValueError(
                "Stored Euler count does not match loaded pattern count: "
                f"Euler shape={euler.shape}, expected=({expected}, 3)."
            )

        header_data = ut.extract_header_data(str(self.modified_data_path))
        detector_cfg = dict(self.config.get("detector", {}) or {})
        detector = kp.detectors.EBSDDetector(
            shape=tuple(int(value) for value in self.dataset.data.shape[-2:]),
            sample_tilt=float(detector_cfg.get("sample_tilt", header_data.get("Sample Tilt", 0.0))),
            tilt=float(detector_cfg.get("tilt", header_data.get("Camera Elevation Angle", 0.0))),
            azimuthal=float(detector_cfg.get("azimuthal", header_data.get("Camera Azimuthal Angle", 0.0))),
            convention=str(detector_cfg.get("convention", self.config.get("detector_convention", "edax"))),
            pc=tuple(detector_cfg.get("pc", self.config.get("pc", header_data.get("pc", (0.5, 0.5, 0.5))))),
        )
        direction = str(self.config.get("orientation_direction", "lab2crystal"))
        rotations = Rotation.from_euler(euler, direction=direction, degrees=False)
        rotations = rotations.reshape(*self.dataset.data.shape[:2])
        reflectors = ReciprocalLatticeVector(
            phase=phase,
            hkl=self.config["hkl_list"],
        ).symmetrise()
        simulation = CustomKikuchiPatternSimulator(reflectors).on_detector(detector, rotations)
        simulation.phase = phase
        _, grouped = simulation.as_markers(
            kikuchi_line_labels=True,
            desired_hkl=str(self.config.get("desired_hkl", "1,1,1")),
        )
        if len(grouped) != expected:
            grouped = self._pad_grouped_annotations(grouped)
        logging.info(
            "Generated production Kikuchi-line annotations for %d HDF5/OH5 pixels "
            "from the original stored Euler angles.",
            len(grouped),
        )
        return grouped

    def _read_h5_original_eulers(self) -> np.ndarray:
        """Read original Euler angles from the working HDF5 file in radians.

        Returns:
            Euler angle array shaped ``(n_pixels, 3)`` in radians.

        Raises:
            KeyError: If neither the TSL radian nor degree Euler fields exist.
        """

        with h5py.File(self.modified_data_path, "r") as handle:
            scan_name = next(
                name
                for name, item in handle.items()
                if name not in {"Manufacturer", "Version"} and isinstance(item, h5py.Group)
            )
            data = handle[f"/{scan_name}/EBSD/Data"]
            radians_fields = ("Phi1", "Phi", "Phi2")
            degrees_fields = ("Euler1", "Euler2", "Euler3")
            if all(name in data for name in radians_fields):
                return np.column_stack(
                    [np.asarray(data[name][()]).reshape(-1) for name in radians_fields]
                ).astype(np.float64)
            if all(name in data for name in degrees_fields):
                degrees = np.column_stack(
                    [np.asarray(data[name][()]).reshape(-1) for name in degrees_fields]
                ).astype(np.float64)
                return np.deg2rad(degrees)
        raise KeyError(
            "HDF5/OH5 input contains neither Phi1/Phi/Phi2 nor "
            "Euler1/Euler2/Euler3 orientation datasets."
        )

    def _build_phase_list(self) -> PhaseList:
        """Build a PhaseList from configuration.

        Returns:
            PhaseList containing the configured crystal phase.
        """

        phase_cfg = self.config["phase_list"]
        atoms_cfg = phase_cfg.get("atoms")
        if atoms_cfg:
            atoms = [Atom(at["element"], at["position"]) for at in atoms_cfg]
        else:
            logging.warning(
                "phase_list.atoms is missing; using one %s atom at [0, 0, 0]. "
                "For production work, add explicit atoms to the YAML phase_list.",
                phase_cfg["name"],
            )
            atoms = [Atom(phase_cfg["name"], [0, 0, 0])]
        return PhaseList(
            Phase(
                name=phase_cfg["name"],
                space_group=phase_cfg["space_group"],
                structure=Structure(
                    lattice=Lattice(*phase_cfg["lattice"]),
                    atoms=atoms,
                ),
            ),
        )

    def _simulate_ctf_band_lines(self):
        """Simulate Kikuchi line annotations from CTF Euler angles.

        Returns:
            Grouped line-annotation dictionaries consumable by KikuchiBatchProcessor.
        """

        if self.ctf_acquisition_result is None:
            raise RuntimeError(
                "CTF acquisition has not been prepared. Call prepare_dataset() before "
                "simulate_and_index(), or use BandWidthAutomator.run()."
            )
        phase_list = self._build_phase_list()
        phase = phase_list[0]
        hkl_list = self.config["hkl_list"]
        euler = self.ctf_acquisition_result.euler_angles_deg
        if euler.shape[0] != int(np.prod(self.dataset.data.shape[:2])):
            raise ValueError(
                "CTF Euler count does not match loaded pattern count. "
                f"Euler rows={euler.shape[0]}, pattern grid={self.dataset.data.shape[:2]}. "
                "Check XCells/YCells and the pattern folder mapping."
            )
        detector = self._build_ctf_detector()
        direction = self.config.get("ctf_euler_direction", "lab2crystal")
        rotations = Rotation.from_euler(euler, direction=direction, degrees=True)
        rotations = rotations.reshape(*self.dataset.data.shape[:2])
        ref = ReciprocalLatticeVector(phase=phase, hkl=hkl_list).symmetrise()
        simulator = CustomKikuchiPatternSimulator(ref)
        sim = simulator.on_detector(detector, rotations)
        sim.phase = phase
        desired_hkl = str(self.config.get("desired_hkl", "1,1,1"))
        _, grouped_dict_list = sim.as_markers(
            kikuchi_line_labels=True,
            desired_hkl=desired_hkl,
        )
        expected = int(np.prod(self.dataset.data.shape[:2]))
        if len(grouped_dict_list) != expected:
            logging.warning(
                "CTF line simulation produced %d grouped pixel entries; expected %d. "
                "Pixels without visible target lines will still be processed with empty annotations. "
                "If many entries are missing, check detector geometry, PC, sample_tilt, Euler convention, and desired_hkl.",
                len(grouped_dict_list),
                expected,
            )
            grouped_dict_list = self._pad_grouped_annotations(grouped_dict_list)
        logging.info("Generated CTF Kikuchi-line annotations for %d pixels.", len(grouped_dict_list))
        return grouped_dict_list

    def _build_ctf_detector(self):
        """Build an Oxford-convention EBSD detector for CTF simulations.

        Returns:
            kikuchipy EBSDDetector configured for the CTF pattern geometry.
        """

        if self.ctf_acquisition_result is None:
            raise RuntimeError("CTF acquisition result is unavailable.")
        detector_cfg = dict(self.config.get("ctf_detector", {}) or {})
        shape = tuple(int(value) for value in self.ctf_acquisition_result.pattern_shape)
        pc = detector_cfg.get("pc", self.config.get("pc", [0.5, 0.5, 0.5]))
        if len(pc) != 3:
            raise ValueError(
                "ctf_detector.pc must contain three values [x*, y*, z*]. "
                "Example: ctf_detector: {pc: [0.5, 0.5, 0.5]}."
            )
        pc = tuple(float(value) for value in pc)
        if any(value <= 0 or value >= 1 for value in pc[:2]) or pc[2] <= 0:
            logging.warning(
                "CTF detector pattern center %s is unusual. Verify the PC convention "
                "and use ctf_detector.convention='oxford' with an Oxford-format PC when possible.",
                pc,
            )
        for key, default in (
            ("sample_tilt", 70.0),
            ("tilt", 0.0),
            ("azimuthal", 0.0),
        ):
            if key not in detector_cfg:
                logging.warning(
                    "ctf_detector.%s is not configured; using default %s. "
                    "For production CTF analysis, set detector geometry explicitly.",
                    key,
                    default,
                )
        return kp.detectors.EBSDDetector(
            shape=shape,
            px_size=float(detector_cfg.get("px_size", 1.0)),
            binning=int(detector_cfg.get("binning", 1)),
            sample_tilt=float(detector_cfg.get("sample_tilt", 70.0)),
            tilt=float(detector_cfg.get("tilt", 0.0)),
            azimuthal=float(detector_cfg.get("azimuthal", 0.0)),
            pc=pc,
            convention=str(detector_cfg.get("convention", "oxford")),
        )

    def _pad_grouped_annotations(self, grouped_dict_list):
        """Pad sparse grouped annotations to one entry per scan pixel.

        Parameters:
            grouped_dict_list: Existing grouped annotation entries.

        Returns:
            Dense grouped annotation list sorted by row-major pixel index.
        """

        n_rows, n_cols = self.dataset.data.shape[:2]
        by_index = {
            int(entry.get("ind", -1)): entry
            for entry in grouped_dict_list
            if isinstance(entry, dict)
        }
        dense = []
        for row in range(n_rows):
            for col in range(n_cols):
                idx = row * n_cols + col
                dense.append(by_index.get(idx, {"x,y": [row, col], "ind": idx, "points": []}))
        return dense

    # ------------------------------------------------------------------
    def detect_band_widths(self, progress_callback=None, cancel_callback=None):
        """Run the :class:`KikuchiBatchProcessor` over all patterns.

        Parameters:
            progress_callback: Optional callback invoked after each processed pixel.
                Signature: (row, col, processed_count, total_count, entry) -> None.
            cancel_callback: Optional callable returning True when cancellation is requested.

        Returns:
            List of processed pixel entries.
        """
        desired_hkl = str(self.config.get("desired_hkl", "1,1,1"))
        ebsd_data = self.dataset.data
        processor = KikuchiBatchProcessor(
            ebsd_data,
            self.grouped_dict_list,
            config=self.config,
            desired_hkl=desired_hkl,
        )
        return processor.process(
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    # ------------------------------------------------------------------
    def _select_best_band(self, bands, pixel_index):
        """
        Select the best band for a pixel based on PSNR.

        Parameters:
            bands: List of band dictionaries.
            pixel_index: Linear pixel index for logging context.

        Returns:
            Selected band dictionary or None if no valid bands are found.
        """
        valid_bands = []
        for band in bands or []:
            if not band.get("band_valid", False):
                continue
            psnr = band.get("psnr", 0)
            if psnr is None or not np.isfinite(psnr):
                continue
            valid_bands.append(band)

        if not valid_bands:
            return None

        psnr_values = [band.get("psnr", 0) for band in valid_bands]
        max_psnr = max(psnr_values)
        matches = [band for band in valid_bands if band.get("psnr", 0) == max_psnr]
        if len(matches) > 1:
            logging.warning(
                "Multiple bands share PSNR %.3f at pixel %d; selecting first match.",
                max_psnr,
                pixel_index,
            )
        return matches[0]

    def _coerce_profile(self, profile, expected_length, pixel_index):
        """
        Coerce a band profile into a fixed-length numpy array.

        Parameters:
            profile: Raw profile list/array.
            expected_length: Target length for the profile vector.
            pixel_index: Linear pixel index for logging context.

        Returns:
            NumPy array of shape (expected_length,).
        """
        if expected_length <= 0:
            raise ValueError("Band profile length must be positive.")
        if profile is None:
            logging.warning("Missing band profile at pixel %d; filling with NaNs.", pixel_index)
            return np.full(expected_length, np.nan, dtype=np.float32)

        profile_arr = np.asarray(profile, dtype=np.float32).ravel()
        if profile_arr.size < expected_length:
            logging.warning(
                "Band profile length %d < %d at pixel %d; padding with zeros.",
                profile_arr.size,
                expected_length,
                pixel_index,
            )
            profile_arr = np.pad(profile_arr, (0, expected_length - profile_arr.size), mode="constant")
        elif profile_arr.size > expected_length:
            logging.warning(
                "Band profile length %d > %d at pixel %d; truncating.",
                profile_arr.size,
                expected_length,
                pixel_index,
            )
            profile_arr = profile_arr[:expected_length]

        if not np.all(np.isfinite(profile_arr)):
            logging.warning("Non-finite values in band profile at pixel %d; replacing with zeros.", pixel_index)
            profile_arr = np.nan_to_num(profile_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return profile_arr.astype(np.float32)

    def _coerce_central_line(self, central_line, pixel_index):
        """
        Coerce a central line into a fixed-length numpy array.

        Parameters:
            central_line: Raw central line list/array.
            pixel_index: Linear pixel index for logging context.

        Returns:
            NumPy array of shape (4,).
        """
        if central_line is None:
            logging.warning("Missing central_line at pixel %d; filling with NaNs.", pixel_index)
            return np.full(4, np.nan, dtype=np.float32)
        line_arr = np.asarray(central_line, dtype=np.float32).ravel()
        if line_arr.size < 4:
            logging.warning(
                "central_line length %d < 4 at pixel %d; padding with NaNs.",
                line_arr.size,
                pixel_index,
            )
            line_arr = np.pad(line_arr, (0, 4 - line_arr.size), mode="constant", constant_values=np.nan)
        elif line_arr.size > 4:
            logging.warning(
                "central_line length %d > 4 at pixel %d; truncating.",
                line_arr.size,
                pixel_index,
            )
            line_arr = line_arr[:4]
        return line_arr.astype(np.float32)

    def _coerce_index(
        self,
        value,
        *,
        name: str,
        default: int,
        max_length: int,
        pixel_index: int,
    ) -> int:
        """
        Coerce a band profile index into a safe integer for HDF5 storage.

        Parameters:
            value: Raw index value (int-like) or None.
            name: Name of the index field (for logging).
            default: Default sentinel value to use when missing/invalid.
            max_length: Exclusive upper bound for valid indices.
            pixel_index: Linear pixel index for logging context.

        Returns:
            Integer index value, or ``default`` when invalid.
        """
        if value is None:
            return default
        try:
            index_value = int(value)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid %s=%r at pixel %d; storing %d.",
                name,
                value,
                pixel_index,
                default,
            )
            return default
        if index_value < 0 or index_value >= max_length:
            logging.warning(
                "%s=%d out of range [0, %d) at pixel %d; storing %d.",
                name,
                index_value,
                max_length,
                pixel_index,
                default,
            )
            return default
        return index_value

    def _write_dataset(self, h5file, dataset_path, data, attrs=None):
        """
        Write an HDF5 dataset, replacing any existing dataset.

        Parameters:
            h5file: Open HDF5 file handle.
            dataset_path: Path for the dataset.
            data: Array data to store.
            attrs: Optional attribute dictionary.

        Returns:
            The created HDF5 dataset.
        """
        if dataset_path in h5file:
            del h5file[dataset_path]
        return write_hdf5_dataset(h5file, dataset_path, data, attrs=attrs)

    # ------------------------------------------------------------------
    def export_results(self, processed):
        """Export CSV summaries, write HDF5 outputs, and generate companion ANG output."""
        output_csv_path = self.output_dir / f"{self.base_name}_bandOutputData.csv"
        filtered_csv_path = self.output_dir / f"{self.base_name}_filtered_band_data.csv"
        ut.save_results_to_csv(processed, str(output_csv_path), str(filtered_csv_path))

        df = pd.read_csv(filtered_csv_path)
        required_cols = [
            "Band Width",
            "psnr",
            "efficientlineIntensity",
            "Ind",
            "defficientlineIntensity",
        ]
        for col in required_cols:
            if col not in df.columns:
                logging.error(f"{col} column not found in filtered_band_data.csv.")
                return

        logging.info(
            "Loaded band_width, psnr, defficientlineIntensity, efficientlineIntensity from CSV."
        )

        with h5py.File(self.modified_data_path, "a") as h5file:
            target_dataset_name = next(name for name in h5file if name not in ["Manufacturer", "Version"])
            ci_data = h5file[f"/{target_dataset_name}/EBSD/Data/CI"]

            max_index = df["Ind"].max()
            if max_index >= len(ci_data):
                logging.error("Maximum index in 'Ind' exceeds CI dataset length.")
                return

            profile_length = int(self.config.get("rectWidth", 20) * 4)
            if profile_length <= 0:
                logging.error("Invalid band profile length: %d", profile_length)
                return
            n_pixels = len(ci_data)
            band_profile_array = np.full((n_pixels, profile_length), np.nan, dtype="float32")
            central_line_array = np.full((n_pixels, 4), np.nan, dtype="float32")
            band_start_idx_array = np.full(n_pixels, -1, dtype="int32")
            band_end_idx_array = np.full(n_pixels, -1, dtype="int32")
            central_peak_idx_array = np.full(n_pixels, -1, dtype="int32")
            profile_length_array = np.full(n_pixels, profile_length, dtype="int32")
            band_valid_array = np.zeros(n_pixels, dtype="int8")

            band_width_array = np.zeros_like(ci_data, dtype="float32")
            psnr_array = np.zeros_like(ci_data, dtype="float32")
            efficientIntensity_array = np.zeros_like(ci_data, dtype="float32")
            defficientIntensity_array = np.zeros_like(ci_data, dtype="float32")
            eff_ratio_array = np.zeros_like(ci_data, dtype="float32")

            for idx, bw, psnr, effI, deffI, ratio in zip(
                df["Ind"],
                df["Band Width"],
                df["psnr"],
                df["efficientlineIntensity"],
                df["defficientlineIntensity"],
                df["efficientDefficientRatio"],
            ):
                band_width_array[idx] = bw
                psnr_array[idx] = psnr
                efficientIntensity_array[idx] = effI
                defficientIntensity_array[idx] = deffI
                eff_ratio_array[idx] = ratio

            for entry in processed:
                idx = entry.get("ind")
                if idx is None or idx >= n_pixels:
                    logging.warning("Skipping band profile for invalid index: %s", idx)
                    continue
                best_band = self._select_best_band(entry.get("bands", []), idx)
                if best_band is None:
                    continue
                band_profile_array[idx] = self._coerce_profile(
                    best_band.get("band_profile"), profile_length, idx
                )
                central_line_array[idx] = self._coerce_central_line(
                    best_band.get("central_line"), idx
                )
                band_valid_array[idx] = 1

                expected_len = int(best_band.get("profile_length", profile_length))
                if expected_len != profile_length:
                    logging.warning(
                        "profile_length mismatch at pixel %d: band=%d config=%d; storing config value.",
                        idx,
                        expected_len,
                        profile_length,
                    )

                band_start_idx_array[idx] = self._coerce_index(
                    best_band.get("band_start_idx", best_band.get("bandStart")),
                    name="band_start_idx",
                    default=-1,
                    max_length=profile_length,
                    pixel_index=idx,
                )
                band_end_idx_array[idx] = self._coerce_index(
                    best_band.get("band_end_idx", best_band.get("bandEnd")),
                    name="band_end_idx",
                    default=-1,
                    max_length=profile_length,
                    pixel_index=idx,
                )
                central_peak_idx_array[idx] = self._coerce_index(
                    best_band.get("central_peak_idx", best_band.get("centralPeak")),
                    name="central_peak_idx",
                    default=-1,
                    max_length=profile_length,
                    pixel_index=idx,
                )
                if (
                    band_start_idx_array[idx] != -1
                    and band_end_idx_array[idx] != -1
                    and band_start_idx_array[idx] >= band_end_idx_array[idx]
                ):
                    logging.warning(
                        "band_start_idx (%d) >= band_end_idx (%d) at pixel %d; resetting indices to -1.",
                        band_start_idx_array[idx],
                        band_end_idx_array[idx],
                        idx,
                    )
                    band_start_idx_array[idx] = -1
                    band_end_idx_array[idx] = -1

            desired_ref_width = self.config["desired_hkl_ref_width"]
            if desired_ref_width == 0:
                logging.error("desired_hkl_ref_width is zero; cannot compute strain.")
                return
            band_strain_array = (band_width_array - desired_ref_width) / desired_ref_width
            elastic_modulus = float(self.config["elastic_modulus"])
            band_stress_array = band_strain_array * elastic_modulus

            base_outputs = {
                "Band_Width": band_width_array,
                "psnr": psnr_array,
                "efficientlineIntensity": efficientIntensity_array,
                "defficientlineIntensity": defficientIntensity_array,
                "band_intensity_ratio": eff_ratio_array,
                "strain": band_strain_array,
                "stress": band_stress_array,
            }
            registry = build_default_registry(logger=logging.getLogger(__name__))
            derived_outputs = registry.compute(base_outputs)
            data_root = f"/{target_dataset_name}/EBSD/Data"
            for field_name, data in base_outputs.items():
                write_hdf5_dataset(h5file, f"{data_root}/{field_name}", data)
            for field_name, data in derived_outputs.items():
                spec = registry.get_spec(field_name)
                dataset_name = spec.dataset_name if spec is not None else field_name
                attrs = spec.attrs if spec is not None else None
                write_hdf5_dataset(
                    h5file, f"{data_root}/{dataset_name}", data, attrs=attrs
                )
            self._write_dataset(
                h5file,
                f"{data_root}/band_profile",
                band_profile_array,
                attrs={
                    "description": "Summed band intensity profile (rectWidth*4 samples)",
                    "units": "arb. intensity",
                },
            )
            self._write_dataset(
                h5file,
                f"{data_root}/central_line",
                central_line_array,
                attrs={
                    "description": "Band central line endpoints [x1, y1, x2, y2]",
                    "units": "pixel",
                },
            )
            self._write_dataset(
                h5file,
                f"{data_root}/band_start_idx",
                band_start_idx_array,
                attrs={
                    "description": "Left local minimum index in band_profile used for bandwidth calculation (-1 when unavailable).",
                    "units": "index",
                },
            )
            self._write_dataset(
                h5file,
                f"{data_root}/band_end_idx",
                band_end_idx_array,
                attrs={
                    "description": "Right local minimum index in band_profile used for bandwidth calculation (-1 when unavailable).",
                    "units": "index",
                },
            )
            self._write_dataset(
                h5file,
                f"{data_root}/central_peak_idx",
                central_peak_idx_array,
                attrs={
                    "description": "Central peak index in band_profile used to split minima search (-1 when unavailable).",
                    "units": "index",
                },
            )
            self._write_dataset(
                h5file,
                f"{data_root}/profile_length",
                profile_length_array,
                attrs={
                    "description": "Expected length of band_profile vector (rectWidth*4).",
                    "units": "samples",
                },
            )
            self._write_dataset(
                h5file,
                f"{data_root}/band_valid",
                band_valid_array,
                attrs={
                    "description": "1 when a valid best-band profile was stored for the pixel; 0 otherwise.",
                },
            )
            logging.info(
                "Wrote HDF5 outputs: %s.",
                ", ".join(
                    list(base_outputs.keys())
                    + list(derived_outputs.keys())
                    + [
                        "band_profile",
                        "central_line",
                        "band_start_idx",
                        "band_end_idx",
                        "central_peak_idx",
                        "profile_length",
                        "band_valid",
                    ]
                ),
            )

            logging.info(
                "Wrote Band_Width, strain, stress, psnr, efficient/defficient intensity to HDF5."
            )

        if self.source_format == "ctf":
            ang_template = self.config.get("ang_template_path")
            if ang_template and Path(str(ang_template)).exists():
                ang_output_path = ut.export_ang_with_prias_metrics(
                    original_ang_path=Path(str(ang_template)),
                    modified_h5_path=self.modified_data_path,
                )
            else:
                ang_output_path = export_ctf_with_prias_metrics(self.modified_data_path)
            oh5_output_path = self._export_oh5_copy()
            self._validate_exported_ang(
                ang_output_path,
                reference_path=Path(str(ang_template)) if ang_template else None,
            )
            logging.info(
                "Exported CTF-derived ANG and OH5 outputs while preserving original Euler values: %s, %s.",
                ang_output_path,
                oh5_output_path,
            )
            return

        try:
            ang_output_path = ut.export_ang_with_prias_metrics(
                original_ang_path=self.in_ang_path,
                modified_h5_path=self.modified_data_path,
            )
            logging.info("Exported companion ANG file for TSL at: %s", ang_output_path)
            self._validate_exported_ang(ang_output_path, reference_path=self.in_ang_path)
            oh5_output_path = self._export_oh5_copy()
            logging.info("Exported OH5 copy at: %s", oh5_output_path)
        except Exception:
            logging.exception(
                "Failed to export companion ANG/OH5 files from %s.",
                self.modified_data_path,
            )
            raise

    def _validate_exported_ang(
        self,
        output_path: Path,
        *,
        reference_path: Optional[Path] = None,
    ) -> None:
        """Validate ANG dimensions, columns, rows, and Euler preservation.

        Parameters:
            output_path: Generated ANG file.
            reference_path: Optional source ANG whose Euler columns must match.

        Returns:
            None.

        Raises:
            ValueError: If the ANG structure or Euler values are inconsistent.
        """

        def _parse(path: Path, *, allow_incomplete: bool = False):
            """Parse ANG metadata and numeric rows for validation.

            Parameters:
                path: ANG file path.

            Returns:
                Tuple containing headers, dimensions, and numeric rows.
            """

            lines = Path(path).read_text(encoding="utf-8").splitlines()
            headers = None
            nrows = None
            ncols = None
            rows = []
            for line in lines:
                if line.startswith("# COLUMN_HEADERS:"):
                    headers = [value.strip() for value in line.split(":", 1)[1].split(",")]
                elif line.startswith("# NROWS:"):
                    nrows = int(line.split(":", 1)[1].strip())
                elif line.startswith("# NCOLS_EVEN:"):
                    ncols = int(line.split(":", 1)[1].strip())
                elif line and not line.startswith("#"):
                    rows.append([float(value) for value in line.split()])
            if headers is None or nrows is None or ncols is None:
                raise ValueError(f"ANG header is incomplete: {path}")
            if len(rows) != nrows * ncols and not allow_incomplete:
                raise ValueError(
                    f"ANG row count {len(rows)} does not match NROWS*NCOLS_EVEN={nrows*ncols}: {path}"
                )
            if any(len(row) != len(headers) for row in rows):
                raise ValueError(f"ANG data column count does not match COLUMN_HEADERS: {path}")
            return headers, nrows, ncols, np.asarray(rows, dtype=np.float64)

        headers, nrows, ncols, rows = _parse(Path(output_path))
        if reference_path is not None and Path(reference_path).exists():
            ref_headers, ref_nrows, ref_ncols, ref_rows = _parse(
                Path(reference_path),
                allow_incomplete=True,
            )
            if (nrows, ncols) != (ref_nrows, ref_ncols):
                raise ValueError("Generated ANG dimensions differ from the source ANG template.")
            normalized = [name.strip().lower() for name in headers]
            ref_normalized = [name.strip().lower() for name in ref_headers]
            euler_names = ("phi1", "phi", "phi2")
            output_has_eulers = [name in normalized for name in euler_names]
            reference_has_eulers = [name in ref_normalized for name in euler_names]
            if any(output_has_eulers + reference_has_eulers) and not all(
                output_has_eulers + reference_has_eulers
            ):
                raise ValueError("ANG output/template contains an incomplete Euler column triplet.")
            for name in euler_names if all(output_has_eulers + reference_has_eulers) else ():
                actual = rows[: ref_rows.shape[0], normalized.index(name)]
                expected = ref_rows[:, ref_normalized.index(name)]
                if not np.array_equal(actual, expected):
                    raise ValueError(f"Generated ANG modified original Euler column '{name}'.")
        logging.info(
            "Validated ANG structure and original Euler preservation: %s (%d x %d, %d columns).",
            output_path,
            ncols,
            nrows,
            len(headers),
        )

    def _export_oh5_copy(self) -> Path:
        """Write an OH5-named copy of the modified HDF5 output.

        Returns:
            Path to the OH5 copy.
        """

        output_path = self.modified_data_path.with_suffix(".oh5")
        if output_path.resolve() == self.modified_data_path.resolve():
            return output_path
        shutil.copy2(self.modified_data_path, output_path)
        return output_path

    # ------------------------------------------------------------------
    def run(self):
        """Execute the complete pipeline."""

        start_time = time.time()
        self.prepare_dataset()
        self.simulate_and_index()
        processed = self.detect_band_widths()
        self.export_results(processed)
        logging.info("Process completed. Results saved to CSV files and modified .ang file.")
        logging.info(f"Total processing time: {time.time() - start_time:.1f} s")


# ---------------------------------------------------------------------- #
#                               main()
# ---------------------------------------------------------------------- #
def main():
    """Run the band-width automator from command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Kikuchi band-width analysis from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        default="bandDetectorOptionsHcp.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force DEBUG logging for this run.",
    )
    args = parser.parse_args()
    from kikuchiBandAnalyzer.band_width.config import load_band_width_config

    config = load_band_width_config(args.config)
    if args.debug or bool(config.get("debug", False)):
        logging.getLogger().setLevel(logging.DEBUG)
    bwa = BandWidthAutomator(config_path=args.config)
    bwa.run()
if __name__ == "__main__":
    main()
