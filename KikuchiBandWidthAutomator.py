
"""Automation entry point for batch Kikuchi band width detection.

This script wraps the lower level detection utilities and provides a
configuration driven pipeline.  The :class:`BandWidthAutomator` class loads
an EBSD data set, simulates Kikuchi patterns, performs band width detection
and finally exports all results.  It is intended to be run either in a normal
mode, reading all options from a YAML configuration file, or in a debug mode
where the data set is cropped and detailed logging is enabled.
"""

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
from orix.vector import Vector3d
from typing import Optional
import numpy as np

import pandas as pd
import json
from kikuchiBandWidthDetector import KikuchiBatchProcessor
import shutil
import h5py
import utilities as ut
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
        self.data_path = Path(self.config.get("h5_file_path", "path_to_default_file.h5"))
        self.output_dir = self.data_path.parent
        self.base_name = self.data_path.stem
        self.modified_data_path = self.output_dir / f"{self.base_name}_modified.h5"
        self.in_ang_path = self.output_dir / f"{self.base_name}.ang"
        self.dataset = None
        self.grouped_dict_list = None
        logging.info('Justc ompleted the object initiation')

    # ------------------------------------------------------------------
    def prepare_dataset(self):
        """Load the EBSD data set and optionally crop for debug mode."""

        path = self.data_path
        if path.suffix == ".oh5":
            new_data_path = path.with_suffix(".h5")
            shutil.copy(path, new_data_path)
            logging.info(f"Copied .oh5 file to new .h5 file: {new_data_path}")
            path = new_data_path
            self.data_path = new_data_path

        shutil.copy(path, self.modified_data_path)
        logging.info(f"Copied HDF5 file to: {self.modified_data_path}")

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
        """Simulate Kikuchi patterns and determine band locations."""
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
        det = kp.detectors.EBSDDetector(
            sig_shape,
            sample_tilt=float(header_data.get("Sample Tilt", 0.0)),
            tilt=float(header_data.get("Camera Elevation Angle", 0.0)),
            azimuthal=float(header_data.get("Camera Azimuthal Angle", 0.0)),
            convention="edax",
            pc=tuple(header_data.get("pc", (0.0, 0.0, 0.0))),
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

        desired_hkl = self.config.get("desired_hkl", "1,1,1")
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
        desired_hkl = self.config.get("desired_hkl", "1,1,1")
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
        """Export CSV summaries and write results back into the HDF5 file."""
        output_csv_path = self.output_dir / f"{self.base_name}_bandOutputData.csv"
        filtered_csv_path = self.output_dir / f"{self.base_name}_filtered_band_data.csv"
        ut.save_results_to_csv(processed, str(output_csv_path), str(filtered_csv_path))

        desired_hkl = self.config.get("desired_hkl", "1,1,1")

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

            ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_band_width", IQ=band_width_array)
            ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_eff_deff_ratio", IQ=eff_ratio_array)
            #ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_stress", IQ=band_stress_array)
            ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_psnr", IQ=psnr_array)
            ut.modify_ang_file(
                self.in_ang_path,
                f"{desired_hkl}_defficientlineIntensity",
                IQ=defficientIntensity_array,
            )
            ut.modify_ang_file(
                self.in_ang_path,
                f"{desired_hkl}_efficientlineIntensity",
                IQ=efficientIntensity_array,
            )
            ut.modify_ang_file(
                self.in_ang_path,
                f"{desired_hkl}_efficientDefficientRatio",
                IQ=eff_ratio_array,
            )
            ut.modify_ang_file(
                self.in_ang_path,
                f"{desired_hkl}_Bandwidth_efficientDefficientRatio",
                IQ=band_width_array,
                Fit=eff_ratio_array,
            )

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
    """Convenience wrapper for command line execution."""

    bwa = BandWidthAutomator()
    bwa.run()
if __name__ == "__main__":
    main()
