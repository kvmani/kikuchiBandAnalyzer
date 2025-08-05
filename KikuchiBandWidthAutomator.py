
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

    def __init__(self, config_path: str = "bandDetectorOptionsDebug.yml"):
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
    def detect_band_widths(self):
        """Run the :class:`KikuchiBatchProcessor` over all patterns."""
        desired_hkl = self.config.get("desired_hkl", "1,1,1")
        ebsd_data = self.dataset.data
        processor = KikuchiBatchProcessor(
            ebsd_data,
            self.grouped_dict_list,
            config=self.config,
            desired_hkl=desired_hkl,
        )
        return processor.process()

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

            desired_ref_width = self.config["desired_hkl_ref_width"]
            band_strain_array = (band_width_array - desired_ref_width) / desired_ref_width
            elastic_modulus = float(self.config["elastic_modulus"])
            band_stress_array = band_strain_array * elastic_modulus

            ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_band_width", IQ=band_width_array)
            ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_strain", IQ=band_strain_array)
            ut.modify_ang_file(self.in_ang_path, f"{desired_hkl}_stress", IQ=band_stress_array)
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

            h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/Band_Width", data=band_width_array)
            h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/psnr", data=psnr_array)
            h5file.create_dataset(
                f"/{target_dataset_name}/EBSD/Data/efficientlineIntensity",
                data=efficientIntensity_array,
            )
            h5file.create_dataset(
                f"/{target_dataset_name}/EBSD/Data/defficientlineIntensity",
                data=defficientIntensity_array,
            )
            h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/strain", data=band_strain_array)
            h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/stress", data=band_stress_array)

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