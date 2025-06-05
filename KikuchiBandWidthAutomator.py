import time
import os
import yaml
import logging

import matplotlib.pyplot as plt
import kikuchipy as kp
import re
from orix import plot
from diffsims.crystallography import ReciprocalLatticeVector
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase, PhaseList
from orix.vector import Vector3d
from typing import Optional
import numpy as np
#from hyperspy.utils.markers import line_segment, point, text
import pandas as pd
from kikuchiBandWidthDetector import process_kikuchi_images
from kikuchiBandWidthDetector import save_results_to_excel
import shutil
import h5py
import utilities as ut
from custom_simulation import (
    CustomGeometricalKikuchiPatternSimulation,
    CustomKikuchiPatternSimulator,
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(file_path="bandDetectorOptions.yml"):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load configuration
    start_time = time.time()
    # config = load_config(file_path="bandDetectorOptions.yml")
    config = load_config(file_path="bandDetectorOptionsDebug.yml")
    #config = load_config(file_path="bandDetectorOptionsHcp.yml")
    # config = load_config(file_path="bandDetectorOptionsMagnetite.yml")
    # config = load_config(file_path="bandDetectorOptionsHeamatite.yml")

    data_path = config.get("h5_file_path", "path_to_default_file.h5")
    output_dir = os.path.dirname(data_path)
    base_name, ext = os.path.splitext(os.path.basename(data_path))

    # Check if the input file ends with .oh5 and create a .h5 copy
    if ext == ".oh5":
        new_data_path = os.path.join(output_dir, f"{base_name}.h5")
        shutil.copy(data_path, new_data_path)
        logging.info(f"Copied .oh5 file to new .h5 file: {new_data_path}")
        data_path = new_data_path  # Update data_path to point to the new .h5 file

    temp_data_path = ut.create_temp_file(data_path)
    modified_data_path = os.path.join(output_dir, f"{base_name}_modified.h5")

    in_ang_path = os.path.join(output_dir, f"{base_name}.ang")
    out_ang_path = os.path.join(output_dir, f"{base_name}_modified.ang")
    shutil.copy(data_path, modified_data_path)
    logging.info(f"Copied HDF5 file to: {modified_data_path}")

    crop_start, crop_end = config.get("crop_start", 5), config.get("crop_end", 25)
    debug = config.get("debug", False)
    desired_hkl = config.get("desired_hkl", "111")

    # Load dataset and optional cropping based on debug flag
    logging.info(f"Loading dataset from: {data_path}")
    s = kp.load(data_path, lazy=False)

    if debug:
        logging.info("Debug mode enabled: Cropping data for faster processing.")
        s.crop(1, start=crop_start, end=crop_end + 10)
        s.crop(0, start=crop_start, end=crop_end)

    # Setup detector and phase list based on material properties
    phase_config = config["phase_list"]
    phase_list = PhaseList(
        Phase(
            name=phase_config["name"],
            space_group=phase_config["space_group"],
            structure=Structure(
                lattice=Lattice(*phase_config["lattice"]),
                atoms=[
                    Atom(atom["element"], atom["position"])
                    for atom in phase_config["atoms"]
                ],
            ),
        ),
    )
    hkl_list = config["hkl_list"]
    pc = config.get("pc", [0.545, 0.610, 0.6863])
    header_data = ut.extract_header_data(modified_data_path)

    # Detector setup and pattern extraction
    sig_shape = s.axes_manager.signal_shape[::-1]
    det = kp.detectors.EBSDDetector(
        sig_shape,
        sample_tilt=float(header_data.get("Sample Tilt", 0.0)),
        tilt=float(header_data.get("Camera Elevation Angle", 0.0)),
        azimuthal=float(header_data.get("Camera Azimuthal Angle", 0.0)),
        convention="edax",
        pc=tuple(header_data.get("pc", (0.0, 0.0, 0.0)))
    )

    # det = kp.detectors.EBSDDetector(sig_shape,
    #                                 sample_tilt=float(header_data.get("Sample Tilt", 0.0)),
    #                                 camera_tilt=float(header_data.get("Camera Elevation Angle", 0.0)),
    #                                 azimuthal_angle=float(header_data.get("Camera Azimuthal Angle", 0.0))
    #                                 convention="edax", pc=header_data.get("pc",(0.,0.,0.)))

    # PhaseList and indexing
    indexer = det.get_indexer(phase_list, hkl_list, nBands=10, tSigma=2, rSigma=2)
    xmap, index_data, indexed_band_data = s.hough_indexing(
        phase_list=phase_list, indexer=indexer, return_index_data=True, return_band_data=True, verbose=1
    )

    # Use the CustomKikuchiPatternSimulator for geometrical simulations
    ref = ReciprocalLatticeVector(phase=xmap.phases[0], hkl=hkl_list)
    ref = ref.symmetrise()
    simulator = CustomKikuchiPatternSimulator(ref)  # Use the custom simulator
    sim = simulator.on_detector(det, xmap.rotations.reshape(*xmap.shape))

    # Add markers and Kikuchi line labels to the signal
    markers, grouped_kikuchi_dict_list = sim.as_markers(kikuchi_line_labels=True, desired_hkl=config["desired_hkl"])
    s.add_marker(markers, plot_marker=False, permanent=True)
    logging.info("Completed band identification for width estimation. Now starting the band width estimation!")

    skip_display_EBSDmap = config.get("skip_display_EBSDmap", False)
    if not skip_display_EBSDmap:
        v_ipf = Vector3d.xvector()
        sym = xmap.phases[0].point_group
        ckey = plot.IPFColorKeyTSL(sym, v_ipf)
        rgb_x = ckey.orientation2color(xmap.rotations)
        maps_nav_rgb = kp.draw.get_rgb_navigator(rgb_x.reshape(xmap.shape + (3,)))
        s.plot(maps_nav_rgb)
        plt.show()

    # Call the process_kikuchi_images function
    ebsd_data = s.data  # EBSD dataset where each (row, col) contains the Kikuchi pattern (2D numpy array)
    processed_results = process_kikuchi_images(ebsd_data, grouped_kikuchi_dict_list, desired_hkl=desired_hkl,
                                               config=config)
    output_excel_path = os.path.join(output_dir, f"{base_name}_bandOutputData.xlsx")
    filtered_excel_path = os.path.join(output_dir, f"{base_name}_filtered_band_data.xlsx")
    save_results_to_excel(processed_results, output_excel_path, filtered_excel_path)

    # Modified section of your calling code after reading Excel:
    df = pd.read_excel(filtered_excel_path)

    # Check columns
    required_columns = ["Band Width", "psnr", "efficientlineIntensity", "Ind", "defficientlineIntensity"]
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"{col} column not found in filtered_band_data.xlsx.")
            return
    logging.info("Loaded band_width, psnr,defficientlineIntensity and efficientlineIntensity data from Excel.")

    # Open the copied HDF5 file and add the data nodes
    with h5py.File(modified_data_path, "a") as h5file:
        target_dataset_name = next(name for name in h5file if name not in ["Manufacturer", "Version"])

        ci_data = h5file[f"/{target_dataset_name}/EBSD/Data/CI"]

        max_index = df["Ind"].max()
        if max_index >= len(ci_data):
            logging.error("Maximum index in 'Ind' exceeds length of /Nickel/EBSD/Data/CI.")
            return

        # Create zero-initialized arrays
        band_width_array = np.zeros_like(ci_data, dtype="float32")
        psnr_array = np.zeros_like(ci_data, dtype="float32")
        efficientlineIntensity_array = np.zeros_like(ci_data, dtype="float32")  # Newly added array
        defficientlineIntensity_array = np.zeros_like(ci_data, dtype="float32")  # Newly added array
        efficientDefficientRatio_array = np.zeros_like(ci_data, dtype="float32")  # Newly added array

        # Populate arrays
        for idx, band_width, psnr, defficientlineIntensity, efficientlineIntensity, efficientDefficientRatio in zip(
                df["Ind"], df["Band Width"], df["psnr"],
                df["efficientlineIntensity"], df["defficientlineIntensity"], df["efficientDefficientRatio"]):
            band_width_array[idx] = band_width
            psnr_array[idx] = psnr
            # efficientDefficientRatio_array=efficientDefficientRatio
            efficientlineIntensity_array[idx] = efficientlineIntensity  # New line here
            defficientlineIntensity_array[idx] = defficientlineIntensity  # New line here
            efficientDefficientRatio_array[idx] = efficientDefficientRatio  # New line here

        desired_hkl_ref_width = config["desired_hkl_ref_width"]
        band_strain_array = (band_width_array - desired_hkl_ref_width) / desired_hkl_ref_width
        elastic_modulus = float(config["elastic_modulus"])
        band_stress_array = band_strain_array * elastic_modulus

        angInputDict = {
            "IQ": band_width_array,
            "PRIAS_Bottom_Strip": band_strain_array,
            "PRIAS_Center_Square": band_stress_array,
            "PRIAS_Top_Strip": psnr_array,
            "efficientlineIntensity": efficientlineIntensity_array,  # Included here if needed
            "defficientlineIntensity": efficientlineIntensity_array,  # Included here if needed
            "efficientDefficientRatio": efficientDefficientRatio_array  # Included here if needed
        }

        # Modify the .ang file for efficientlineIntensity as well
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_band_width", IQ=band_width_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_strain", IQ=band_strain_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_stress", IQ=band_stress_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_psnr", IQ=psnr_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_defficientlineIntensity",
                           IQ=defficientlineIntensity_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_efficientlineIntensity",
                           IQ=efficientlineIntensity_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_efficientDefficientRatio",
                           IQ=efficientDefficientRatio_array)
        ut.modify_ang_file(in_ang_path, f"{desired_hkl}_Bandwidth_efficientDefficientRatio",
                           IQ=band_width_array, Fit=efficientDefficientRatio_array)

        # New addition

        logging.info(
            "Successfully wrote band_width, strain, stress, psnr, defficientlineIntensity and efficientlineIntensity in modified ang file!")

        # Store datasets in HDF5
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/Band_Width", data=band_width_array)
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/psnr", data=psnr_array)
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/efficientlineIntensity",
                              data=efficientlineIntensity_array)  # New dataset
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/defficientlineIntensity",
                              data=defficientlineIntensity_array)  # New dataset
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/strain", data=band_strain_array)
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/stress", data=band_stress_array)

        logging.info("Added Band_Width, psnr, defficientlineIntensity, efficientlineIntensity data to HDF5 file.")

    logging.info("Process completed. Results saved to Excel files and modified .ang file.")

    end_time = time.time()  # End timing
    logging.info(f"The total processing time is : {end_time - start_time} seconds")

    # Optional visualization


if __name__ == "__main__":
    main()



