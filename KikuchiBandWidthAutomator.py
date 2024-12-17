import time
import warnings, os
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
from hyperspy.utils.markers import line_segment, point, text
import pandas as pd
from collections import defaultdict
import json
from kikuchiBandWidthDetector import process_kikuchi_images
from kikuchiBandWidthDetector import save_results_to_excel
import shutil
import h5py
import utilities as ut

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path="bandDetectorOptions.yml"):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


GeometricalKikuchiPatternSimulation = kp.simulations.GeometricalKikuchiPatternSimulation
KikuchiPatternSimulator = kp.simulations.KikuchiPatternSimulator
# Custom class as defined previously
class CustomGeometricalKikuchiPatternSimulation(GeometricalKikuchiPatternSimulation):
    from collections import defaultdict


    def _group_by_ind(self, data):
        """
        Groups a list of dictionaries by the 'ind' field and consolidates related fields.

        Parameters:
        - data (list): A list of dictionaries containing fields '(x,y)', 'ind', 'hkl', 'central_line', 'line_mid_xy', and 'line_dist'.

        Returns:
        - result (list): A list of dictionaries, where each dictionary is grouped by 'ind' and includes a single '(x,y)' and a list of 'points' for each 'hkl'.
        """
        grouped_data = defaultdict(lambda: {"x,y": None, "ind": None, "points": []})

        # Loop through the data to group by "ind"
        for entry in data:
            ind = entry["ind"]

            # Set (x,y) for the first entry of each group
            if grouped_data[ind]["x,y"] is None:
                grouped_data[ind]["x,y"] = entry["(x,y)"]
                grouped_data[ind]["ind"] = ind

            # Append the current point's details to the "points" list
            grouped_data[ind]["points"].append({
                "hkl": entry["hkl"],
                "hkl_group": entry["hkl_group"],
                "central_line": entry["central_line"],
                "line_mid_xy": entry["line_mid_xy"],
                "line_dist": entry["line_dist"]
            })

        # Convert defaultdict back to a regular list of dictionaries
        return list(grouped_data.values())

    def _df_to_grouped_json(self,df):
        # Convert DataFrame to JSON string
        json_str = df.to_json(orient="records")

        # Load JSON string into Python data structure (list of dictionaries)
        data = json.loads(json_str)

        # Apply the grouping method
        grouped_data = self._group_by_ind(data)

        # Return grouped JSON as a string
        return json.dumps(grouped_data, indent=4), grouped_data


    @staticmethod
    def remove_newlines_from_fields(json_str):
        """
        Removes newline characters from the "central_line" and "x,y" fields in the JSON string,
        ensuring that only single spaces remain between elements in the arrays without duplicating the content.

        Parameters:
        - json_str (str): The JSON string to process.

        Returns:
        - str: The processed JSON string with newlines removed from "central_line" and "x,y" fields.
        """
        # Regex to match "central_line" and "x,y" fields with values inside []
        central_line_pattern = r'("central_line":\s*\[[^\]]*\])'
        line_mid_xy_pattern = r'("line_mid_xy":\s*\[[^\]]*\])'
        xy_pattern = r'("x,y":\s*\[[^\]]*\])'

        # Function to clean up excess spaces and ensure arrays are on one line
        def cleanup_array(match):
            # Extract the field and the array
            field, array_content = match.group(0).split(":")
            # Remove newlines and excess spaces within the array
            clean_array_content = re.sub(r'\s+', ' ', array_content).replace('[ ', '[').replace(' ]', ']')
            return f'{field}: {clean_array_content}'

        # Replace newlines and excess spaces in "central_line" field
        json_str = re.sub(central_line_pattern, cleanup_array, json_str, flags=re.DOTALL)
        # Replace newlines and excess spaces in "line_mid_xy" field
        json_str = re.sub(line_mid_xy_pattern, cleanup_array, json_str, flags=re.DOTALL)
        # Replace newlines and excess spaces in "x,y" field
        json_str = re.sub(xy_pattern, cleanup_array, json_str, flags=re.DOTALL)

        return json_str

    @staticmethod
    def get_hkl_group(hkl_str):
        # Split string into integers, apply abs, sort, convert back to string without spaces
        hkl_values = sorted(map(abs, map(int, hkl_str.split())))
        return ''.join(map(str, hkl_values))
    def _kikuchi_line_labels_as_markers(self, desired_hkl='111', **kwargs) -> list:
        """Return a list of Kikuchi line label text markers."""
        coords = self.lines_coordinates(index=(), exclude_nan=False)
        coords=np.around(coords,3)
        # Labels for Kikuchi lines can be based on reflectors (e.g., hkl values)
        reflectors = self._reflectors.coordinates.round().astype(int)
        array_str = np.array2string(reflectors, threshold=reflectors.size)
        texts = re.sub("[][ ]", " ", array_str).split("\n")

        filtered_texts = np.array([text if self.get_hkl_group(text) == desired_hkl else '' for text in texts])

        #refelctorGroup  = [''.join(map(str, [abs(i.h), abs(i.k), abs(i.l)])) for i in reflectors]
        refelctorGroup = [''.join(map(str, sorted(map(abs, row)))) for row in reflectors]

        kw = {
            "color": "b",
            "zorder": 4,
            "ha": "center",
            "va": "bottom",
            "bbox": {"fc": "w", "ec": "b", "boxstyle": "round,pad=0.3"},
        }
        kw.update(kwargs)
        kikuchi_line_label_list = []
        # Initialize a 2D list to store dictionaries at each (row, col) position
        rows, cols, n, _ = coords.shape
        num_cols = cols
        kikuchi_line_dict_list = [[[] for _ in range(cols)] for _ in range(rows)]
        is_finite = np.isfinite(coords)[..., 0]
        coords[~is_finite] = -1
        det_bounds = self.detector.shape
        det_mid_point=0.5*det_bounds[0],0.5*det_bounds[1]
        dist_threshold = 0.75*0.5*det_bounds[0] ## 75% of radius of the detector
        for i in range(reflectors.shape[0]):
            if not np.allclose(coords[..., i, :], -1):  # Check for all NaNs
                x1 = coords[..., i, 0]
                y1 = coords[..., i, 1]
                x2 = coords[..., i, 2]
                y2 = coords[..., i, 3]
                x = 0.5 * (x1 + x2)
                y = 0.5 * (y1 + y2)

                line_dist = np.sqrt((x- det_mid_point[0])**2+(y- det_mid_point[1])**2)

                x[~is_finite[..., i]] = np.nan
                y[~is_finite[..., i]] = np.nan
                x = x.squeeze()
                y = y.squeeze()

                # Create a text marker with the label for each Kikuchi line
                text_marker = text(x=x, y=y, text=filtered_texts[i], **kw)
                kikuchi_line_label_list.append(text_marker)

                # Vectorized approach to fill kikuchi_line_dict_list
                valid_mask = is_finite[..., i]  # Get the mask of finite values
                if np.any(valid_mask):  # Check if there are any valid points
                    # Create indices of valid points
                    row_indices, col_indices = np.where(valid_mask)
                    # Prepare the data for valid points
                    num_valid = row_indices.size
                    kikuchi_data = {
                        "hkl": texts[i].strip(),
                        "hkl_group":refelctorGroup[i],
                        #"hkl_value":reflectors[i],
                        "central_line": np.vstack(
                            [x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask]]).T.tolist(),
                        "line_mid_xy": np.vstack([x[valid_mask], y[valid_mask]]).T.tolist(),  # Midpoint coordinates
                        "line_dist": line_dist[valid_mask].T.tolist(),  # line distance coordinates

                    }

                    # Append data to the corresponding positions in the 2D list
                    for idx in range(num_valid):
                        row, col = row_indices[idx], col_indices[idx]
                        if kikuchi_data["line_dist"][idx]<dist_threshold:
                            kikuchi_line_dict_list[row][col].append({
                                "(x,y)":(row,col),
                                "ind": (row * num_cols) + col,
                                "hkl": kikuchi_data["hkl"],
                                "hkl_group":kikuchi_data["hkl_group"],
                                "central_line": kikuchi_data["central_line"][idx],
                                "line_mid_xy": kikuchi_data["line_mid_xy"][idx],
                                "line_dist": kikuchi_data["line_dist"][idx]
                            })


        # Convert to DataFrame
        flat_kikuchi_line_dict_list = []
        for sublist in kikuchi_line_dict_list:
            for entry in sublist:
                if entry:  # Ensure the entry is not empty
                    flat_kikuchi_line_dict_list.extend(entry)

        df = pd.DataFrame(flat_kikuchi_line_dict_list)
        final_json_str, grouped_kikuchi_dict_list = self._df_to_grouped_json(df)

        cleaned_json_str = CustomGeometricalKikuchiPatternSimulation.remove_newlines_from_fields(final_json_str)
        # Dump the final JSON string to disk
        with open("kikuchi_lines.json", "w") as f:
            f.write(cleaned_json_str)

        # Save as JSON
        #df.to_json("kikuchi_lines.json", orient="records", indent=4)


        # Save as Excel
        df.to_excel("kikuchi_lines.xlsx", index=False, engine="openpyxl")

        return kikuchi_line_label_list, grouped_kikuchi_dict_list

    def as_markers(
            self,
            lines: bool = True,
            zone_axes: bool = False,
            zone_axes_labels: bool = False,
            kikuchi_line_labels: bool = False,  # New flag for Kikuchi line labels
            pc: bool = False,
            lines_kwargs: Optional[dict] = None,
            zone_axes_kwargs: Optional[dict] = None,
            zone_axes_labels_kwargs: Optional[dict] = None,
            kikuchi_line_labels_kwargs: Optional[dict] = None,  # kwargs for Kikuchi line labels,
            desired_hkl='111',
            pc_kwargs: Optional[dict] = None,
    ) -> list:
        """Return a list of simulation markers, including Kikuchi line labels."""
        markers = super().as_markers(
            lines=lines,
            zone_axes=zone_axes,
            zone_axes_labels=zone_axes_labels,
            pc=pc,
            lines_kwargs=lines_kwargs,
            zone_axes_kwargs=zone_axes_kwargs,
            zone_axes_labels_kwargs=zone_axes_labels_kwargs,
            pc_kwargs=pc_kwargs,
        )

        if kikuchi_line_labels:
            if kikuchi_line_labels_kwargs is None:
                kikuchi_line_labels_kwargs = {}
            markersTmp,  grouped_kikuchi_dict_list = self._kikuchi_line_labels_as_markers(desired_hkl=desired_hkl, **kikuchi_line_labels_kwargs)
            markers+=markersTmp


        return markers, grouped_kikuchi_dict_list


# Main function for testing the custom class
class CustomKikuchiPatternSimulator(KikuchiPatternSimulator):
    def on_detector(self, detector, rotations):
        # Get reflectors from the instance
        reflectors = self.reflectors

        # Call the base class method `on_detector`, which internally processes the lines and zone_axes
        # Let the base class handle this part
        result = super().on_detector(detector, rotations)

        # Now, return the custom GeometricalKikuchiPatternSimulation using the same components
        return CustomGeometricalKikuchiPatternSimulation(
            detector, rotations, result.reflectors, result._lines, result._zone_axes
        )
# Use this CustomKikuchiPatternSimulator in the main method


def main():
    # Load configuration
    start_time = time.time()
    config = load_config(file_path="bandDetectorOptions.yml")
    #config = load_config(file_path="bandDetectorOptionsMagnetite.yml")
    #config = load_config(file_path="bandDetectorOptionsHeamatite.yml")
    
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
        s.crop(1, start=crop_start, end=crop_end+10)
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
    
    v_ipf = Vector3d.xvector()
    sym = xmap.phases[0].point_group
    ckey = plot.IPFColorKeyTSL(sym, v_ipf)
    rgb_x = ckey.orientation2color(xmap.rotations)
    maps_nav_rgb = kp.draw.get_rgb_navigator(rgb_x.reshape(xmap.shape + (3,)))
    s.plot(maps_nav_rgb)
    plt.show()

    exit(-200)


    # Call the process_kikuchi_images function
    ebsd_data = s.data  # EBSD dataset where each (row, col) contains the Kikuchi pattern (2D numpy array)
    processed_results = process_kikuchi_images(ebsd_data, grouped_kikuchi_dict_list, desired_hkl=desired_hkl, config=config)
    output_excel_path = os.path.join(output_dir, f"{base_name}_bandOutputData.xlsx")
    filtered_excel_path = os.path.join(output_dir, f"{base_name}_filtered_band_data.xlsx")
    save_results_to_excel(processed_results, output_excel_path,filtered_excel_path)

    df = pd.read_excel(filtered_excel_path)
    if "Band Width" not in df.columns:
        logging.error("Band Width column not found in filtered_band_data.xlsx.")
        return
    logging.info("Loaded band_width and psnr data from Excel.")

    # Open the copied HDF5 file and add the Band_Width data node
    with h5py.File(modified_data_path, "a") as h5file:
        target_dataset_name = next(name for name in h5file if name not in ["Manufacturer", "Version"])

        ci_data = h5file[f"/{target_dataset_name}/EBSD/Data/CI"]

        # Sanity check: Ensure the maximum index in the Ind column does not exceed the length of CI data
        max_index = df["Ind"].max()
        if max_index >= len(ci_data):
            logging.error("Maximum index in 'Ind' column exceeds length of /Nickel/EBSD/Data/CI.")
            return

        # Create a zero-initialized array of the same length as CI data for Band_Width
        band_width_array = np.zeros_like(ci_data, dtype="float32")
        psnr_array = np.zeros_like(ci_data, dtype="float32")


        # Fill band_width_array at positions specified by the Ind column in the Excel data
        for idx, band_width, psnr in zip(df["Ind"], df["Band Width"], df["psnr"]):
            band_width_array[idx] = band_width
            psnr_array[idx] = psnr

        # Create the new dataset for Band_Width with the filled values
        desired_hkl_ref_width = config["desired_hkl_ref_width"]
        band_strain_array = (band_width_array - desired_hkl_ref_width) / desired_hkl_ref_width
        elastic_modulus = float(config["elastic_modulus"])
        band_stress_array = band_strain_array * elastic_modulus
        angInputDict = {"IQ":band_width_array, "PRIAS_Bottom_Strip":band_strain_array,
                        "PRIAS_Center_Square":band_stress_array,
                        "PRIAS_Top_Strip":psnr_array,
                        }

        ut.modify_ang_file(in_ang_path, "band_width", IQ=band_width_array)
        ut.modify_ang_file(in_ang_path, "strain", IQ=band_strain_array)
        ut.modify_ang_file(in_ang_path, "stress", IQ=band_stress_array)
        ut.modify_ang_file(in_ang_path, "psnr", IQ=psnr_array)

        logging.info("Succesfully wrote band_width in IQ, strain in PRIAS_Bottom_Strip, stress in PRIAS_Center_Square, psnr in PRIAS_Top_Strip of modified ang file!!")
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/Band_Width", data=band_width_array)
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/psnr", data=psnr_array)
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/strain", data=band_strain_array)
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/stress", data=band_stress_array)

        logging.info("Added Band_Width data to /Nickel/EBSD/Data/Band_Width in modified HDF5 file.")
    logging.info("Process completed. Results saved to Excel files and modified .ang file.")

    end_time = time.time()  # End timing
    logging.info(f"The total processing time is : {end_time - start_time} seconds")

    # Optional visualization

    

if __name__ == "__main__":
    main()



