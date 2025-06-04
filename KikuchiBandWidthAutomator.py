import time
import os
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
from hyperspy.utils.markers import text
import pandas as pd
from kikuchiBandWidthDetector import process_kikuchi_images
from kikuchiBandWidthDetector import save_results_to_excel
import shutil
import h5py
import utilities as ut
import configLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


GeometricalKikuchiPatternSimulation = kp.simulations.GeometricalKikuchiPatternSimulation
KikuchiPatternSimulator = kp.simulations.KikuchiPatternSimulator


# Custom class as defined previously
class CustomGeometricalKikuchiPatternSimulation(GeometricalKikuchiPatternSimulation):

    def _kikuchi_line_labels_as_markers(self, desired_hkl='111', **kwargs) -> list:
        """Return a list of Kikuchi line label text markers."""
        coords = self.lines_coordinates(index=(), exclude_nan=False)
        coords = np.around(coords, 3)
        # Labels for Kikuchi lines can be based on reflectors (e.g., hkl values)
        reflectors = self._reflectors.coordinates.round().astype(int)
        array_str = np.array2string(reflectors, threshold=reflectors.size)
        texts = re.sub("[][ ]", " ", array_str).split("\n")

        filtered_texts = np.array([text if ut.get_hkl_group(text) == desired_hkl else '' for text in texts])

        # refelctorGroup  = [''.join(map(str, [abs(i.h), abs(i.k), abs(i.l)])) for i in reflectors]
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
        det_mid_point = 0.5 * det_bounds[0], 0.5 * det_bounds[1]
        dist_threshold = 0.75 * 0.5 * det_bounds[0]  ## 75% of radius of the detector
        for i in range(reflectors.shape[0]):
            if not np.allclose(coords[..., i, :], -1):  # Check for all NaNs
                x1 = coords[..., i, 0]
                y1 = coords[..., i, 1]
                x2 = coords[..., i, 2]
                y2 = coords[..., i, 3]
                x = 0.5 * (x1 + x2)
                y = 0.5 * (y1 + y2)

                line_dist = np.sqrt((x - det_mid_point[0]) ** 2 + (y - det_mid_point[1]) ** 2)

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
                        "hkl_group": refelctorGroup[i],
                        # "hkl_value":reflectors[i],
                        "central_line": np.vstack(
                            [x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask]]).T.tolist(),
                        "line_mid_xy": np.vstack([x[valid_mask], y[valid_mask]]).T.tolist(),  # Midpoint coordinates
                        "line_dist": line_dist[valid_mask].T.tolist(),  # line distance coordinates

                    }

                    # Append data to the corresponding positions in the 2D list
                    for idx in range(num_valid):
                        row, col = row_indices[idx], col_indices[idx]
                        if kikuchi_data["line_dist"][idx] < dist_threshold:
                            kikuchi_line_dict_list[row][col].append({
                                "(x,y)": (row, col),
                                "ind": (row * num_cols) + col,
                                "hkl": kikuchi_data["hkl"],
                                "hkl_group": kikuchi_data["hkl_group"],
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
        final_json_str, grouped_kikuchi_dict_list = ut.df_to_grouped_json(df)

        cleaned_json_str = ut.remove_newlines_from_fields(final_json_str)
        # Dump the final JSON string to disk
        with open("kikuchi_lines.json", "w") as f:
            f.write(cleaned_json_str)

        # Save as JSON
        # df.to_json("kikuchi_lines.json", orient="records", indent=4)

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
            markersTmp, grouped_kikuchi_dict_list = self._kikuchi_line_labels_as_markers(desired_hkl=desired_hkl,
                                                                                         **kikuchi_line_labels_kwargs)
            markers += markersTmp

        return markers, grouped_kikuchi_dict_list


# Main function for testing the custom class
class CustomKikuchiPatternSimulator(KikuchiPatternSimulator):
    def on_detector(self, detector, rotations):
        result = super().on_detector(detector, rotations)

        # Now, return the custom GeometricalKikuchiPatternSimulation using the same components
        return CustomGeometricalKikuchiPatternSimulation(
            detector, rotations, result.reflectors, result._lines, result._zone_axes
        )


# Use this CustomKikuchiPatternSimulator in the main method


class BandWidthAutomator:
    def __init__(self, config_path="bandDetectorOptionsDebug.yml"):
        self.config = configLoader.load_config(config_path)
        self.start_time = time.time()
        self.data_path = self.config.get("h5_file_path", "path_to_default_file.h5")
        self.output_dir = os.path.dirname(self.data_path)
        self.base_name, self.ext = os.path.splitext(os.path.basename(self.data_path))
        if self.ext == ".oh5":
            new_data_path = os.path.join(self.output_dir, f"{self.base_name}.h5")
            shutil.copy(self.data_path, new_data_path)
            logging.info(f"Copied .oh5 file to new .h5 file: {new_data_path}")
            self.data_path = new_data_path
        self.modified_data_path = os.path.join(self.output_dir, f"{self.base_name}_modified.h5")
        shutil.copy(self.data_path, self.modified_data_path)
        logging.info(f"Copied HDF5 file to: {self.modified_data_path}")
        self.in_ang_path = os.path.join(self.output_dir, f"{self.base_name}.ang")

    def run(self):
        config = self.config
        crop_start, crop_end = config.get("crop_start", 5), config.get("crop_end", 25)
        debug = config.get("debug", False)
        desired_hkl = config.get("desired_hkl", "111")

        logging.info(f"Loading dataset from: {self.data_path}")
        s = kp.load(self.data_path, lazy=False)

        if debug:
            logging.info("Debug mode enabled: Cropping data for faster processing.")
            s.crop(1, start=crop_start, end=crop_end + 10)
            s.crop(0, start=crop_start, end=crop_end)

        phase_config = config["phase_list"]
        phase_list = PhaseList(
            Phase(
                name=phase_config["name"],
                space_group=phase_config["space_group"],
                structure=Structure(
                    lattice=Lattice(*phase_config["lattice"]),
                    atoms=[Atom(a["element"], a["position"]) for a in phase_config["atoms"]],
                ),
            ),
        )
        hkl_list = config["hkl_list"]
        header_data = ut.extract_header_data(self.modified_data_path)

        sig_shape = s.axes_manager.signal_shape[::-1]
        det = kp.detectors.EBSDDetector(
            sig_shape,
            sample_tilt=float(header_data.get("Sample Tilt", 0.0)),
            tilt=float(header_data.get("Camera Elevation Angle", 0.0)),
            azimuthal=float(header_data.get("Camera Azimuthal Angle", 0.0)),
            convention="edax",
            pc=tuple(header_data.get("pc", (0.0, 0.0, 0.0)))
        )

        indexer = det.get_indexer(phase_list, hkl_list, nBands=10, tSigma=2, rSigma=2)
        xmap, index_data, indexed_band_data = s.hough_indexing(
            phase_list=phase_list, indexer=indexer, return_index_data=True,
            return_band_data=True, verbose=1
        )

        ref = ReciprocalLatticeVector(phase=xmap.phases[0], hkl=hkl_list)
        ref = ref.symmetrise()
        simulator = CustomKikuchiPatternSimulator(ref)
        sim = simulator.on_detector(det, xmap.rotations.reshape(*xmap.shape))

        markers, grouped_kikuchi_dict_list = sim.as_markers(kikuchi_line_labels=True, desired_hkl=desired_hkl)
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

        ebsd_data = s.data
        processed_results = process_kikuchi_images(
            ebsd_data, grouped_kikuchi_dict_list, desired_hkl=desired_hkl, config=config
        )
        output_excel_path = os.path.join(self.output_dir, f"{self.base_name}_bandOutputData.xlsx")
        filtered_excel_path = os.path.join(self.output_dir, f"{self.base_name}_filtered_band_data.xlsx")
        save_results_to_excel(processed_results, output_excel_path, filtered_excel_path)

        df = pd.read_excel(filtered_excel_path)
        required_columns = ["Band Width", "psnr", "efficientlineIntensity", "Ind", "defficientlineIntensity"]
        for col in required_columns:
            if col not in df.columns:
                logging.error(f"{col} column not found in filtered_band_data.xlsx.")
                return
        logging.info("Loaded band_width, psnr,defficientlineIntensity and efficientlineIntensity data from Excel.")

        with h5py.File(self.modified_data_path, "a") as h5file:
            target_dataset_name = next(name for name in h5file if name not in ["Manufacturer", "Version"])
            ci_len = len(h5file[f"/{target_dataset_name}/EBSD/Data/CI"])

        max_index = df["Ind"].max()
        if max_index >= ci_len:
            logging.error("Maximum index in 'Ind' exceeds length of /Nickel/EBSD/Data/CI.")
            return

        desired_hkl_ref_width = config["desired_hkl_ref_width"]
        elastic_modulus = float(config["elastic_modulus"])

        arrays = ut.compute_band_arrays(
            df,
            ci_len,
            desired_hkl_ref_width,
            elastic_modulus,
        )

        ut.save_band_data_to_ang(self.in_ang_path, desired_hkl, arrays)
        ut.add_band_results_to_hdf5(self.modified_data_path, arrays)

        logging.info("Added Band_Width, psnr, defficientlineIntensity, efficientlineIntensity data to HDF5 file.")
        logging.info("Process completed. Results saved to Excel files and modified .ang file.")
        end_time = time.time()
        logging.info(f"The total processing time is : {end_time - self.start_time} seconds")


if __name__ == "__main__":
    BandWidthAutomator().run()



