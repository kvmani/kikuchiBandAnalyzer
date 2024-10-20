
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
        return json.dumps(grouped_data, indent=4)


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

    def _kikuchi_line_labels_as_markers(self, **kwargs) -> list:
        """Return a list of Kikuchi line label text markers."""
        coords = self.lines_coordinates(index=(), exclude_nan=False)

        # Labels for Kikuchi lines can be based on reflectors (e.g., hkl values)
        reflectors = self._reflectors.coordinates.round().astype(int)
        array_str = np.array2string(reflectors, threshold=reflectors.size)
        texts = re.sub("[][ ]", "", array_str).split("\n")

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
                text_marker = text(x=x, y=y, text=texts[i], **kw)
                kikuchi_line_label_list.append(text_marker)

                # Vectorized approach to fill kikuchi_line_dict_list
                valid_mask = is_finite[..., i]  # Get the mask of finite values

                if np.any(valid_mask):  # Check if there are any valid points
                    # Create indices of valid points
                    row_indices, col_indices = np.where(valid_mask)
                    # Prepare the data for valid points
                    num_valid = row_indices.size
                    kikuchi_data = {
                        "hkl": texts[i],
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
        final_json_str = self._df_to_grouped_json(df)
        cleaned_json_str = CustomGeometricalKikuchiPatternSimulation.remove_newlines_from_fields(final_json_str)
        # Dump the final JSON string to disk
        with open("kikuchi_lines.json", "w") as f:
            f.write(cleaned_json_str)

        # Save as JSON
        #df.to_json("kikuchi_lines.json", orient="records", indent=4)


        # Save as Excel
        df.to_excel("kikuchi_lines.xlsx", index=False, engine="openpyxl")

        return kikuchi_line_label_list, kikuchi_line_dict_list

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
            kikuchi_line_labels_kwargs: Optional[dict] = None,  # kwargs for Kikuchi line labels
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
            markersTmp,  kikuchi_line_dict_list = self._kikuchi_line_labels_as_markers(**kikuchi_line_labels_kwargs)
            markers+=markersTmp


        return markers, kikuchi_line_dict_list


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
    choice = 'test_data'  # Variable to control which dataset and phase_list to load
    choice = 'debarna_data'  # Variable to control which dataset and phase_list to load

    # Load dataset based on choice
    if choice == 'test_data':
        data_path = r"C:\Users\kvman\Downloads\IS_Ni_ebsd_data\Nickel.h5"
        phase_list = PhaseList(
            Phase(
                name="Ni",
                space_group=225,
                structure=Structure(
                    lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
                    atoms=[Atom("Ni", [0, 0, 0])],
                ),
            ),
        )
        hkl_list = [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]  # Example HKL list for Nickel
        pc = [0.545, 0.610, 0.6863]
    else:
        data_path = r"C:\Users\kvman\Downloads\OneDrive_1_10-20-2024\magnetite_data.h5"
        phase_list = PhaseList(
            Phase(
                name="Magnetite",
                space_group=227,
                structure=Structure(
                    lattice=Lattice(8.3959, 8.3959, 8.3959, 90, 90, 90),
                    atoms=[Atom("X", [0, 0, 0]), Atom("Y", [0.5, 0.5, 0.5])],
                ),
            ),
        )
        hkl_list = [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]  # Example HKL list for the other material
        pc = [0.54728, 0.7169, 0.6969]

    # Load the dataset and preprocess
    s = kp.load(data_path, lazy=True)
    s.crop(1, start=10, end=15)

    # Detector setup and pattern extraction
    sig_shape = s.axes_manager.signal_shape[::-1]
    det = kp.detectors.EBSDDetector(sig_shape, sample_tilt=70, convention="edax", pc=pc)

    # PhaseList and indexing
    indexer = det.get_indexer(phase_list, hkl_list, nBands=10, tSigma=2, rSigma=2)
    xmap = s.hough_indexing(phase_list=phase_list, indexer=indexer, verbose=1)

    # Use the CustomKikuchiPatternSimulator for geometrical simulations
    ref = ReciprocalLatticeVector(phase=xmap.phases[0], hkl=hkl_list)
    ref = ref.symmetrise()
    simulator = CustomKikuchiPatternSimulator(ref)  # Use the custom simulator
    sim = simulator.on_detector(det, xmap.rotations.reshape(*xmap.shape))

    # Add markers (including zone axis labels) to the signal
    markers, kikuchi_line_dict_list = sim.as_markers(kikuchi_line_labels=True, zone_axes_labels=True)
    s.add_marker(markers, plot_marker=False, permanent=True)

    # RGB navigator (optional visualization)
    v_ipf = Vector3d.xvector()
    sym = xmap.phases[0].point_group
    ckey = plot.IPFColorKeyTSL(sym, v_ipf)
    rgb_x = ckey.orientation2color(xmap.rotations)
    maps_nav_rgb = kp.draw.get_rgb_navigator(rgb_x.reshape(xmap.shape + (3,)))
    s.plot(maps_nav_rgb)
    plt.show()

if __name__ == "__main__":
    main()



