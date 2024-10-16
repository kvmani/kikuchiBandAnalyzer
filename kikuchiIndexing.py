
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

GeometricalKikuchiPatternSimulation = kp.simulations.GeometricalKikuchiPatternSimulation
KikuchiPatternSimulator = kp.simulations.KikuchiPatternSimulator
# Custom class as defined previously
class CustomGeometricalKikuchiPatternSimulation(GeometricalKikuchiPatternSimulation):
    def _kikuchi_line_labels_as_markers(self, **kwargs) -> list:
        """Return a list of Kikuchi line label text markers."""
        coords = self.lines_coordinates(index=(), exclude_nan=False)

        # Labels for Kikuchi lines can be based on reflectors (e.g., hkl values)
        reflectors = self._reflectors.coordinates.round().astype(int)
        array_str = np.array2string(reflectors, threshold=reflectors.size)
        texts = re.sub("[][ ]", "", array_str).split("\n")

        kw = {
            "color": "b",
            "zorder": 3,
            "ha": "center",
            "va": "bottom",
            "bbox": {"fc": "w", "ec": "b", "boxstyle": "round,pad=0.3"},
        }
        kw.update(kwargs)

        kikuchi_line_label_list = []
        kikuchi_line_dict_list=[]
        is_finite = np.isfinite(coords)[..., 0]
        coords[~is_finite] = -1

        for i in range(reflectors.shape[0]):
            if not np.allclose(coords[..., i, :], -1):  # All NaNs
                x1 = coords[..., i, 0]
                y1 = coords[..., i, 1]
                x2 = coords[..., i, 2]
                y2 = coords[..., i, 3]
                x=0.5*(x1+x2)
                y=0.5*(y1+y2)
                x[~is_finite[..., i]] = np.nan
                y[~is_finite[..., i]] = np.nan
                x = x.squeeze()
                y = y.squeeze()
                # Create a text marker with the label for each Kikuchi line
                text_marker = text(x=x, y=y, text=texts[i], **kw)
                kikuchi_line_label_list.append(text_marker)
                # kikuchi_line_dict_list.append({
                # "hkl":texts[i], "line_coordinates":[x1,y1,x2,y2],
                # "line_mid_points":[x,y],
                #  "reflector":self._reflectors[i],
                # })
        for i in range(reflectors.shape[0]):
            # Assuming x1, y1, x2, y2 are (r, c) matrices (line coordinates) and x, y are midpoints
            # Flatten the (r, c) matrices
            x1_flat = x1.flatten()
            y1_flat = y1.flatten()
            x2_flat = x2.flatten()
            y2_flat = y2.flatten()
            x = x.flatten()
            y = y.flatten()

            # Construct a dict for each Kikuchi line, separating flattened line coordinates
            for j in range(len(x1_flat)):  # Iterate over the flattened coordinates
                kikuchi_line_dict_list.append({
                    "pointId":j,
                    "hkl": texts[i],
                    "central_line": [x1_flat[j],y1_flat[j],x2_flat[j],y2_flat[j],],
                    "line_mid_xy": [x[j], y[j]],  # Midpoint x
                    "line_mid_y": 0.5 * (y1_flat[j] + y2_flat[j]),  # Midpoint y

                })

        # Convert to DataFrame
        df = pd.DataFrame(kikuchi_line_dict_list)

        # Save as JSON
        df.to_json("kikuchi_lines.json", orient="records", indent=4)

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
    # Load dataset and preprocess
    data_path = r"C:\Users\kvman\Downloads\IS_Ni_ebsd_data\Nickel.h5"
    s = kp.load(data_path, lazy=True)
    s.crop(1, start=10, end=15)

    # Detector setup and pattern extraction
    sig_shape = s.axes_manager.signal_shape[::-1]
    det = kp.detectors.EBSDDetector(sig_shape, sample_tilt=70, convention="edax", pc=[0.545, 0.610, 0.6863])

    # Extract selected patterns and optimize projection center (PC)
    grid_shape = (5, 4)
    s_grid, idx = s.extract_grid(grid_shape, return_indices=True)

    # PhaseList and indexing
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
    indexer = det.get_indexer(phase_list, [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]], nBands=10, tSigma=2, rSigma=2)
    xmap = s.hough_indexing(phase_list=phase_list, indexer=indexer, verbose=1)

    # Use the CustomKikuchiPatternSimulator for geometrical simulations
    ref = ReciprocalLatticeVector(phase=xmap.phases[0], hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]])
    ref = ref.symmetrise()
    simulator = CustomKikuchiPatternSimulator(ref)  # Use the custom simulator
    sim = simulator.on_detector(det, xmap.rotations.reshape(*xmap.shape))

    # Add markers (including zone axis labels) to the signal
    markers, kikuchi_line_dict_list = sim.as_markers(kikuchi_line_labels=True,zone_axes_labels=True)
    #markers = sim.as_markers(zone_axes_labels=True)
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

