
import time
import os
from pathlib import Path
from configLoader import load_config
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
# from hyperspy.utils.markers import line_segment, point, text


import pandas as pd
from collections import defaultdict
import json
from kikuchiBandWidthDetector import process_kikuchi_images
from kikuchiBandWidthDetector import save_results_to_csv      # <-- CSV helper
import shutil
import h5py
import utilities as ut
from packaging.version import parse as _v
import hyperspy.api as hs

_HS_VERSION = _v(hs.__version__)

if _HS_VERSION < _v("2.0"):
    from hyperspy.utils.markers import text as _Text

    def make_text_marker(x, y, label, **kw):
        return _Text(x=x, y=y, text=label, **kw)
else:
    def make_text_marker(x, y, label, **kw):
        # HyperSpy ≥ 2.0
        return hs.plot.markers.Texts(offsets=(x, y), texts=[label], **kw)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


GeometricalKikuchiPatternSimulation = kp.simulations.GeometricalKikuchiPatternSimulation
KikuchiPatternSimulator = kp.simulations.KikuchiPatternSimulator


class CustomGeometricalKikuchiPatternSimulation(GeometricalKikuchiPatternSimulation):
    """Adds helpers for grouping / serialising Kikuchi-line metadata."""

    def _group_by_ind(self, data):
        grouped_data = defaultdict(lambda: {"x,y": None, "ind": None, "points": []})
        for entry in data:
            ind = entry["ind"]
            if grouped_data[ind]["x,y"] is None:
                grouped_data[ind]["x,y"] = entry["(x,y)"]
                grouped_data[ind]["ind"] = ind
            grouped_data[ind]["points"].append(
                {
                    "hkl": entry["hkl"],
                    "hkl_group": entry["hkl_group"],
                    "central_line": entry["central_line"],
                    "line_mid_xy": entry["line_mid_xy"],
                    "line_dist": entry["line_dist"],
                }
            )
        return list(grouped_data.values())

    def _df_to_grouped_json(self, df):
        json_str = df.to_json(orient="records")
        data = json.loads(json_str)
        grouped_data = self._group_by_ind(data)
        return json.dumps(grouped_data, indent=4), grouped_data

    @staticmethod
    def remove_newlines_from_fields(json_str):
        central_line_pattern = r'("central_line":\s*\[[^\]]*\])'
        line_mid_xy_pattern = r'("line_mid_xy":\s*\[[^\]]*\])'
        xy_pattern = r'("x,y":\s*\[[^\]]*\])'

        def cleanup_array(match):
            field, array_content = match.group(0).split(":")
            clean_array_content = re.sub(r'\s+', ' ', array_content).replace('[ ', '[').replace(' ]', ']')
            return f'{field}: {clean_array_content}'

        json_str = re.sub(central_line_pattern, cleanup_array, json_str, flags=re.DOTALL)
        json_str = re.sub(line_mid_xy_pattern, cleanup_array, json_str, flags=re.DOTALL)
        json_str = re.sub(xy_pattern, cleanup_array, json_str, flags=re.DOTALL)
        return json_str

    @staticmethod
    def get_hkl_group(hkl_str):
        hkl_values = sorted(map(abs, map(int, hkl_str.split())))
        return ''.join(map(str, hkl_values))

    # ------------------------------------------------------------------ #
    #  Generate Kikuchi‐line label markers and dump metadata to disk
    # ------------------------------------------------------------------ #
    def _kikuchi_line_labels_as_markers(self, desired_hkl='111', **kwargs) -> list:
        coords = self.lines_coordinates(index=(), exclude_nan=False)
        coords = np.around(coords, 3)
        reflectors = self._reflectors.coordinates.round().astype(int)
        array_str = np.array2string(reflectors, threshold=reflectors.size)
        texts = re.sub("[][ ]", " ", array_str).split("\n")

        filtered_texts = np.array(
            [t if self.get_hkl_group(t) == desired_hkl else '' for t in texts]
        )
        refelctor_group = [''.join(map(str, sorted(map(abs, row)))) for row in reflectors]

        kw = {
            "color": "b",
            "zorder": 4,
            "ha": "center",
            "va": "bottom",
            "bbox": {"fc": "w", "ec": "b", "boxstyle": "round,pad=0.3"},
        }
        kw.update(kwargs)

        kikuchi_line_label_list = []
        rows, cols, n, _ = coords.shape
        num_cols = cols
        kikuchi_line_dict_list = [[[] for _ in range(cols)] for _ in range(rows)]
        is_finite = np.isfinite(coords)[..., 0]
        coords[~is_finite] = -1
        det_bounds = self.detector.shape
        det_mid_point = 0.5 * det_bounds[0], 0.5 * det_bounds[1]
        dist_threshold = 0.75 * 0.5 * det_bounds[0]

        for i in range(reflectors.shape[0]):
            if not np.allclose(coords[..., i, :], -1):
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

                text_marker = make_text_marker(x, y, filtered_texts[i], **kw)
                kikuchi_line_label_list.append(text_marker)

                valid_mask = is_finite[..., i]
                if np.any(valid_mask):
                    row_indices, col_indices = np.where(valid_mask)
                    num_valid = row_indices.size
                    kikuchi_data = {
                        "hkl": texts[i].strip(),
                        "hkl_group": refelctor_group[i],
                        "central_line": np.vstack([x1[valid_mask], y1[valid_mask],
                                                   x2[valid_mask], y2[valid_mask]]).T.tolist(),
                        "line_mid_xy": np.vstack([x[valid_mask], y[valid_mask]]).T.tolist(),
                        "line_dist": line_dist[valid_mask].T.tolist(),
                    }
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

        flat_list = [e for sub in kikuchi_line_dict_list for entry in sub if entry for e in entry]
        df = pd.DataFrame(flat_list)

        final_json_str, grouped_dict_list = self._df_to_grouped_json(df)
        cleaned_json_str = CustomGeometricalKikuchiPatternSimulation.remove_newlines_from_fields(final_json_str)
        with open("kikuchi_lines.json", "w") as f:
            f.write(cleaned_json_str)

        # --- CSV instead of Excel
        df.to_csv("kikuchi_lines.csv", index=False)

        return kikuchi_line_label_list, grouped_dict_list

    # ------------------------------------------------------------------ #
    #  Override as_markers to hook in Kikuchi-line labels
    # ------------------------------------------------------------------ #
    def as_markers(
            self,
            lines: bool = True,
            zone_axes: bool = False,
            zone_axes_labels: bool = False,
            kikuchi_line_labels: bool = False,
            pc: bool = False,
            lines_kwargs: Optional[dict] = None,
            zone_axes_kwargs: Optional[dict] = None,
            zone_axes_labels_kwargs: Optional[dict] = None,
            kikuchi_line_labels_kwargs: Optional[dict] = None,
            desired_hkl='111',
            pc_kwargs: Optional[dict] = None,
    ) -> list:
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
            line_markers, grouped_dict_list = self._kikuchi_line_labels_as_markers(
                desired_hkl=desired_hkl, **kikuchi_line_labels_kwargs)
            markers += line_markers
        else:
            grouped_dict_list = None

        return markers, grouped_dict_list


class CustomKikuchiPatternSimulator(KikuchiPatternSimulator):
    def on_detector(self, detector, rotations):
        reflectors = self.reflectors
        result = super().on_detector(detector, rotations)
        return CustomGeometricalKikuchiPatternSimulation(
            detector, rotations, result.reflectors, result._lines, result._zone_axes
        )

class BandWidthAutomator:
    """Pipeline wrapper for band-width detection."""

    def __init__(self, config_path: str = "bandDetectorOptionsDebug.yml"):
        self.config = load_config(config_path)
        self.data_path = Path(self.config.get("h5_file_path", "path_to_default_file.h5"))
        self.output_dir = self.data_path.parent
        self.base_name = self.data_path.stem
        self.modified_data_path = self.output_dir / f"{self.base_name}_modified.h5"
        self.in_ang_path = self.output_dir / f"{self.base_name}.ang"
        self.dataset = None
        self.grouped_dict_list = None

    # ------------------------------------------------------------------
    def prepare_dataset(self):
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

        ref = ReciprocalLatticeVector(phase=xmap.phases[0], hkl=hkl_list).symmetrise()
        simulator = CustomKikuchiPatternSimulator(ref)
        sim = simulator.on_detector(det, xmap.rotations.reshape(*xmap.shape))

        desired_hkl = self.config.get("desired_hkl", "111")
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
        desired_hkl = self.config.get("desired_hkl", "111")
        ebsd_data = self.dataset.data
        return process_kikuchi_images(
            ebsd_data,
            self.grouped_dict_list,
            desired_hkl=desired_hkl,
            config=self.config,
        )

    # ------------------------------------------------------------------
    def export_results(self, processed):
        output_csv_path = self.output_dir / f"{self.base_name}_bandOutputData.csv"
        filtered_csv_path = self.output_dir / f"{self.base_name}_filtered_band_data.csv"
        save_results_to_csv(processed, str(output_csv_path), str(filtered_csv_path))

        desired_hkl = self.config.get("desired_hkl", "111")

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
    BandWidthAutomator().run()
