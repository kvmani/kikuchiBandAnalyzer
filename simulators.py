import json
import re
from collections import defaultdict
from typing import Optional

import hyperspy.api as hs
import kikuchipy as kp
import numpy as np
import pandas as pd
from packaging.version import parse as _v

_HS_VERSION = _v(hs.__version__)

if _HS_VERSION < _v("2.0"):
    from hyperspy.utils.markers import text as _Text

    def make_text_marker(x, y, label, **kw):
        return _Text(x=x, y=y, text=label, **kw)
else:
    def make_text_marker(x, y, label, **kw):
        return hs.plot.markers.Texts(offsets=(x, y), texts=[label], **kw)


class CustomGeometricalKikuchiPatternSimulation(kp.simulations.GeometricalKikuchiPatternSimulation):
    """Adds helpers for grouping/serialising Kikuchi-line metadata."""

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
            clean_array_content = re.sub(r"\s+", " ", array_content).replace("[ ", "[").replace(" ]", "]")
            return f"{field}: {clean_array_content}"

        json_str = re.sub(central_line_pattern, cleanup_array, json_str, flags=re.DOTALL)
        json_str = re.sub(line_mid_xy_pattern, cleanup_array, json_str, flags=re.DOTALL)
        json_str = re.sub(xy_pattern, cleanup_array, json_str, flags=re.DOTALL)
        return json_str

    @staticmethod
    def get_hkl_group(hkl_str):
        hkl_values = sorted(map(abs, map(int, hkl_str.split())))
        return "".join(map(str, hkl_values))

    def _kikuchi_line_labels_as_markers(self, desired_hkl="111", **kwargs) -> list:
        coords = self.lines_coordinates(index=(), exclude_nan=False)
        coords = np.around(coords, 3)
        reflectors = self._reflectors.coordinates.round().astype(int)
        array_str = np.array2string(reflectors, threshold=reflectors.size)
        texts = re.sub("[][ ]", " ", array_str).split("\n")

        filtered_texts = np.array(
            [t if self.get_hkl_group(t) == desired_hkl else "" for t in texts]
        )
        refelctor_group = ["".join(map(str, sorted(map(abs, row)))) for row in reflectors]

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
                        "central_line": np.vstack([x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask]]).T.tolist(),
                        "line_mid_xy": np.vstack([x[valid_mask], y[valid_mask]]).T.tolist(),
                        "line_dist": line_dist[valid_mask].T.tolist(),
                    }
                    for idx in range(num_valid):
                        row, col = row_indices[idx], col_indices[idx]
                        if kikuchi_data["line_dist"][idx] < dist_threshold:
                            kikuchi_line_dict_list[row][col].append(
                                {
                                    "(x,y)": (row, col),
                                    "ind": (row * num_cols) + col,
                                    "hkl": kikuchi_data["hkl"],
                                    "hkl_group": kikuchi_data["hkl_group"],
                                    "central_line": kikuchi_data["central_line"][idx],
                                    "line_mid_xy": kikuchi_data["line_mid_xy"][idx],
                                    "line_dist": kikuchi_data["line_dist"][idx],
                                }
                            )

        flat_list = [e for sub in kikuchi_line_dict_list for entry in sub if entry for e in entry]
        df = pd.DataFrame(flat_list)

        final_json_str, grouped_dict_list = self._df_to_grouped_json(df)
        cleaned_json_str = CustomGeometricalKikuchiPatternSimulation.remove_newlines_from_fields(final_json_str)
        with open("kikuchi_lines.json", "w") as f:
            f.write(cleaned_json_str)

        df.to_csv("kikuchi_lines.csv", index=False)

        return kikuchi_line_label_list, grouped_dict_list

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
        desired_hkl: str = "111",
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
                desired_hkl=desired_hkl, **kikuchi_line_labels_kwargs
            )
            markers += line_markers
        else:
            grouped_dict_list = None

        return markers, grouped_dict_list


class CustomKikuchiPatternSimulator(kp.simulations.KikuchiPatternSimulator):
    def on_detector(self, detector, rotations):
        reflectors = self.reflectors
        result = super().on_detector(detector, rotations)
        return CustomGeometricalKikuchiPatternSimulation(
            detector, rotations, result.reflectors, result._lines, result._zone_axes
        )


__all__ = [
    "make_text_marker",
    "CustomGeometricalKikuchiPatternSimulation",
    "CustomKikuchiPatternSimulator",
]
