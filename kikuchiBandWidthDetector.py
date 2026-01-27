
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core band width detection utilities.

This module contains the :class:`BandDetector` and
:class:`KikuchiBatchProcessor` classes used by the automation pipeline.
It can also be executed directly for standalone experiments.
"""

import copy
import json
import logging
import math
import os
import time
from typing import List, Dict, Any

import cv2
from configLoader import load_config
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import dask.array as da
import matplotlib.pyplot as plt
import utilities as ut
from packaging.version import parse as _v
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase
from orix.vector import Miller
from typing import Union, Tuple

from strategies import RectangularAreaBandDetector

# ──────────────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bandDetector.log"),
              logging.StreamHandler()]
)

# ──────────────────────────────────────────────────────────────────────────────
# Band-detector class
# ──────────────────────────────────────────────────────────────────────────────
class BandDetector:
    """Detect band widths for a single Kikuchi pattern."""
    def __init__(self, image=None, image_path=None, points=None,
                 desired_hkl="1,1,1", config=None, phase = None):
        """Initialise the detector with either an image array or path."""
        if image is not None:
            if isinstance(image, da.Array):
                self.image = image.compute()
            else:
                self.image = self._ensure_grayscale(image)
        elif image_path is not None:
            self.image = self._load_image(image_path)
        else:
            raise ValueError("Provide either an image array or an image path.")

        self.points = points
        self.desired_hkl = desired_hkl
        self.config = config if config else self._load_config()
        if phase is None:
            self.phase = ut.make_phase(self.config)
        else:
            self.phase = phase

    # ─────────────────────────────────── helpers
    def _ensure_grayscale(self, image):
        """Ensure that the image is in grayscale format."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            logging.info("Converting provided image to grayscale.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 2:
            return image
        raise ValueError("Image must be 2-D grayscale or 3-D RGB/BGR.")

    def _load_image(self, image_path):
        """Load an image from disk and convert it to grayscale."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        logging.info("Image loaded from %s", image_path)
        return self._ensure_grayscale(image)

    def _load_config(self, path="bandDetectorOptionsMagnetite.yml"):
        """Wrapper around :func:`configLoader.load_config`."""
        return load_config(path)

    # ─────────────────────────────────── public
    def detect_bands(self) -> List[Dict[str, Any]]:
        """Detect valid bands for all provided marker points."""
        results = []
        for point in self.points:
            hkl        = point["hkl"]
            cline      = np.around(point["central_line"], 3).tolist()
            mid_xy     = np.around(point["line_mid_xy"], 3).tolist()
            dist       = point["line_dist"]
            hkl_group  = point.get("hkl_group", "unknown")
            belongToDesiredGroup, ang_deg=ut.belongs_to_group(self.config["desired_hkl"], hkl, phase=self.phase)

            if belongToDesiredGroup:
                result = self._detect_band(cline, hkl)
                if result["band_valid"]:
                    result.update({
                        "hkl": hkl,
                        "line_mid_xy": mid_xy,
                        "line_dist": dist,
                        "hkl_group": hkl_group
                    })
                    results.append(result)
                if len(results) > 3:
                    break
            else:
                if self.config.get('debug', False):
                    logging.warning(f"current point hkl : {hkl} is skipped as it is not sym_equl to desired group {self.config['desired_hkl']}")
        return results

    # ─────────────────────────────────── internals
    def _detect_band(self, central_line, hkl):
        """Run a detection strategy on a single band."""
        detector = RectangularAreaBandDetector(self.image,
                                               central_line,
                                               self.config,
                                               hkl)
        return detector.detect()


class KikuchiBatchProcessor:
    """Process a grid of Kikuchi patterns and detect bands."""
    def __init__(self, ebsd_data, json_input, config=None, desired_hkl="1,1,1", phase=None):
        """Store data and configuration for batch processing."""
        self.ebsd_data = ebsd_data
        self.json_input = json_input
        self.config = config if config is not None else load_config("bandWidthOptions.yml")
        self.desired_hkl = desired_hkl
        phase = ut.make_phase(config["phase_list"])
        self.phase = phase

    def process_kikuchi_image_at_pixel(self, row, col, json_entry):
        """Process a single image from the EBSD grid."""
        image = self.ebsd_data[row, col]
        points = json_entry.get("points", [])
        if not isinstance(points, list):
            logging.warning(
                "Malformed points list for pixel (%d, %d); defaulting to empty list.",
                row,
                col,
            )
            points = []
        bdet = BandDetector(
            image=image,
            points=points,
            desired_hkl=self.desired_hkl,
            config=self.config, phase = self.phase,
        )
        try:
            results = bdet.detect_bands()
        except Exception as e:
            logging.warning("pattern [%d,%d] error: %s", row, col, e)
            results = []

        entry = json_entry.copy()
        entry.update({
            "bands": results,
            "x,y": [row, col],
            "ind": row * self.ebsd_data.shape[1] + col,
        })
        if "pattern_path" in json_entry:
            entry["pattern_path"] = json_entry["pattern_path"]
        return entry

    def _process_serial(self):
        """Serial implementation used for all processing."""
        processed = []
        ncol = self.ebsd_data.shape[1]
        for row in tqdm(range(self.ebsd_data.shape[0]), desc="Processing rows"):
            for col in range(self.ebsd_data.shape[1]):
                idx = ncol * row + col
                entry = self.process_kikuchi_image_at_pixel(
                    row, col, self.json_input[idx]
                )
                processed.append(entry)
        return processed

    def process(self):
        """Process the entire data set and return results list."""
        start = time.time()
        processed = self._process_serial()
        dur = time.time() - start
        n_patterns = np.prod(self.ebsd_data.shape[:2])
        logging.info(
            "Processed %d patterns in %.2f s (%.4f s / 1000 patterns).",
            n_patterns,
            dur,
            1000 * dur / n_patterns,
        )
        return processed

def _square_pattern(arr: np.ndarray, tol: float = 0.95) -> np.ndarray:
    """Resize a pattern to square shape if necessary."""
    h, w = arr.shape
    if h == w:
        return arr
    ratio = min(h, w) / max(h, w)
    if ratio < tol:
        raise ValueError(f"Pattern {h}×{w} too far from square (ratio {ratio:.3f}).")
    target = max(h, w)
    logging.warning("Pattern %d×%d not square; resizing to %d×%d.",
                    h, w, target, target)
    return cv2.resize(arr, (target, target), interpolation=cv2.INTER_NEAREST)


def load_ebsd_data(source: str, tile_rows: int = 10, tile_cols: int = 10):
    """Load EBSD patterns from an image folder or numpy array."""
    IMG_EXT = (".png", ".bmp")

    # folder of images
    if os.path.isdir(source):
        imgs = sorted(os.path.join(source, f) for f in os.listdir(source)
                      if f.lower().endswith(IMG_EXT))
        if not imgs:
            raise FileNotFoundError("No *.png or *.bmp in folder.")
        with Image.open(imgs[0]) as im:
            ref = _square_pattern(np.asarray(im.convert("L")))
        h = w = ref.shape[0]
        arrs = [ref]
        for p in imgs[1:]:
            with Image.open(p) as im:
                a = _square_pattern(np.asarray(im.convert("L")))
            if a.shape != (h, w):
                raise ValueError(f"{p} became {a.shape}, expected {(h, w)}.")
            arrs.append(a)
        n = len(arrs)
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / rows)
        pad = rows * cols - n
        if pad:
            arrs.extend([np.zeros((h, w), dtype=ref.dtype)] * pad)
        return np.stack(arrs).reshape(rows, cols, h, w)

    # single image
    if os.path.isfile(source) and source.lower().endswith(IMG_EXT):
        with Image.open(source) as im:
            arr = _square_pattern(np.asarray(im.convert("L")))
        return np.tile(arr, (tile_rows, tile_cols, 1, 1))

    # .npy
    if source.lower().endswith(".npy"):
        arr = np.load(source)
        if arr.ndim == 4:
            return arr
        if arr.ndim == 2:
            arr = _square_pattern(arr)
            return np.tile(arr, (tile_rows, tile_cols, 1, 1))
        raise ValueError(".npy must be 2-D or 4-D.")

    raise ValueError("source must be folder, *.png / *.bmp, or *.npy.")

# ──────────────────────────────────────────────────────────────────────────────
def _validate_json_entry(entry: Dict[str, Any], index: int) -> None:
    """
    Validate a single JSON annotation entry.

    Parameters:
        entry: JSON entry containing at least a "points" list.
        index: Position of the entry within the JSON list.

    Returns:
        None.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"JSON entry {index} must be a dictionary.")
    if "points" not in entry:
        raise ValueError(f"JSON entry {index} missing required 'points' field.")
    if not isinstance(entry["points"], list):
        raise ValueError(f"JSON entry {index} has malformed 'points' field.")
    if "pattern_path" in entry and not isinstance(entry["pattern_path"], str):
        raise ValueError(f"JSON entry {index} has non-string 'pattern_path'.")


def prepare_json_input(path: str, n_patterns: int, tile_from_single: bool):
    """Load JSON annotation file and repeat entries if required."""
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON file: {path}") from exc

    if tile_from_single:
        if isinstance(data, list):
            if not data:
                raise ValueError("JSON list is empty; cannot tile entries.")
            base = data[0]
        elif isinstance(data, dict):
            base = data
        else:
            raise TypeError("JSON must be a list or object when tiling from a single entry.")
        _validate_json_entry(base, 0)
        return [copy.deepcopy(base) for _ in range(n_patterns)]

    if not isinstance(data, list):
        raise TypeError("JSON must be a list when loading from folder.")
    if len(data) != n_patterns:
        raise ValueError(f"JSON {len(data)} entries ≠ patterns {n_patterns}.")
    for idx, entry in enumerate(data):
        _validate_json_entry(entry, idx)
    return data

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    SOURCE_ROOT = r"tmp_sim_out/"
    json_root = r"testData/"

    config = load_config("bandDetectorOptionsHcp.yml")
    config = load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    SOURCE = r"testData/Med_Mn_10k_4x4_00995.png"
    SOURCE = (r"C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\0pct_8.396\0 0 0\230x230.bmp")
    SOURCE = (r"C:\Users\kvman\Document"
              r"s\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\5pct_8.8158\0 0 0\230x230.bmp")
    SOURCE = (r"C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\3pct_8.64788\0 0 0\230x230.bmp")
    SOURCE = (r"C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\4pct_8.73184\0 0 0\230x230.bmp")
    jsonFile = r"testData/Med_Mn_10k_4x4_00995.json"
    jsonFile = r"C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\0pct_0_0_0_230_230.json"

    ##### now 460X460 cases
    SOURCE, jsonFile, config = SOURCE_ROOT+r"groundTruth\0_0_0_0pctStrain_460x460.bmp", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    SOURCE, jsonFile, config = SOURCE_ROOT+r"noisyImages_2\0_0_0_0pctStrain_460x460_noisy.bmp", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"Images_ai_processed\cyclegan_kikuchi_model_weights\de_blur_noise_accuracy\test_latest\images\0_0_0_0pctStrain_460x460__level_1_noisy.png", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"0pct_8.396\0 0 0\460x460_noisy.bmp", json_root+r"0pct_0_0_0_460_460.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"ai_processed\0_0_0_0_pct.bmp", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")

    ### for cropped
    SOURCE_ROOT = r"tmp_sim_out/"
    SOURCE_ROOT = r"testData/Images_ai_processed_cropped/cyclegan_kikuchi_model_weights/de_blur_noise_accuracy/test_latest/images/"
    json_root = r"testData/"
    SOURCE, jsonFile, config = SOURCE_ROOT+r"1840x1840_25_pct.png", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"1840x1840_25_pct_noisy.png", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"Images_ai_processed\cyclegan_kikuchi_model_weights\de_blur_noise_accuracy\test_latest\images\0_0_0_0pctStrain_460x460__level_1_noisy.png", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"0pct_8.396\0 0 0\460x460_noisy.bmp", json_root+r"0pct_0_0_0_460_460.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")
    # SOURCE, jsonFile, config = SOURCE_ROOT+r"ai_processed\0_0_0_0_pct.bmp", json_root+r"0_pct_8.396.json",load_config("bandDetectorOptionsMagnetiteAccuracyTesting.yml")

    phase = ut.make_phase(config["phase_list"])

    ebsd_data  = load_ebsd_data(SOURCE, tile_rows=1, tile_cols=1)
    n_patterns = np.prod(ebsd_data.shape[:2])

    json_input = prepare_json_input(jsonFile,
                                    n_patterns,
                                    tile_from_single=True)

    processor = KikuchiBatchProcessor(
        ebsd_data,
        json_input,
        config=config,
        desired_hkl=config.get("desired_hkl", "110"), phase=phase
    )
    results = processor.process()

    ut.save_results_to_csv(
        results,
        raw_path="bandOutputData.csv",
        filtered_path="filtered_band_data.csv",
    )
