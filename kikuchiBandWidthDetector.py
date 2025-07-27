
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Band-width detection workflow
• Excel → CSV conversion (10 Jun 2025)
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
    def __init__(self, image=None, image_path=None, points=None,
                 desired_hkl="111", config=None):
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

    # ─────────────────────────────────── helpers
    def _ensure_grayscale(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            logging.info("Converting provided image to grayscale.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 2:
            return image
        raise ValueError("Image must be 2-D grayscale or 3-D RGB/BGR.")

    def _load_image(self, image_path):
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
        results = []
        for point in self.points:
            hkl        = point["hkl"]
            cline      = np.around(point["central_line"], 3).tolist()
            mid_xy     = np.around(point["line_mid_xy"], 3).tolist()
            dist       = point["line_dist"]
            hkl_group  = point.get("hkl_group", "unknown")

            if sorted(hkl_group) == sorted(self.desired_hkl):
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
        return results

    # ─────────────────────────────────── internals
    def _detect_band(self, central_line, hkl):
        detector = RectangularAreaBandDetector(self.image,
                                               central_line,
                                               self.config,
                                               hkl)
        return detector.detect()


class KikuchiBatchProcessor:
    def __init__(self, ebsd_data, json_input, config=None, desired_hkl="111"):
        self.ebsd_data = ebsd_data
        self.json_input = json_input
        self.config = config if config is not None else load_config("bandWidthOptions.yml")
        self.desired_hkl = desired_hkl

    def process_kikuchi_image_at_pixel(self, row, col, json_entry):
        image = self.ebsd_data[row, col]
        points = json_entry["points"]
        bdet = BandDetector(
            image=image,
            points=points,
            desired_hkl=self.desired_hkl,
            config=self.config,
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
        return entry

    def _process_serial(self):
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
def prepare_json_input(path: str, n_patterns: int, tile_from_single: bool):
    with open(path) as f:
        data = json.load(f)
    if tile_from_single:
        base = data[0] if isinstance(data, list) else data
        return [copy.deepcopy(base) for _ in range(n_patterns)]
    if not isinstance(data, list):
        raise TypeError("JSON must be a list when loading from folder.")
    if len(data) != n_patterns:
        raise ValueError(f"JSON {len(data)} entries ≠ patterns {n_patterns}.")
    return data

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config("bandDetectorOptions.yml")

    SOURCE = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\ML_4x4Patterns\ML_4x4Patterns\Med_Mn_10k_4x4_00995.png"
    jsonFile = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\inference_testing_MLOutput\SingleMed_Mn_10k_4x4_00995.json"

    ebsd_data  = load_ebsd_data(SOURCE, tile_rows=1, tile_cols=1)
    n_patterns = np.prod(ebsd_data.shape[:2])

    json_input = prepare_json_input(jsonFile,
                                    n_patterns,
                                    tile_from_single=True)

    processor = KikuchiBatchProcessor(
        ebsd_data,
        json_input,
        config=config,
        desired_hkl=config.get("desired_hkl", "002"),
    )
    results = processor.process()

    ut.save_results_to_csv(
        results,
        raw_path="bandOutputData.csv",
        filtered_path="filtered_band_data.csv",
    )
