import copy
import logging
import json
import cv2
import yaml
from tqdm import tqdm
import time

from strategies import RectangularAreaBandDetector
import pandas as pd
import dask.array as da
# Set up logging
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bandDetector.log"),
        logging.StreamHandler()  # This will log to the console
    ]
)


class BandDetector:
    def __init__(self, image=None, image_path=None, points=None, desired_hkl="111",config=None):
        """
        Initializes the band detector for multiple bands in an image.

        :param image: Kikuchi image (2D numpy array). If provided, image_path is ignored.
        :param image_path: Path to the input image. If image is not provided, image is loaded from this path.
        :param points: List of dictionaries with "hkl", "central_line", "line_mid_xy", and "line_dist" for each band.
        :param config: Configuration dictionary. If None, load from default YAML.
        """
        if image is not None:
            # Convert Dask array to NumPy array if necessary
            if isinstance(image, da.Array):
                self.image = image.compute()  # Convert Dask array to NumPy array

            else:
                self.image = self._ensure_grayscale(image)
        elif image_path is not None:
            self.image = self._load_image(image_path)
        else:
            raise ValueError("Either an image array or an image path must be provided.")

        self.points = points  # Multiple bands data
        self.desired_hkl=desired_hkl
        self.config = config if config else self._load_config()

    def _ensure_grayscale(self, image):
        """
        Ensures the image is in grayscale. Converts to grayscale if it's a color image.
        :param image: Input image (numpy array).
        :return: Grayscale image.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:  # Color image (RGB/BGR)
            logging.info("Converting provided image to grayscale.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:  # Already grayscale
            return image
        else:
            raise ValueError("Provided image must be either 2D grayscale or 3D RGB/BGR.")

    def _load_image(self, image_path):
        """
        Loads an image from the given path and converts it to grayscale.
        :param image_path: Path to the input image.
        :return: Grayscale image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        logging.info(f"Image loaded from {image_path}.")
        return self._ensure_grayscale(image)

    def _load_config(self, config_path="bandDetectorOptionsMagnetite.yml"):
        """
        Loads configuration from a YAML file.
        :param config_path: Path to the YAML configuration file.
        :return: Dictionary with configuration parameters.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def detect_bands(self):
        """
        Detects the bands for each point in the `points` list.
        :return: A list of dictionaries with detection results for each band.
        """
        results = []
        for point in self.points:
            hkl = point["hkl"]  # Extract hkl value from JSON
            #central_line = point["central_line"]
            central_line = np.around(point["central_line"], 3).tolist()
            line_mid_xy = np.around(point["line_mid_xy"],3).tolist()
            line_dist = point["line_dist"]
            hkl_group = point.get("hkl_group", "unknown")
            if sorted(hkl_group) == sorted(self.desired_hkl):
            # Detect the band, using RectangularAreaBandDetector
                result = self._detect_band(central_line, hkl)
                if result["band_valid"]:
                    result["hkl"] = hkl  # Add hkl label to the result
                    result["line_mid_xy"] = line_mid_xy  # Add line midpoint
                    result["line_dist"] = line_dist  # Add line distance
                    result["hkl_group"] = hkl_group
                    results.append(result)
                if len(results)>3:
                    break
        return results

    def _detect_band(self, central_line, hkl):
        """
        Detect a single band using RectangularAreaBandDetector and central line.
        :param central_line: Central line of the band.
        :param hkl: Miller indices of the band.
        :return: A dictionary with detection results.
        """
        detector = RectangularAreaBandDetector(self.image, central_line, self.config, hkl)
        return detector.detect()


def process_kikuchi_image_at_pixel(row, col, ebsd_data, json_entry, desired_hkl, config):
    """
    Processes a single Kikuchi image and its bands for the specified pixel coordinates.

    :param row: Row index of the EBSD dataset.
    :param col: Column index of the EBSD dataset.
    :param ebsd_data: 2D numpy array (m x n) with Kikuchi patterns.
    :param json_entry: JSON dictionary with band points for the current pixel.
    :param desired_hkl: Desired HKL for filtering bands.
    :param config: Configuration dictionary with processing parameters.
    :return: Processed entry with band detection results and pixel coordinates.
    """
    image = ebsd_data[row, col]
    points = json_entry["points"]

    # Initialize BandDetector for the current image and pixel
    band_detector = BandDetector(image=image, points=points, desired_hkl=desired_hkl, config=config)
    try:
        results = band_detector.detect_bands()
    except Exception as e:
        logging.warning(f"pattern: [{row} : {col}] An error occurred while detecting bands: {e}")
        results = []

    # Prepare processed entry
    processed_entry = json_entry.copy()
    processed_entry["bands"] = results
    processed_entry["x,y"] = [row, col]
    processed_entry["ind"] = row * ebsd_data.shape[1] + col
    return processed_entry

def process_kikuchi_images_serial(ebsd_data, json_input, desired_hkl='111', config=None):
    """
    Serial version: Processes multiple Kikuchi images for each pixel in the EBSD dataset.

    :param ebsd_data: 2D numpy array (m x n) where each entry is a Kikuchi pattern (2D numpy array).
    :param json_input: List of dictionaries with band points.
    :param desired_hkl: Desired HKL for filtering bands.
    :param config: Configuration dictionary.
    :return: Updated list of dictionaries with band detection results for each pixel.
    """
    processed_data = []
    ncol = ebsd_data.shape[1]

    for row in tqdm(range(ebsd_data.shape[0]), desc="Processing rows"):
        for col in range(ebsd_data.shape[1]):
            index = ncol*row+col
            json_entry = json_input[index]
            processed_entry = process_kikuchi_image_at_pixel(row, col, ebsd_data, json_entry, desired_hkl, config)
            processed_data.append(processed_entry)

    return processed_data


def process_kikuchi_images(ebsd_data, json_input, desired_hkl='111', config=None):
    """
    Processes multiple Kikuchi images for each pixel in the EBSD dataset.
    Decides between serial and parallel execution based on config.

    :param ebsd_data: 2D numpy array (m x n) where each entry is a Kikuchi pattern (2D numpy array).
    :param json_input: List of dictionaries with band points.
    :param desired_hkl: Desired HKL for filtering bands.
    :param config: Configuration dictionary containing 'execution_mode' and 'num_cores'.
    :return: Updated list of dictionaries with band detection results for each pixel.
    """
    execution_mode = config.get("execution_mode", "serial")
    num_cores = config.get("num_cores", None)

    logging.info(f"Starting {execution_mode} processing for Kikuchi images.")
    start_time = time.time()  # Start timing

    if execution_mode == "parallel":
        # Run the parallel version
        processed_data = process_kikuchi_images_parallel(ebsd_data, json_input, desired_hkl, config, num_cores)

        # Measure the duration of the parallel execution
        parallel_duration = time.time() - start_time
        logging.info(f"Completed parallel processing in {parallel_duration:.2f} seconds.")

        # Run the serial version separately for timing comparison
        logging.info("Running serial version for timing comparison...")
        serial_start_time = time.time()
        process_kikuchi_images_serial(ebsd_data, json_input, desired_hkl, config)
        serial_duration = time.time() - serial_start_time
        logging.info(f"Serial processing took {serial_duration:.2f} seconds.")

        # Calculate and log the speed gain
        speed_gain = serial_duration / parallel_duration if parallel_duration > 0 else 0
        logging.info(f"Estimated speed gain due to parallelization: {speed_gain:.2f}x.")

    else:
        # Run the serial version
        processed_data = process_kikuchi_images_serial(ebsd_data, json_input, desired_hkl, config)
        serial_duration = time.time() - start_time
        logging.info(f"Completed serial processing in {serial_duration:.2f} seconds.")

    return processed_data
def load_config(file_path="bandDetectorOptionsMagnetite.yml"):
    """
    Loads configuration options from a YAML file.

    :param file_path: Path to the YAML configuration file.
    :return: Dictionary with configuration settings.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config



def process_kikuchi_images(ebsd_data, json_input, config_file = "bandWidthOptions.yml", desired_hkl='111', config=None):
    """
    Processes multiple Kikuchi images for each pixel in the EBSD dataset.
    Decides between serial and parallel execution based on config.

    :param ebsd_data: 2D numpy array (m x n) where each entry is a Kikuchi pattern (2D numpy array).
    :param json_input: List of dictionaries with band points.
    :param desired_hkl: Desired HKL for filtering bands.
    :param config: Configuration dictionary containing 'execution_mode' and 'num_cores'.
    :return: Updated list of dictionaries with band detection results for each pixel.
    """
    if config is None:
        config = load_config(file_path=config_file)

    logging.info(f"Starting processing for Kikuchi images.")
    start_time = time.time()  # Start timing
    nPatterns = np.prod(ebsd_data.shape[:2]) ### only rows and cols are needed hence :2
    processed_data = process_kikuchi_images_serial(ebsd_data, json_input, desired_hkl, config)
    end_time = time.time()  # End timing
    duration = end_time - start_time
    logging.info(f"Completed  processing in {duration:.2f} seconds.")
    logging.info(f"Time per 1000 patterns :  {1000*duration/nPatterns : .4f} seconds.")

    return processed_data


def convert_results(results):
    if isinstance(results, dict):
        return {key: convert_results(value) for key, value in results.items()}
    elif isinstance(results, list):
        return [convert_results(item) for item in results]
    elif isinstance(results, np.integer):
        return int(results)
    elif isinstance(results, np.floating):
        return float(results)
    elif isinstance(results, np.ndarray):
        return results.tolist()  # Convert arrays to lists
    else:
        return results


def save_results_to_json(results, output_path="bandOutputData.json"):
    """
    Saves the processed results to a JSON file.
    :param results: List of dictionaries with processed band data.
    :param output_path: Path to save the JSON output file.
    """
    results_serializable = convert_results(results)

    with open(output_path, 'w') as file:
        json.dump(results_serializable, file, indent=4)
    logging.info(f"Results saved to {output_path}.")
    print(output_path)


def save_results_to_excel(results, output_path="bandOutputData.xlsx",filtered_excel_path='filtered_band_data.xlsx'):
    """
    Saves the processed results to an Excel (.xlsx) file.
    :param results: List of dictionaries with processed band data.
    :param output_path: Path to save the Excel output file.
    """
    # Convert the list of dictionaries to a pandas DataFrame
    data = []

    for result in results:
        xy = result.get("x,y")
        ind = result.get("ind")
        for band in result["bands"]:
            hkl = band.get("hkl", None,)
            central_line = band.get("central_line", None)
            line_mid_xy = band.get("line_mid_xy")
            line_dist = band.get("line_dist")
            band_width = band.get("bandWidth", None)  # Band width from detection
            central_peak = band.get("centralPeak", None)  # Central peak from detection
            band_valid = band.get("band_valid", False)  # Detection success
            hkl_group = band.get("hkl_group", "unknown")
            band_peak = band.get("band_peak", "0")
            band_bkg = band.get("band_bkg", "0")
            psnr = band.get("psnr", "0")
            efficientlineIntensity = band.get("efficientlineIntensity", "0")
            defficientlineIntensity = band.get("defficientlineIntensity", "0")
            #efficientDefficientRatio = efficientlineIntensity/defficientlineIntensity
            efficientDefficientRatio = (efficientlineIntensity / defficientlineIntensity
                                        if defficientlineIntensity and efficientlineIntensity
                                           and np.isfinite(efficientlineIntensity)
                                           and np.isfinite(defficientlineIntensity)
                                        else 0)
            # Append the row to data
            data.append({
                "X,Y": xy,
                "Ind": ind,
                "hkl": hkl,
                "hkl_group":hkl_group,
                "Central Line": central_line,
                "Line Distance": line_dist,
                "Band Width": band_width,
                "band_peak": band_peak,
                "band_bkg": band_bkg,
                "psnr": psnr,
                "efficientlineIntensity": efficientlineIntensity,
                "defficientlineIntensity": defficientlineIntensity,
                "efficientDefficientRatio":np.around(efficientDefficientRatio,2),
                "band_valid": band_valid
            })

    # Create a DataFrame
    df = pd.DataFrame(data).round(3)
    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False, engine='openpyxl')
    logging.info(f"Results saved to {output_path}.")

    df_filtered = df[df['band_valid'] == True]
    df_grouped = df_filtered.loc[df_filtered.groupby('Ind')['psnr'].idxmax()]
    df_grouped.to_excel(filtered_excel_path, index=False, engine='openpyxl')

from PIL import Image
import numpy as np
import os, math, logging

from PIL import Image
import numpy as np
import os, math, logging


def _square_pattern(arr: np.ndarray, tol: float = 0.95) -> np.ndarray:
    """Return a square version of *arr* (2‑D).  See previous answer for details."""
    h, w = arr.shape
    if h == w:
        return arr

    ratio = min(h, w) / max(h, w)
    if ratio < tol:
        raise ValueError(f"Pattern {h}×{w} too far from square (ratio {ratio:.3f}).")

    target = max(h, w)
    logging.warning(f"Pattern {h}×{w} not square; resizing to {target}×{target}.")
    return cv2.resize(arr, (target, target), interpolation=cv2.INTER_NEAREST)



def load_ebsd_data(source: str,
                   tile_rows: int = 10,
                   tile_cols: int = 10) -> np.ndarray:
    """
    • Folder            → load all *.png / *.bmp files.
    • Single PNG or BMP → load, square‑pad/‑resize if needed, tile.
    • .npy              → 2‑D (tile) or 4‑D (return as‑is).
    """
    IMG_EXT = (".png", ".bmp")  # allowed image extensions (lower‑case)

    # ── Folder of images ────────────────────────────────────────────
    if os.path.isdir(source):
        img_files = sorted([os.path.join(source, f)
                            for f in os.listdir(source)
                            if f.lower().endswith(IMG_EXT)])
        if not img_files:
            raise FileNotFoundError("No *.png or *.bmp files in the folder.")

        # read first image to set reference size / dtype
        with Image.open(img_files[0]) as im:
            ref = _square_pattern(np.asarray(im.convert("L")))
        h = w = ref.shape[0]
        imgs = [ref]

        for path in img_files[1:]:
            with Image.open(path) as im:
                arr = _square_pattern(np.asarray(im.convert("L")))
            if arr.shape != (h, w):
                raise ValueError(f"{path} became {arr.shape}, "
                                 f"expected {(h, w)} after squaring.")
            imgs.append(arr)

        n = len(imgs)
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / rows)
        if cols - rows > 1:
            rows, cols = cols, rows
        pad = rows * cols - n
        if pad:
            imgs.extend([np.zeros((h, w), dtype=ref.dtype)] * pad)

        ebsd_data = np.stack(imgs).reshape(rows, cols, h, w)
        logging.info(f"Loaded {n} patterns → grid {rows}×{cols}.")
        return ebsd_data

    # ── Single PNG or BMP file ──────────────────────────────────────
    if os.path.isfile(source) and source.lower().endswith(IMG_EXT):
        with Image.open(source) as im:
            arr = _square_pattern(np.asarray(im.convert("L")))
        h = w = arr.shape[0]
        tiled = np.tile(arr, (tile_rows, tile_cols, 1, 1))
        logging.info(f"Loaded single image {source} {h}×{w} → "
                     f"{tile_rows}×{tile_cols} grid.")
        return tiled

    # ── .npy file ───────────────────────────────────────────────────
    if source.lower().endswith(".npy"):
        arr = np.load(source)
        if arr.ndim == 4:  # already tiled
            logging.info(f"Loaded EBSD data {arr.shape} from {source}.")
            return arr
        if arr.ndim == 2:  # single pattern → tile
            arr = _square_pattern(arr)
            h = w = arr.shape[0]
            tiled = np.tile(arr, (tile_rows, tile_cols, 1, 1))
            logging.info(
                f"Tiled single pattern {h}×{w} to {tile_rows}×{tile_cols} grid.")
            return tiled
        raise ValueError("Loaded .npy must be 2‑D or 4‑D.")

    # ----------------------------------------------------------------
    raise ValueError("source must be a folder, a *.png / *.bmp file, or a *.npy file.")



def prepare_json_input(json_path: str,
                       n_patterns: int,
                       tile_from_single: bool = False):
    """
    Return a flat list of JSON dictionaries with length == n_patterns.

    * When `tile_from_single` is True  →  deep‑copy the first entry
      repeatedly (for the legacy *.npy route).

    * Otherwise                       →  load the file and **validate**
      that it already contains `n_patterns` items.
    """
    with open(json_path) as f:
        data = json.load(f)

    if tile_from_single:                       # .npy case
        base = data[0] if isinstance(data, list) else data
        return [copy.deepcopy(base) for _ in range(n_patterns)]

    # folder‑mode: expect ready‑made list
    if not isinstance(data, list):
        raise TypeError("JSON must be a list when loading from folder.")

    if len(data) != n_patterns:
        raise ValueError(
            f"JSON has {len(data)} entries, but {n_patterns} patterns "
            f"are loaded.  They must match 1:1.")
    return data


if __name__ == "__main__":
    #config = load_config("sumitBandDetectorOptions.yml")
    config = load_config("bandDetectorOptions.yml")

    # ----------------------------------------------------------------
    # Select *either* a .npy file *or* a folder containing PNGs
    # ----------------------------------------------------------------
    #SOURCE = "real_kikuchi.npy"  # old route
    #SOURCE = r"C:\Users\kvman\Downloads\Center\Center\1.bmp"  # old route
    #SOURCE = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\ML_1X1Patterns\ML_1X1Patterns\Med_Mn_10k_1x1_00507.png"  # 1X1RAW
    SOURCE = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\ML_4x4Patterns\ML_4x4Patterns\Med_Mn_10k_4x4_00995.png" # 4X4Raw
    #SOURCE = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\inference_testing_MLOutput\Med_Mn_10k_4x4_00995_out.png" # 1X1ML
    #jsonFile = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\ML_1X1Patterns\ML_1X1Patterns\single_Med_Mn_10k_1x1.json"
    #jsonFile = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\inference_testing_MLOutput\SingleMed_Mn_10k_4x4_00995_out.json"
    jsonFile = r"C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\inference_testing_MLOutput\SingleMed_Mn_10k_4x4_00995.json"
    # SOURCE = "patterns_folder"        # new route

    ebsd_data = load_ebsd_data(SOURCE, tile_rows=1, tile_cols=1)
    rows, cols = ebsd_data.shape[:2]
    n_patterns = rows * cols

    # ----------------------------------------------------------------
    # JSON handling
    # ----------------------------------------------------------------
    if SOURCE.lower().endswith(".npy") or SOURCE.lower().endswith(".png") or SOURCE.lower().endswith(".bmp"):
        json_input = prepare_json_input(
            jsonFile,
            n_patterns,
            tile_from_single=True  # deep‑copy first entry
        )
    else:  # folder route
        json_input = prepare_json_input(
            "bandInputData.json",
            n_patterns,
            tile_from_single=False  # expect full list
        )

    processed_results = process_kikuchi_images(
            ebsd_data,
            json_input,
            desired_hkl=config.get("desired_hkl", "002"),
            config=config,
        )

    save_results_to_excel(processed_results, "sumit_bandOutputData.xlsx")