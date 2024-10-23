import logging
import json
import cv2
import numpy as np
import yaml
from strategies import RectangularAreaBandDetector
import pandas as pd
import dask.array as da
# Set up logging
logging.basicConfig(filename="bandDetector.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
import matplotlib.pyplot as plt

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

    def _load_config(self, config_path="bandDetectorOptions.yml"):
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
            central_line = point["central_line"]
            line_mid_xy = point["line_mid_xy"]
            line_dist = point["line_dist"]
            hkl_group = point.get("hkl_group", "unknown")
            if hkl_group in self.desired_hkl:
            # Detect the band, using RectangularAreaBandDetector
                result = self._detect_band(central_line, hkl)
                result["hkl"] = hkl  # Add hkl label to the result
                result["line_mid_xy"] = line_mid_xy  # Add line midpoint
                result["line_dist"] = line_dist  # Add line distance
                result["hkl_group"] = hkl_group
                results.append(result)
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


def process_kikuchi_images(ebsd_data, json_input, desired_hkl='111', config=None):
    """
    Processes multiple Kikuchi images (in memory) and their corresponding bands for each pixel in the EBSD dataset.

    :param ebsd_data: 2D numpy array (m x n) where each entry is a Kikuchi pattern (2D numpy array).
    :param json_input: List of dictionaries with band points.
    :param config: Configuration dictionary, optional. If None, default configuration will be loaded.
    :return: Updated list of dictionaries with band detection results for each pixel.
    """
    processed_data = []

    # Iterate over each pixel in the EBSD dataset
    ncol = ebsd_data.shape[1]
    for row in range(ebsd_data.shape[0]):
        for col in range(ebsd_data.shape[1]):
            image = ebsd_data[row, col]  # Get Kikuchi image at the current pixel

            # Process corresponding JSON entry (assuming one-to-one mapping of json_input to images)
            for entry in json_input:
                points = entry["points"]
                # Initialize BandDetector for the current image and pixel
                band_detector = BandDetector(image=image, points=points, desired_hkl=desired_hkl, config=config)
                try:
                    results = band_detector.detect_bands()
                except Exception as e:
                    print(f"pattern: [{row}{col}] An error occurred while detecting bands: {e}")
                    results = []  # You can also set a default value or take other appropriate actions
                    logging.warning(f"exception in pattern id : [{row}{col}] did not have any identifed bands !!!")
                #results = band_detector.detect_bands()
                if len(results)==0:
                    logging.warning(f"pattern id : [{row}{col}] did not have any identifed bands !!!")
                # Add results to the processed entry
                processed_entry = entry.copy()  # Copy the original entry to avoid overwriting
                processed_entry["bands"] = results
                processed_entry["x,y"] = [row, col]  # Update x,y with the current pixel coordinates
                processed_entry["ind"] = row*ncol+col

                processed_data.append(processed_entry)


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


def save_results_to_excel(results, output_path="bandOutputData.xlsx"):
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
                "band_valid": band_valid
            })

    # Create a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False, engine='openpyxl')
    logging.info(f"Results saved to {output_path}.")
    df_filtered = df[df['band_valid'] == True]
    df_grouped = df_filtered.loc[df_filtered.groupby('Ind')['psnr'].idxmax()]
    df_grouped.to_excel('filtered_band_data.xlsx', index=False, engine='openpyxl')


if __name__ == "__main__":
    # Example EBSD data (2D array of Kikuchi images)
    # ebsd_data = np.random.rand(10, 10, 128, 128)  # Example 10x10 grid of Kikuchi images of size 128x128

    ebsd_data = np.load('real_kikuchi.npy')

    # plt.imshow(ebsd_data)
    # plt.show()
    ebsd_data = np.tile(ebsd_data, (2, 2, 1, 1))
    with open("bandInputData.json", "r") as json_file:
        json_input = json.load(json_file)

    # Load input from JSON (new structure with "x,y", "ind", "points")
    #json_input = load_json("bandInputData.json")

    # Process the EBSD data and corresponding bands
    processed_results = process_kikuchi_images(ebsd_data, json_input,desired_hkl="111")

    # Save the results to Excel and JSON
    save_results_to_excel(processed_results, "bandOutputData.xlsx")
