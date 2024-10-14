import logging
import json
import cv2
import yaml
from strategies import GradientBandDetector, GaussianBandDetector, RectangularAreaBandDetector

# Set up logging
logging.basicConfig(filename="bandDetector.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path="bandDetectorOptions.yml"):
    """
    Loads configuration from a YAML file.
    :param config_path: Path to the YAML configuration file.
    :return: Dictionary with configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_json(json_path="bandInputData.json"):
    """
    Loads band data from a JSON file.
    :param json_path: Path to the JSON input file.
    :return: List of dictionaries with band data.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


class BandDetector:
    def __init__(self, image_path, points, strategy):
        """
        Initializes the band detector for multiple bands in an image.

        :param image_path: Path to the input image.
        :param points: List of dictionaries with "hkl", "central_line", and "refWidth" for each band.
        :param strategy: Strategy string ('gradient', 'gaussian', 'rectangular_area') to choose the detection strategy.
        """
        self.image_path = image_path
        self.points = points  # Multiple bands data
        self.strategy = strategy
        self.image = self.load_image()
        self.config = load_config()

    def load_image(self):
        """
        Loads an image from the given path and converts it to grayscale.

        :return: Grayscale image.
        """
        image = cv2.imread(self.image_path)
        if len(image.shape) == 3:  # Check if the image is not grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.info(f"Image loaded from {self.image_path}.")
        return image

    def detect_bands(self):
        """
        Detects the bands for each point in the `points` list.

        :return: A list of dictionaries with detection results for each band.
        """
        results = []

        for point in self.points:
            hkl = point["hkl"]
            central_line = point["central_line"]
            ref_width = point["refWidth"]

            # Choose the strategy and detect the band
            if self.strategy == 'gradient':
                detector = GradientBandDetector(self.image, central_line, self.config)
            elif self.strategy == 'gaussian':
                detector = GaussianBandDetector(self.image, central_line, self.config)
            elif self.strategy == 'rectangular_area':
                detector = RectangularAreaBandDetector(self.image, central_line, self.config)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            result = detector.detect()
            result["hkl"] = hkl  # Add hkl label to the result
            result["central_line"] = central_line  # Add central line for reference
            result["refWidth"] = ref_width  # Add reference width

            # Store each band's result
            results.append(result)

            logging.info(f"Band detection for {hkl} completed using {self.strategy} strategy.")

        return results


def process_images(json_input):
    """
    Processes multiple images and their corresponding bands.

    :param json_input: List of dictionaries with image data and band points.
    :return: Updated list of dictionaries with band detection results.
    """
    processed_data = []

    for entry in json_input:
        image_path = entry["patternFileName"]
        points = entry["points"]

        # Load strategy from config
        config = load_config()
        strategy = config.get("strategy", "gradient")

        # Initialize BandDetector for the current image
        band_detector = BandDetector(image_path, points, strategy)
        results = band_detector.detect_bands()

        # Add results to the processed entry
        entry["bands"] = results

        processed_data.append(entry)

    return processed_data


def save_results_to_json(results, output_path="bandOutputData.json"):
    """
    Saves the processed results to a JSON file.

    :param results: List of dictionaries with processed band data.
    :param output_path: Path to save the JSON output file.
    """
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)
    logging.info(f"Results saved to {output_path}.")


if __name__ == "__main__":
    # Load input from JSON
    json_input = load_json("bandInputData.json")

    # Process each image and its bands
    processed_results = process_images(json_input)

    # Save the results to JSON
    save_results_to_json(processed_results)
