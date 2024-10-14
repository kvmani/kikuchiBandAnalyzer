import cv2
import yaml
import logging
from strategies import GradientBandDetector, GaussianBandDetector, RectangularAreaBandDetector
from configLoader import load_config
import matplotlib.pyplot as plt
# Set up logging
logging.basicConfig(filename="bandDetector.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class BandDetector:
    def __init__(self, image_path, central_line, strategy):
        """
        Initializes the band detector with image and strategy.

        :param image_path: Path to the input image.
        :param central_line: A tuple (x1, y1, x2, y2) defining the central line of the band.
        :param strategy: Strategy string ('gradient', 'gaussian', 'rectangular_area') to choose the detection strategy.
        """
        self.image_path = image_path
        self.central_line = central_line
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

    def detect_band(self):
        """
        Detects the band width based on the selected strategy.

        :return: A dictionary with results from the detection strategy.
        """
        if self.strategy == 'gradient':
            detector = GradientBandDetector(self.image, self.central_line, self.config)
        elif self.strategy == 'gaussian':
            detector = GaussianBandDetector(self.image, self.central_line, self.config)
        elif self.strategy == 'rectangular_area':
            detector = RectangularAreaBandDetector(self.image, self.central_line, self.config)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        result = detector.detect()
        logging.info(f"Band detection completed using {self.strategy} strategy.")
        return result




if __name__ == "__main__":
    # Example usage
    image_path = r"testData/ML_kikuchi_test_1.png"
    central_line = (65, 44, 237, 151)  # Example central line (x1, y1, x2, y2)
    #central_line = (142, 239, 234, 87)  # Example central line (x1, y1, x2, y2)


    image_path = r"testData/poorKikuci.png"
    #central_line = (63, 105, 210, 220)  # Example central line (x1, y1, x2, y2)
    central_line = (141, 304, 243, 130)  # Example central line (x1, y1, x2, y2)

    # image_path = r"../../data/testingData/HR_test_2x2_random3_Ni_1840_0000.jpg"
    # #central_line = (63, 105, 210, 220)  # Example central line (x1, y1, x2, y2)
    # central_line = (281, 688, 1670, 1351)  # Example central line (x1, y1, x2, y2)

    # Load strategy from config
    config = load_config()
    strategy = config.get("strategy", "gradient")

    # Initialize and run band detection
    band_detector = BandDetector(image_path, central_line, strategy)
    result = band_detector.detect_band()

    # Plot results

