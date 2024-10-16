import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian(x, amp, mean, stddev):
    """Gaussian function for fitting."""
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


class BandDetectionStrategy:
    def __init__(self, image, central_line, config,hkl):
        self.image = image
        self.central_line = central_line
        self.hkl = hkl
        self.config = config

    def detect(self):
        """
        Abstract method to detect band. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GradientBandDetector(BandDetectionStrategy):
    def detect(self):
        """
        Detect the band using gradient-based method.
        :return: Dictionary with bandWidth, perpendicularLine, success status.
        """
        # Get the midpoint of the central line
        x1, y1, x2, y2 = self.central_line
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Construct perpendicular line
        perp_len = self.config.get("perpendicular_line_length", 20)
        perpendicular_line = [(mid_x, mid_y - perp_len // 2), (mid_x, mid_y + perp_len // 2)]

        # Sample the line profile
        line_profile = cv2.line(self.image, perpendicular_line[0], perpendicular_line[1], (255, 0, 0))

        # Smooth the line profile
        smoothed_profile = gaussian_filter1d(line_profile, sigma=self.config.get("smoothing_sigma", 2))

        # Detect the start and end of the band using gradient threshold
        gradient_threshold = self.config.get("gradient_threshold", 10)
        # Implement the gradient detection logic...

        # Return results in the expected dictionary format
        result = {
            "bandWidth": None,  # Replace with actual width calculation
            "perpendicularLine": perpendicular_line,
            "success": False  # Set to True after finding the band
        }
        return result


class GaussianBandDetector(BandDetectionStrategy):
    def detect(self):
        """
        Detect the band using Gaussian curve fitting.
        :return: Dictionary with bandWidth, perpendicularLine, success status, and fitted Gaussian curve.
        """

        # Fit a Gaussian curve to the intensity profile
        def gaussian(x, amp, mean, stddev):
            return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

        # Assume we have the intensity profile...
        # Fit the curve...

        result = {
            "bandWidth": None,  # Replace with actual width calculation
            "perpendicularLine": None,
            "success": False,  # Set to True after fitting successfully
            "fitted_gaussian": None  # Return the fitted Gaussian curve for plotting
        }
        return result


class RectangularAreaBandDetector:
    def __init__(self, image, central_line, config,hkl):
        """
        Initializes the band detector for rectangular area-based strategy.

        :param image: Input grayscale image.
        :param central_line: A tuple (x1, y1, x2, y2) defining the central line of the band.
        :param config: Configuration dictionary from YAML file.
        """
        self.image = image
        self.central_line = central_line
        self.config = config
        self.hkl = hkl
        self.debug = config.get('debug', False)

    def detect(self):
        """
        Detect the band using intensity profile in a rectangular area around the band.
        Correct the band width using the scaling factor and store it as a class attribute.
        :return: Dictionary with bandWidth and success status.
        """
        # Extract the rectangular region and sample the intensities
        x1, y1, x2, y2 = self.central_line
        rect_width = self.config.get('rectWidth', 20)  # Width of the rectangle
        logging.info(f"Central line: {self.central_line}, Rectangle width: {rect_width}")
        rect_area, rotated_image, rect_corners = self.extract_rotated_rectangle(x1, y1, x2, y2, rect_width)
        summed_profile = np.sum(self.sample_intensities(rect_area), axis=0)

        # Detect band start, end, and central peak in the profile
        # band_start, band_end, central_peak = self.detect_edges(summed_profile)
        result = self.detect_edges(summed_profile)
        result["central_line"]=self.central_line
        band_start, band_end, central_peak,psnr=result["bandStart"],result["bandEnd"],result["centralPeak"],result["psnr"]

        if band_start is not None and band_end is not None:
            # Calculate the band width and adjust for the scaling factor
            raw_band_width = band_end - band_start
            self.band_width = raw_band_width / self.scaling_factor  # Adjust band width based on the scaling factor
            success = True
            logging.info(f"Band width detected: {self.band_width}")
        else:
            self.band_width = None
            success = False
            logging.info("Band width detection failed.")
        if self.debug:
            self.plot_debug(rotated_image, rect_corners, rect_area, summed_profile, band_start, band_end,
                            central_peak)

        final_result = {
            "bandWidth": self.band_width,
            "success": success,
            **result  # Merging result with psnr
        }

        print(f"Final result: {final_result}")  # Add this for debugging

        return final_result

    def extract_rotated_rectangle(self, x1, y1, x2, y2, rect_width):
        """
        Extract a rotated rectangular area around the central line.
        First shift the image so that the midpoint of the central line is at the center of the image.
        The central line is then aligned to the Y-axis (vertical direction).

        :param x1, y1, x2, y2: Coordinates of the central line.
        :param rect_width: Width of the rectangle around the central line.
        :return: Cropped rotated rectangle region, rotated image, and the rectangle corners.
        """
        # Calculate the center of the central line
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate image center
        image_center_x, image_center_y = self.image.shape[1] // 2, self.image.shape[0] // 2

        # Calculate the translation required to move the band's midpoint to the image center
        dx = image_center_x - mid_x
        dy = image_center_y - mid_y

        # Step 1: Shift the image to place the band's midpoint at the image center
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_image = cv2.warpAffine(self.image, translation_matrix, (self.image.shape[1], self.image.shape[0]))

        # Recalculate the shifted coordinates of the central line
        shifted_x1, shifted_y1 = x1 + dx, y1 + dy
        shifted_x2, shifted_y2 = x2 + dx, y2 + dy

        # Step 2: Get the angle of the shifted central line and rotate the image
        angle = np.arctan2(shifted_y2 - shifted_y1, shifted_x2 - shifted_x1) * 180 / np.pi

        # Define the rectangle (rect_width x line length)
        line_length = np.sqrt((shifted_x2 - shifted_x1) ** 2 + (shifted_y2 - shifted_y1) ** 2)
        rect_size = (rect_width, line_length)  # Rect width across the band, length along the band

        # Step 3: Rotate the shifted image so that the band is aligned vertically (along Y-axis)
        rot_matrix = cv2.getRotationMatrix2D((image_center_x, image_center_y), angle - 90, 1.0)
        rotated_image = cv2.warpAffine(shifted_image, rot_matrix, (shifted_image.shape[1], shifted_image.shape[0]))

        # Step 4: Extract the rectangle from the rotated image
        box_x1, box_y1 = int(image_center_x - rect_size[0] / 2), int(image_center_y - rect_size[1] / 2)
        box_x2, box_y2 = int(image_center_x + rect_size[0] / 2), int(image_center_y + rect_size[1] / 2)

        # Rectangle corners in the original image (for plotting)
        rect_corners = np.array([
            [box_x1, box_y1],
            [box_x2, box_y1],
            [box_x2, box_y2],
            [box_x1, box_y2]
        ])

        return rotated_image[box_y1:box_y2, box_x1:box_x2], rotated_image, rect_corners

    def sample_intensities(self, upright_rectangle):
        """
        Sample the intensities from the upright rectangular region.
        :param upright_rectangle: Aligned rectangle region.
        :return: Sampled intensity profile and scaling factor.
        """
        num_rows = upright_rectangle.shape[0]  # Sampling across the width of the band (which is now vertical)
        num_columns = 200  # Sampling along the length of the band

        # Calculate the scaling factor (since the width is resized)
        original_width = upright_rectangle.shape[1]  # Original width
        scaling_factor = num_columns / original_width

        # Resize the image
        resized_image = cv2.resize(upright_rectangle, (num_columns, num_rows,))  # Correct shape order
        logging.info(
            f"Sampled intensities from the rectangle: {resized_image.shape} with scaling factor: {scaling_factor}")

        # If debugging is enabled, plot the intensity profiles
        # if self.debug:
        #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        #
        #     # Plot intensity profiles row by row for the upright_rectangle
        #     ax[0].imshow(upright_rectangle, cmap='gray')
        #     ax[0].set_title("Upright Rectangle")
        #
        #     # Plot intensity profiles row by row for the resized_image
        #     ax[1].imshow(resized_image, cmap='gray')
        #     ax[1].set_title("Resized Image")
        #
        #     for row in range(num_rows):
        #         ax[0].plot(upright_rectangle[row, :], label=f"Row {row}")
        #         ax[1].plot(resized_image[row, :], label=f"Row {row}")
        #
        #     ax[0].set_ylabel("Intensity")
        #     ax[1].set_ylabel("Intensity")
        #     ax[0].set_xlabel("Pixel Index")
        #     ax[1].set_xlabel("Pixel Index")
        #
        #     plt.tight_layout()
        #     plt.show()

        # Store the scaling factor as a class attribute
        self.scaling_factor = scaling_factor

        return resized_image



    def detect_edges(self, profile):
        """
        Detect edges on the intensity profile using minima detection and calculate PSNR.
        :param profile: 1D intensity profile.
        :return: Indices of the band start (left_min), band end (right_min), central peak, and PSNR value.
        """
        # Step 1: Smooth the profile to reduce noise
        smoothed_profile = gaussian_filter1d(profile, sigma=self.config.get("smoothing_sigma", 2))

        # Step 2: Find the maximum (central peak) in the smoothed profile
        central_peak_index = np.argmax(smoothed_profile)
        peak_max = smoothed_profile[central_peak_index]  # Peak maximum

        # Step 3: Find the minimum on the left side of the central peak
        left_min_index = np.argmin(smoothed_profile[:central_peak_index])  # Minimum before the peak
        left_min = smoothed_profile[left_min_index]

        # Step 4: Find the minimum on the right side of the central peak
        right_min_index = np.argmin(
            smoothed_profile[central_peak_index:]) + central_peak_index  # Minimum after the peak
        right_min = smoothed_profile[right_min_index]

        # Step 5: Calculate the average of the left and right minima
        noise_average = (left_min + right_min) / 2

        # Step 6: Calculate PSNR (Peak Max divided by the average of left and right minima)
        if noise_average != 0:
            psnr_value = peak_max / noise_average
        else:
            psnr_value = np.inf  # Handle case where noise is zero to avoid division by zero

        logging.info(
            f"Central peak detected at: {central_peak_index}, with band start at {left_min_index} and end at {right_min_index}")
        logging.info(f"PSNR value: {psnr_value}")

        # Step 7: Return the results in the dictionary
        return {
            "bandStart": left_min_index,
            "bandEnd": right_min_index,
            "centralPeak": central_peak_index,
            "psnr": psnr_value  # Include PSNR in the results dictionary
        }

    def plot_debug(self, rotated_image, rect_corners, rect_area, summed_profile, band_start, band_end, central_peak):
        """
        Plot debugging figures to visualize the image, rectangle, and intensity profile.
        :param rotated_image: The image after rotation.
        :param rect_corners: The corners of the detected rectangle.
        :param rect_area: The extracted rectangular region after rotation.
        :param summed_profile: Summed intensity profile across the rectangle.
        :param band_start: Index of the detected band start.
        :param band_end: Index of the detected band end.
        :param central_peak: Index of the central peak in the intensity profile.
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Plot 1: Original image with band start/end points and lines parallel to central line
        ax[0, 0].imshow(self.image, cmap='gray')
        ax[0, 0].plot([self.central_line[0], self.central_line[2]],
                      [self.central_line[1], self.central_line[3]], 'r-', label='Central Line')

        if band_start is not None and band_end is not None:
            # Calculate the vector along the central line
            dx = self.central_line[2] - self.central_line[0]
            dy = self.central_line[3] - self.central_line[1]
            line_length = np.sqrt(dx ** 2 + dy ** 2)
            unit_vector_x = dx / line_length
            unit_vector_y = dy / line_length

            # Calculate perpendicular unit vector
            perp_vector_x = -unit_vector_y  # Negative to rotate by 90 degrees
            perp_vector_y = unit_vector_x

            # Midpoint of the central line (this is where band_start and band_end offset is measured)
            mid_x, mid_y = (self.central_line[0] + self.central_line[2]) / 2, (
                    self.central_line[1] + self.central_line[3]) / 2

            # Convert band start/end profile indices into offsets in the image space
            start_offset = (band_start - central_peak) / self.scaling_factor  # Use central_peak instead of midpoint
            end_offset = (band_end - central_peak) / self.scaling_factor  # Use central_peak instead of midpoint

            # Find the band start and end points in the image coordinates by moving along the perpendicular
            start_x = mid_x + start_offset * perp_vector_x
            start_y = mid_y + start_offset * perp_vector_y
            end_x = mid_x + end_offset * perp_vector_x
            end_y = mid_y + end_offset * perp_vector_y

            # Draw the band start and end lines parallel to the central line
            # Start line parallel to central line
            start_line_x1 = start_x - (dx / 2)
            start_line_y1 = start_y - (dy / 2)
            start_line_x2 = start_x + (dx / 2)
            start_line_y2 = start_y + (dy / 2)

            # End line parallel to central line
            end_line_x1 = end_x - (dx / 2)
            end_line_y1 = end_y - (dy / 2)
            end_line_x2 = end_x + (dx / 2)
            end_line_y2 = end_y + (dy / 2)

            # Plot the band start and end lines (parallel to the central line)
            ax[0, 0].plot([start_line_x1, start_line_x2], [start_line_y1, start_line_y2], 'g--',
                          label='Band Start Line')
            ax[0, 0].plot([end_line_x1, end_line_x2], [end_line_y1, end_line_y2], 'g--', label='Band End Line')

        # Step 2: Plot the rectangle with reverse rotation to align with the band in original image coordinates
        angle = np.arctan2(dy, dx) * 180 / np.pi - 90
        inverse_rot_matrix = cv2.getRotationMatrix2D((mid_x, mid_y), -angle, 1.0)  # Reverse the rotation

        # Apply inverse rotation to the corners
        rotated_rect_corners = np.dot(rect_corners, inverse_rot_matrix[:, :2].T) + inverse_rot_matrix[:, 2]

        # # Plot the original rectangle
        # rect = plt.Polygon(rotated_rect_corners, fill=None, edgecolor='b')
        # ax[0, 0].add_patch(rect)
        ax[0, 0].set_title("Original Image with Band Start/End and Rectangle")

        # Plot 3: Upright rectangle region after rotation with bandwidth in title
        ax[0, 1].imshow(rotated_image, cmap='gray')
        rect_patch = plt.Polygon(rect_corners, fill=None, edgecolor='b')
        ax[0, 1].add_patch(rect_patch)
        # Add band width to title
        if self.band_width is not None:
            ax[0, 1].set_title(f"Rotated Image with Rectangle (bandwidth={self.band_width:.2f})")
        else:
            ax[0, 1].set_title("Rotated Image with Rectangle")

        # Plot 4: Summed intensity profile with detected edges
        # ax[1, 0].imshow(rect_area, cmap='gray')
        # ax[1, 0].set_title("Upright Rectangular Region")
        # Plot 4: Upright rectangle region after rotation with detected band edges
        ax[1, 0].imshow(rect_area, cmap='gray')
        ax[1, 0].set_title("Upright Rectangular Region")

        # Convert band start/end profile indices into image space considering the scaling factor
        if band_start is not None and band_end is not None:
            # The rectangle was resized, so we need to adjust band_start and band_end back to the original dimensions
            start_x = int(band_start / self.scaling_factor)
            end_x = int(band_end / self.scaling_factor)

            # Draw vertical lines at the detected band start and band end positions in the rectangle
            ax[1, 0].axvline(x=start_x, color='g', linestyle='--', label='Band Start')
            ax[1, 0].axvline(x=end_x, color='r', linestyle='--', label='Band End')

        # Adding a legend for band start and end
        ax[1, 0].legend()

        # Plot 5: Intensity profile and detected band start/end and central peak
        ax[1, 1].plot(summed_profile, label='Summed Intensity')
        if band_start is not None and band_end is not None:
            ax[1, 1].axvline(x=band_start, color='g', linestyle='--', label='Band Start')
            ax[1, 1].axvline(x=band_end, color='r', linestyle='--', label='Band End')
        if central_peak is not None:
            ax[1, 1].axvline(x=central_peak, color='b', linestyle=':', label='Central Peak')

        # ax[1, 1].set_title(f"Summed Intensity Profile\n BandWidth={self.band_width} ")

        ax[1, 1].set_title(f"Summed Intensity Profile (hkl: {self.hkl})\nBandWidth={self.band_width}")

        ax[1, 1].legend()

        plt.tight_layout()
        plt.show()
