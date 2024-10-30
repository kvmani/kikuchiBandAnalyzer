import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian(x, amp, mean, stddev):
    """Gaussian function for fitting."""
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

class LineTrimmer:
    def __init__(self, image_width, image_height):
        """
        Initializes the LineTrimmer with image bounds.
        Calculates the largest inscribed circle.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.image_center = (image_width / 2, image_height / 2)
        self.radius = 0.9*min(image_width, image_height) / 2  # Radius of the inscribed circle

    def _distance(self, x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _trim_line_to_circle(self, x1, y1, x2, y2):
        """
        Trim the central line coordinates (x1, y1, x2, y2) to ensure they stay within the inscribed circle,
        while preserving the original slope and minimizing the shift.
        Only trim the points that are outside the circle.
        If the entire line is inside the circle, return the original points unmodified.
        """
        # Get the center and radius of the circle
        cx, cy = self.image_center
        r = self.radius

        # Check if each point is inside the circle
        dist1 = self._distance(x1, y1, cx, cy)
        dist2 = self._distance(x2, y2, cx, cy)

        # If both points are inside the circle, return the original points
        if dist1 <= r and dist2 <= r:
            return x1, y1, x2, y2

        # Parametrize the line equation: (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)
        dx = x2 - x1
        dy = y2 - y1
        a = dx ** 2 + dy ** 2
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx) ** 2 + (y1 - cy) ** 2 - r ** 2
        discriminant = b ** 2 - 4 * a * c

        # If discriminant is negative, the line is completely outside the circle
        if discriminant < 0:
            raise ValueError("The line does not intersect the circle and lies completely outside.")

        # Find the two possible t values (t1 and t2) where the line intersects the circle
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        # Find the new clipped points on the circle
        new_x1, new_y1 = x1, y1  # Default to original points
        new_x2, new_y2 = x2, y2

        # Only update the points that lie outside the circle
        if dist1 > r:
            new_x1 = x1 + t1 * dx
            new_y1 = y1 + t1 * dy

        if dist2 > r:
            new_x2 = x1 + t2 * dx
            new_y2 = y1 + t2 * dy

        return new_x1, new_y1, new_x2, new_y2

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
        self.plot_band_detection=config.get('plot_band_detection',False)
        self.psnr = 0
        self.band_valid=False


    def _trim_line_to_image_bounds(self, x1, y1, x2, y2):
        """
        Trim the central line coordinates (x1, y1, x2, y2) to ensure they stay within the image bounds,
        maintaining the slope of the line.

        :param x1, y1, x2, y2: Coordinates of the central line.
        :return: Trimmed coordinates that are inside the image bounds.
        """
        image_height, image_width = self.image.shape

        # Calculate the slope of the line
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = float('inf')  # Infinite slope, vertical line

        # Find the intersection points of the line with the image boundary
        new_x1, new_y1 = self._clip_line_to_bounds(x1, y1, slope, image_width, image_height)
        new_x2, new_y2 = self._clip_line_to_bounds(x2, y2, slope, image_width, image_height)

        # Assertion to ensure the trimmed coordinates are within the image bounds
        assert 0 <= new_x1 < image_width and 0 <= new_y1 < image_height, "Trimmed x1, y1 are out of bounds."
        assert 0 <= new_x2 < image_width and 0 <= new_y2 < image_height, "Trimmed x2, y2 are out of bounds."

        return new_x1, new_y1, new_x2, new_y2

    def _clip_line_to_bounds(self, x, y, slope, image_width, image_height):
        """
        Clip a point (x, y) on the line to ensure it stays within the image bounds.

        :param x, y: Point coordinates to be clipped.
        :param slope: Slope of the line.
        :param image_width: Width of the image.
        :param image_height: Height of the image.
        :return: Trimmed coordinates (new_x, new_y) that lie within the image bounds.
        """
        if x < 0:  # Clip to the left boundary (x = 0)
            new_x = 0
            new_y = y + slope * (new_x - x)
        elif x > image_width - 1:  # Clip to the right boundary (x = image_width - 1)
            new_x = image_width - 1
            new_y = y + slope * (new_x - x)
        else:
            new_x = x
            new_y = y

        if new_y < 0:  # Clip to the top boundary (y = 0)
            new_y = 0
            if slope != float('inf'):
                new_x = x + (new_y - y) / slope
        elif new_y > image_height - 1:  # Clip to the bottom boundary (y = image_height - 1)
            new_y = image_height - 1
            if slope != float('inf'):
                new_x = x + (new_y - y) / slope

        # Ensure the new point stays within bounds
        new_x = np.clip(new_x, 0, image_width - 1)
        new_y = np.clip(new_y, 0, image_height - 1)

        return new_x, new_y

    def detect(self):
        """
        Detect the band using intensity profile in a rectangular area around the band.
        Correct the band width using the scaling factor and store it as a class attribute.
        :return: Dictionary with bandWidth and success status.
        """
        # Extract the rectangular region and sample the intensities
        x1, y1, x2, y2 = self.central_line
        trimmer = LineTrimmer(self.image.shape[0], self.image.shape[1])
        #x1, y1, x2, y2 = self._trim_line_to_image_bounds(x1, y1, x2, y2)
        x1, y1, x2, y2 = trimmer._trim_line_to_circle(x1, y1, x2, y2)
        self.central_line = [x1, y1, x2, y2]
        rect_width = self.config.get('rectWidth', 20)  # Width of the rectangle
        logging.debug(f"Central line: {self.central_line}, Rectangle width: {rect_width}")
        #avg_rect_intensity, avg_img_intensity = self.extract_rotated_rectangle_properties(x1, y1, x2, y2, rect_width, plotResults=True)
        rect_area, rotated_image, rect_corners = self.extract_rotated_rectangle(x1, y1, x2, y2, rect_width)
        summed_profile = np.sum(self.sample_intensities(rect_area), axis=0)
        # Detect band start, end, and central peak in the profile
        # band_start, band_end, central_peak = self.detect_edges(summed_profile)
        result = self.detect_edges(summed_profile)
        result["central_line"]=self.central_line
        band_start, band_end, central_peak,psnr,band_valid=result["bandStart"],result["bandEnd"],result["centralPeak"],result["psnr"], result["band_valid"]

        if band_start is not None and band_end is not None and band_valid:
            # Calculate the band width and adjust for the scaling factor
            raw_band_width = band_end - band_start
            self.band_width = raw_band_width / self.scaling_factor  # Adjust band width based on the scaling factor
            self.psnr=psnr
            if self.band_width <self.config["rectWidth"]*.8 and self.band_width>self.config["rectWidth"]*.1:

                band_valid = True
                self.band_valid=True
                logging.debug(f"Band width detected: {self.band_width}")
            else:
                band_valid = False
                self.band_valid=False
        else:
            self.band_width = 0
            self.band_valid = False
            logging.info("Band width detection failed.")
        if self.plot_band_detection:
            self.plot_debug(rotated_image, rect_corners, rect_area, summed_profile, band_start, band_end,
                            central_peak)

        # self.band_valid=band_valid
        # self.psnr=psnr

        final_result = {
            **result,  # Merging result with psnr
            "bandWidth": self.band_width,
            "band_valid": band_valid,
            "psnr":self.psnr
        }



        return final_result



    def extract_rotated_rectangle_properties(self, x1, y1, x2, y2, band_width, plotResults=False):
        """
        Extract a narrow band (thick line) around the central line (x1, y1) to (x2, y2) of the specified width.
        Handles cases where part of the band exceeds the image boundary by filling those regions with zeros.
        Also computes the average pixel intensity inside the band and the entire image average using a circular mask.

        :param x1, y1, x2, y2: Coordinates of the central line.
        :param band_width: Width of the band around the central line.
        :param plotResults: If True, plot intermediate results for debugging.
        :return: (avg_band_intensity, avg_img_intensity)
        """

        # Create a blank mask the same size as the image
        mask = np.zeros_like(self.image, dtype=np.uint8)

        # Draw a thick line on the mask to represent the band
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=int(band_width/2))

        # Ensure the mask has the same number of channels as the image (if the image is multi-channel)
        # if len(self.image.shape) == 3 and self.image.shape[2] > 1:
        #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Extract the region inside the band by applying the mask to the image
        band_region = cv2.bitwise_and(self.image, self.image, mask=mask)

        # Calculate the average intensity inside the band
        avg_band_intensity = np.sum(band_region) / (np.sum(mask > 0) + 1e-5)  # Avoid division by zero

        # Create a circular mask for the entire image (biggest possible circle)
        center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        radius = min(center)
        circular_mask = np.zeros_like(self.image, dtype=np.uint8)
        cv2.circle(circular_mask, center, radius, 255, -1)

        # Apply the circular mask to the image
        masked_image = cv2.bitwise_and(self.image, self.image, mask=circular_mask)

        # Calculate the average intensity of the circularly masked image
        avg_img_intensity = np.sum(masked_image) / (np.sum(circular_mask > 0) + 1e-5)

        # Plot results if plotResults is True
        if plotResults:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original Image with Central Line
            axes[0].imshow(self.image, cmap='gray')
            axes[0].set_title('Original Image with Central Line')
            axes[0].plot([x1, x2], [y1, y2], 'r-', label='Central Line')

            # Masked Band Region
            axes[1].imshow(band_region, cmap='gray')
            axes[1].set_title('Extracted Band Region (Masked)')

            # Circular Mask Applied to Original Image
            masked_overlay = np.copy(self.image)
            masked_overlay[circular_mask == 0] = 0
            axes[2].imshow(masked_overlay, cmap='gray')
            axes[2].set_title(f'Image with Circular Mask\n'
                              f'{avg_band_intensity=}\n'
                              f'{avg_img_intensity}')
            circle = plt.Circle(center, radius, color='blue', fill=False)
            axes[2].add_patch(circle)

            plt.tight_layout()
            plt.show()

        return avg_band_intensity, avg_img_intensity

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
        band_valid=False
        smoothed_profile = gaussian_filter1d(profile, sigma=self.config.get("smoothing_sigma", 2))

        # Step 2: Find the maximum (central peak) in the smoothed profile
        central_peak_index = np.argmax(smoothed_profile)
        peak_max=0
        noise_average=0
        psnr_value=0
        left_min_index=0
        right_min_index=0
        #central_peak_index=0

        if central_peak_index !=0 and central_peak_index!= smoothed_profile.size-1:

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
                if psnr_value>self.config["min_psnr"]:
                    band_valid=True
            else:
                psnr_value = 0  # Handle case where noise is zero to avoid division by zero

            if left_min_index==0 or right_min_index==smoothed_profile.size:
                band_valid = False
                #print("index is either first or last of array!!!")

        logging.info(
            f"Central peak detected at: {central_peak_index}, with band start at {left_min_index} and end at {right_min_index}")
        logging.info(f"PSNR value: {psnr_value}")

        # Step 7: Return the results in the dictionary
        return {
            "band_peak":peak_max,
            "band_bkg":noise_average,
            "bandStart": left_min_index,
            "bandEnd": right_min_index,
            "centralPeak": central_peak_index,
            "psnr": psnr_value,  # Include PSNR in the results dictionary
            "band_valid": band_valid
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

        ax[1, 1].set_title(f"Summed Intensity Profile (hkl: {self.hkl})\n"
                           f"BandWidth={self.band_width} \n PSNR={np.around(self.psnr,2)}\n "
                           f"valid?{self.band_valid}")

        ax[1, 1].legend()

        plt.tight_layout()
        plt.show()
