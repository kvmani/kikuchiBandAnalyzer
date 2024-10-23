import numpy as np
import matplotlib.pyplot as plt
import cv2


def create_simulated_kikuchi_pattern(image_size=(230, 230), band_width=10):
    """
    Creates a simulated Kikuchi pattern image with 3 main lines (diagonal, vertical center, and horizontal center)
    and additional random lines. The image will have bands with pixel value bandIntsityValue and the rest set to 0.

    Args:
        image_size (tuple): The size of the image (height, width).
        band_width (int): The width of each line/band.

    Returns:
        np.array: The simulated Kikuchi pattern image.
    """
    height, width = image_size
    image = np.zeros((height, width), dtype=np.uint16)
    bandIntsityValue = np.iinfo(image.dtype).max

    # 1. Diagonal line (from top-left to bottom-right)
    cv2.line(image, (0, 0), (width - 1, height - 1), bandIntsityValue, band_width)

    cv2.line(image, (0, 100), (width - 1-100, height - 1), bandIntsityValue, band_width)

    # 2. Vertical line in the center

    cv2.line(image, (width // 2-50, 0), (width // 2-50, height - 1), int(bandIntsityValue/2), int(band_width/2))

    # 3. Horizontal line in the center
    cv2.line(image, (0, height // 2-60), (width - 1, height // 2-60), int(bandIntsityValue/2), int(band_width/2))

    # 4-10. Add random lines (bands) to meet the 10 bands requirement
    for i in range(7):
        x1, y1 = np.random.randint(50, width - 50), np.random.randint(0, height)
        x2, y2 = np.random.randint(50, width - 50), np.random.randint(0, height)
        cv2.line(image, (x1, y1), (x2, y2), int(bandIntsityValue/(i+4)), int(band_width / 3))

    #Plot the image for confirmation
    plt.imshow(image, cmap='gray', origin='lower')
    plt.title("Simulated Kikuchi Pattern")
    # plt.show()

    return image


def calculate_expected_rho_theta(image_size=(230, 230)):
    """
    Calculates the expected rho and theta for the 3 main lines:
    - Diagonal line
    - Vertical center line
    - Horizontal center line

    Args:
        image_size (tuple): The size of the image (height, width).

    Returns:
        list of tuples: List of (rho, theta) for each line.
    """
    height, width = image_size
    center_x, center_y = width / 2, height / 2

    # Diagonal line (from top-left to bottom-right)
    rho_diagonal = 0  # Diagonal line passes through the origin
    theta_diagonal = np.pi / 4  # 45 degrees

    # Vertical center line
    rho_vertical = center_x  # Distance from the center to the line
    theta_vertical = np.pi / 2  # 90 degrees (vertical line)

    # Horizontal center line
    rho_horizontal = center_y
    theta_horizontal = 0  # Horizontal line

    expected_values = [
        (rho_diagonal, theta_diagonal),
        (rho_vertical, theta_vertical),
        (rho_horizontal, theta_horizontal)
    ]

    return expected_values


def find_lines_from_rho_theta(image, rho_theta_data):
    """
    Finds the line coordinates from rho and theta values and superimposes them on the image.

    Args:
        image (np.array): Input Kikuchi image.
        rho_theta_data (list of dicts): List of Radon peak information (containing rho and theta).

    Returns:
        None. Superimposes the lines on the image and plots them.
    """
    height, width = image.shape

    # The origin for rho lies at the middle of the image height-wise (center_y)
    center_y = height / 2  # Middle of the image in the vertical direction
    fig, ax = plt.subplots()

    # Plot the image on the axes
    ax.imshow(image, cmap='gray', origin='lower')
    #plt.imshow(image, cmap='gray', origin='lower')
    for i,line_data in enumerate(rho_theta_data):
        # if i>4:
        #     break
        rho =  -line_data['rho']  # Rho value
        theta = np.pi-line_data['theta']  # Theta in radians (assumed already converted if needed)

        # Debugging: Print the rho and theta values for each line
        print(f"Rho: {rho}, Theta: {theta} radians")

        # Create an array for x values from 0 to width (230)
        x_vals = np.linspace(-230, 230, num=width)

        # Calculate the y-values based on the line equation in polar coordinates
        if not np.allclose(np.sin(theta),0):
            sintheta = np.sin(theta)
            if rho>-0.1:
                y_vals = (rho - x_vals * np.cos(theta)) / np.sin(theta)
            else:
                y_vals = (center_y+rho - x_vals * np.cos(theta)) / np.sin(theta)
            #if abs(sintheta)<0.01:
            print(f"{i=} {theta=}, {sintheta=} {np.max(y_vals)=}")
            #y_vals += center_y
        else:
            y_vals = np.linspace(-230, 230, num=width)
            x_vals = np.full_like(y_vals, rho+center_y)
            #y_vals = np.full_like(x_vals, rho + center_y)  # Constant y-value
        #x_vals+=x_vals

        # Add center_y to shift y-values based on the center of the image


        # Clip both x_vals and y_vals to ensure they lie within the image boundaries
        # x_vals = np.clip(x_vals, 0, width - 1)
        # y_vals = np.clip(y_vals, 0, height - 1)

        # Plot the lines
        ax.plot(x_vals, y_vals, 'r-', linewidth=2)

    ax.set_aspect(1)

    # Set the title
    ax.set_title("Kikuchi Pattern with Superimposed Lines (PyEBSD Data)")

    # Show the plot
    plt.show()
def main():
    # Create the simulated Kikuchi pattern image
    image = create_simulated_kikuchi_pattern()
    np.save('simulated_kikuchi.npy', image)

    # Calculate the expected rho and theta values for the 3 main lines
    expected_rho_theta = calculate_expected_rho_theta()

    # Print expected rho, theta values for verification
    print("Expected rho and theta values (in radians):")
    for i, (rho, theta) in enumerate(expected_rho_theta, 1):
        print(f"Line {i}: rho={rho}, theta={theta:.2f} radians ({np.degrees(theta):.2f} degrees)")

    # PyEBSDIndex package band data for comparison
    realImage=True
    if realImage:
        image = np.load('simulated_kikuchi.npy')
        pyebsd_band_data = [
            {'max': 13349401.0, 'maxloc': [62.0, 42.0], 'width': 3.6621719e-07, 'theta': 2.400463, 'rho': -46.445145},
            {'max': 11410441.0, 'maxloc': [59.0, 161.0], 'width': 5.0434824e-07, 'theta': 0.33275744,
             'rho': -38.568188},
            {'max': 11000612.0, 'maxloc': [45.0, 101.0], 'width': 6.8168879e-07, 'theta': 1.3698503, 'rho': -1.519887},
            {'max': 10819828.0, 'maxloc': [34.0, 131.0], 'width': 6.6961923e-07, 'theta': 0.8536699, 'rho': 25.090218},
            {'max': 10494454.0, 'maxloc': [36.0, 72.0], 'width': 7.3314158e-07, 'theta': 1.8912684, 'rho': 20.916367},
            {'max': 10453215.0, 'maxloc': [71.0, 130.0], 'width': 1.1066954e-06, 'theta': 0.8762146, 'rho': -68.500496},
            {'max': 10368726.0, 'maxloc': [46.0, 12.0], 'width': 8.6238418e-07, 'theta': 2.9371843, 'rho': -4.0015593},
            {'max': 10360235.0, 'maxloc': [73.0, 74.0], 'width': 1.2968941e-06, 'theta': 1.8499055, 'rho': -73.31508},
            {'max': 9056425.0, 'maxloc': [27.0, 161.0], 'width': 1.3160310e-06, 'theta': 0.33033556, 'rho': 44.407455},
            {'max': 8583648.0, 'maxloc': [29.0, 41.0], 'width': 1.3975440e-06, 'theta': 2.4329102, 'rho': 38.083237}
        ]
        pyebsd_band_data = [
            {'max': 1.06e+08, 'maxloc': [72.0, 135.0], 'width': 6.63e-08, 'theta': 0.79, 'rho': -70.98},
            {'max': 9.46e+07, 'maxloc': [44.0, 135.0], 'width': 8.72e-08, 'theta': 0.79, 'rho': 0.00},
            {'max': 6.40e+07, 'maxloc': [21.0, 90.0], 'width': 1.35e-07, 'theta': 1.57, 'rho': 59.32},
            #{'max': 5.26e+07, 'maxloc': [25.0, 1.0], 'width': 2.47e-07, 'theta': 3.12, 'rho': 49.04},
            #{'max': 4.56e+07, 'maxloc': [63.0, 178.0], 'width': 6.41e-07, 'theta': 0.03, 'rho': -49.80},
            {'max': 4.17e+07, 'maxloc': [21.0, 30.0], 'width': 6.37e-07, 'theta': 2.62, 'rho': 58.52},
            {'max': 4.01e+07, 'maxloc': [50.0, 40.0], 'width': 8.47e-07, 'theta': 2.45, 'rho': -14.75},
            {'max': 4.01e+07, 'maxloc': [15.0, 53.0], 'width': 8.54e-07, 'theta': 2.22, 'rho': 74.50},
            {'max': 4.01e+07, 'maxloc': [49.0, 69.0], 'width': 8.70e-07, 'theta': 1.93, 'rho': -13.21},
            {'max': 3.98e+07, 'maxloc': [55.0, 82.0], 'width': 1.06e-06, 'theta': 1.72, 'rho': -28.48}
        ]


    else:
        pyebsd_band_data = [
            {'max': 70098760.0, 'maxloc': [44.0, 135.0], 'width': 1.120e-07, 'theta': 0.784, 'rho': -0.041},
            {'max': 51453320.0, 'maxloc': [19.0, 90.0], 'width': 1.299e-07, 'theta': np.pi/4, 'rho': 65},
             {'max': 41243536.0, 'maxloc': [25.0, 1.0], 'width': 2.227e-07, 'theta': np.pi/2., 'rho': 60},
             {'max': 36843360.0, 'maxloc': [64.0, 178.0], 'width': 3.121e-07, 'theta': 0.0, 'rho': -50.536},
            # {'max': 27450852.0, 'maxloc': [34.0, 111.0], 'width': 9.969e-07, 'theta': 1.198, 'rho': 25.717},
            # {'max': 27436112.0, 'maxloc': [44.0, 175.0], 'width': 6.375e-07, 'theta': 0.080, 'rho': 0.351},
            # {'max': 26658668.0, 'maxloc': [44.0, 91.0], 'width': 9.031e-07, 'theta': 1.551, 'rho': 0.100},
            # {'max': 26631714.0, 'maxloc': [43.0, 74.0], 'width': 8.744e-07, 'theta': 1.844, 'rho': 1.585},
            # {'max': 26602880.0, 'maxloc': [45.0, 27.0], 'width': 1.355e-06, 'theta': 2.674, 'rho': -1.984},
            # {'max': 26565130.0, 'maxloc': [45.0, 16.0], 'width': 9.004e-07, 'theta': 2.872, 'rho': -2.098}
        ]
    # Compare the first three bands from pyEBSD data with the expected rho/theta values
    print("\nComparing PyEBSD band data with expected values:")
    for i, (expected, pyebsd) in enumerate(zip(expected_rho_theta, pyebsd_band_data), 1):
        expected_rho, expected_theta = expected
        print(f"Line {i}:")
        print(f"  Expected  -> rho: {expected_rho}, theta: {np.degrees(expected_theta):.2f} degrees")
        print(f"  PyEBSD    -> rho: {pyebsd['rho']}, theta: {np.degrees(pyebsd['theta']):.2f} degrees")

    # Superimpose the first three lines from the PyEBSD data onto the image
    #plt.imshow(image, cmap='gray', origin='lower')
    find_lines_from_rho_theta(image, pyebsd_band_data)
    plt.title("Simulated Kikuchi Pattern with Superimposed Lines (PyEBSD Data)")
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()
