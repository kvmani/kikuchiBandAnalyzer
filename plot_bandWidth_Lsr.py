import matplotlib.pyplot as plt
import numpy as np


def read_ang_file(filepath):
    """
    Reads a .ang file and extracts the IQ column.

    Parameters:
        filepath (str): Path to the .ang file.

    Returns:
        list: IQ column values as a list of floats.
    """
    iq_values = []
    with open(filepath, 'r') as file:
        for line in file:
            # Skip header lines
            if line.strip().startswith('#') or line.strip() == '':
                continue
            # Split the line into columns and extract the IQ column (index 5)
            columns = line.split()
            if len(columns) >= 6:  # Ensure there are enough columns
                try:
                    iq_values.append(float(columns[5]))
                except ValueError:
                    continue  # Skip lines that cannot be parsed
    return iq_values


def filter_data(iq1, iq2):
    """
    Filters out points from iq1 where values are either == 0 or > 35.
    Simultaneously removes corresponding points from iq2.

    Parameters:
        iq1 (list): List of IQ values from file 1.
        iq2 (list): List of IQ values from file 2.

    Returns:
        tuple: Filtered iq1 and iq2 lists.
    """
    filtered_iq1 = []
    filtered_iq2 = []

    for val1, val2 in zip(iq1, iq2):
        if np.around(val1,0) != 0 and val1 <= 35:
            filtered_iq1.append(val1)
            filtered_iq2.append(val2)

    return filtered_iq1, filtered_iq2

def plot_iq_scatter(file1, file2):
    """
    Plots a scatter plot of IQ values from two .ang files.

    Parameters:
        file1 (str): Path to the first .ang file.
        file2 (str): Path to the second .ang file.
    """
    iq1 = read_ang_file(file1)
    iq2 = read_ang_file(file2)

    iq1, iq2 = filter_data(iq1, iq2)

    # Ensure the two IQ lists have the same length
    min_length = min(len(iq1), len(iq2))
    iq1 = iq1[:min_length]
    iq2 = iq2[:min_length]



    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(iq1, iq2, alpha=0.7)
    plt.xlabel("BandWidth from File 1")
    plt.ylabel("LRS from File 2")
    plt.title("Scatter Plot of BandWidth Vs Lrs")
    plt.grid(True)
    plt.show()
  

# Paths to the .ang files
file1 = r"E:\Debarna_LRS\Automation\coarsened data sets\rawData\cropped_modified_band_width.ang" ### band width
file2 = r"E:\Debarna_LRS\Automation\coarsened data sets\bandWidth_superResolutuon\cropped_out_lrs.ang"


# Generate the scatter plot
plot_iq_scatter(file1, file2)
