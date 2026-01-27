import os
import json
import pandas as pd
import numpy as np
import logging
import h5py
import shutil
import random
import matplotlib.pyplot as plt
import re
from packaging.version import parse as _v
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase
from orix.vector import Miller
from typing import Union, Tuple


def parse_indices(obj: Union[str, Tuple[int, ...], list]) -> Tuple[int, ...]:
    """
    Parse Miller indices from a string, list, or tuple into a tuple of integers.

    Accepts inputs like:
        - String formats: "1,2,3", "(1,2,3)", "1 -2 0", "1,-2, 0"
        - Tuple/list formats: (1, 2, 3), [1, -2, 0]

    Parameters
    ----------
    obj : str | tuple[int, ...] | list[int]
        The input object representing a Miller or Miller–Bravais index.

    Returns
    -------
    tuple[int, ...]
        A tuple of parsed integers, e.g., (1, -2, 0).

    Raises
    ------
    ValueError
        If the input is not a string, list, or tuple, or cannot be parsed correctly.
    """
    if isinstance(obj, str):
        nums = re.findall(r"-?\d+", obj)
        return tuple(int(n) for n in nums)
    elif isinstance(obj, (tuple, list)):
        return tuple(int(n) for n in obj)
    else:
        raise ValueError(f"Bad input: {obj!r}")


def make_phase(cfg: dict) -> Phase:
    """
    Construct an orix Phase object from a configuration dictionary.

    Expected format:
        cfg = {
            "name": "Ni",
            "space_group": 225,
            "lattice": [a, b, c, alpha, beta, gamma]  # in Å and degrees
        }

    The function assumes a single atom at the origin (0, 0, 0).

    Parameters
    ----------
    cfg : dict
        Dictionary with keys: "name", "space_group", "lattice".

    Returns
    -------
    Phase
        An orix Phase object representing the crystal structure.

    Raises
    ------
    KeyError
        If any required key is missing from the config dictionary.
    """
    L = Lattice(*cfg["lattice"])
    struct = Structure(lattice=L, atoms=[Atom(cfg["name"], [0, 0, 0])])
    return Phase(name=cfg["name"], space_group=cfg["space_group"], structure=struct)


def belongs_to_group(
    hkl: Union[str, Tuple[int, ...], list],
    desired: Union[str, Tuple[int, ...], list],
    phase: Phase,
    tolerance_deg: float = 0.5
) -> Tuple[bool, float]:
    """
    Check if a given Miller or Miller–Bravais index belongs to the same
    symmetric group as a desired index using symmetry-aware angle comparison.

    The method uses the `orix` + `diffsims` symmetry operations to determine
    equivalence based on angular proximity.

    Parameters
    ----------
    hkl : str | tuple[int, ...] | list[int]
        Index to test. Can be 3-index (hkl) or 4-index (hkil).
    desired : str | tuple[int, ...] | list[int]
        Target index representing the desired symmetric group.
    phase : Phase
        `orix.crystal_map.Phase` object defining the crystal symmetry.
    tolerance_deg : float, optional
        Angular tolerance (in degrees) for equivalence, default is 0.5°.

    Returns
    -------
    tuple[bool, float]
        (True, angle) if within tolerance, else (False, angle).
        `angle` is the minimum angle (in degrees) between the input and desired plane
        under symmetry operations.

    Raises
    ------
    ValueError
        If either hkl or desired cannot be parsed into valid indices.
    """
    h = parse_indices(hkl)
    d = parse_indices(desired)
    m1 = Miller(hkl=[h] if len(h) == 3 else None, hkil=[h] if len(h) == 4 else None, phase=phase)
    m2 = Miller(hkl=[d] if len(d) == 3 else None, hkil=[d] if len(d) == 4 else None, phase=phase)
    ang = m1.angle_with(m2, use_symmetry=True, degrees=True)[0]
    return abs(ang) < tolerance_deg, ang



def create_temp_file(original_path):
    """
    Creates a temporary copy of the original HDF5 file for processing.

    Args:
        original_path (str): Path to the original HDF5 file.

    Returns:
        str: Path to the temporary file.
    """
    temp_path = f"{os.path.splitext(original_path)[0]}_temp.h5"
    shutil.copy(original_path, temp_path)
    logging.info(f"Temporary file created at: {temp_path}")
    return temp_path

def sanity_check_reordering(patterns, reordered_patterns, nRows, nCols, debug=False):
    """
    Perform sanity checks to ensure reordering is happening.
    If debug is enabled, plot a random index comparison of reordered and original patterns.

    Args:
        patterns (np.ndarray): Original patterns array.
        reordered_patterns (np.ndarray): Reordered patterns array.
        nRows (int): Number of rows in the EBSD grid.
        nCols (int): Number of columns in the EBSD grid.
        debug (bool): Whether to plot debug information.
    """
    total_pixels = nRows * nCols
    mismatches = []

    for row in range(nRows):
        for col in range(nCols):
            old_index = row * nCols + col
            new_index = col * nRows + row
            if not np.array_equal(reordered_patterns[old_index], patterns[old_index]):
                mismatches.append((row, col, old_index, new_index))

    if mismatches:
        logging.info(f"Sanity check passed: Reordering is happening. Mismatched indices: {len(mismatches)}")
    else:
        logging.warning("Sanity check failed: Reordered patterns are identical to original patterns.")

    if debug:
        # Randomly select an index to plot
        random_row, random_col, old_index, new_index = random.choice(mismatches if mismatches else [(0, 0, 0, 0)])
        plt.figure(figsize=(12, 8))

        # Plot reordered_patterns[new_index]
        plt.subplot(2, 2, 1)
        plt.imshow(reordered_patterns[new_index], cmap='gray')
        plt.title(f"Reordered Patterns[new_index={new_index}]")

        # Plot patterns[old_index]
        plt.subplot(2, 2, 2)
        plt.imshow(patterns[old_index], cmap='gray')
        plt.title(f"Patterns[old_index={old_index}]")

        # Plot reordered_patterns[old_index]
        plt.subplot(2, 2, 3)
        plt.imshow(reordered_patterns[old_index], cmap='gray')
        plt.title(f"Reordered Patterns[old_index={old_index}]")

        # Plot patterns[new_index]
        plt.subplot(2, 2, 4)
        plt.imshow(patterns[new_index], cmap='gray')
        plt.title(f"Patterns[new_index={new_index}]")

        plt.tight_layout()
        plt.show()

def extract_header_data(h5_file_path):
    """
    Extract specific header data from a given HDF5 file.
    Fields:
        - Camera Azimuthal Angle
        - Camera Elevation Angle
        - Sample Tilt
        - Working Distance

    Args:
        h5_file_path (str): Path to the HDF5 file.

    Returns:
        dict: Extracted header data.
    """
    header_data = {}
    with h5py.File(h5_file_path, "a") as h5file:
        # Identify target dataset name, ignoring Manufacturer and Version
        target_dataset_name = next(name for name in h5file if name not in ["Manufacturer", "Version"])
        header_path = f"{target_dataset_name}/EBSD/Header"
        
        if header_path in h5file:
            header_group = h5file[header_path]
            
            # Extract specific fields
            header_data["Camera Azimuthal Angle"] = float(header_group.get("Camera Azimuthal Angle", "N/A")[0])
            header_data["Camera Elevation Angle"] = float(header_group.get("Camera Elevation Angle", "N/A")[0])
            header_data["Sample Tilt"] = float(header_group.get("Sample Tilt", "N/A")[0])
            header_data["Working Distance"] = header_group.get("Working Distance", "N/A")[:]
            header_data["pc"] = (float(header_group.get("Pattern Center Calibration/x-star", "N/A")[0]),
                                 float(header_group.get("Pattern Center Calibration/y-star", "N/A")[0]),
                                 float(header_group.get("Pattern Center Calibration/z-star", "N/A")[0]))
            
        else:
            logging.warning(f"Header path '{header_path}' not found in the HDF5 file.")
    
    return header_data

def reorder_patterns_in_hdf(file_path, target_dataset_name, debug=False):
    """
    Reorders the patterns in the HDF5 file from row-major to column-major order.

    Args:
        file_path (str): Path to the HDF5 file.
        target_dataset_name (str): Dataset name under /{target_dataset_name}/EBSD/Data/Pattern.

    Returns:
        None
    """
    logging.warning("Reordering patterns from row-major to column-major order.")
    with h5py.File(file_path, "r+") as h5file:
        # Read nRows and nCols from the header
        header = h5file[f"/{target_dataset_name}/EBSD/Header"]
        nRows = int(header["nRows"][:])
        nCols = int(header["nColumns"][:])

        # Access the pattern data
        pattern_dataset = h5file[f"/{target_dataset_name}/EBSD/Data/Pattern"]
        patterns = pattern_dataset[:]
        if debug:
            patterns = np.arange(1,9*6+1).reshape(6,3,3,)
            nRows,nCols = 2,3

        # Verify the shape
        nPixels, pattern_height, pattern_width = patterns.shape
        #assert nPixels == nRows * nCols, "Mismatch between nPixels and nRows*nCols."

        # Create a new array for reordered patterns
        reordered_patterns = np.zeros_like(patterns)
        c_indices = np.arange(nPixels).reshape((nRows, nCols, ), order='F').flatten(order='C')
        f_indices = np.arange(nPixels).reshape((nRows, nCols), order='F').flatten(order='F')

        # Reorder patterns
        # Generate old indices (row-major order)
        for c_index, f_index in zip(c_indices, f_indices):
            reordered_patterns[f_index] = patterns[c_index]

            # Debugging
            if debug and c_index < 10:  # Log first 10 mappings for verification
                logging.info(f"Mapping Patterns[c_index={c_index}] to Reordered[f_index={f_index}]")

        # Sanity check for reordering
        if debug:
            logging.info("Performing sanity check on reordered patterns.")
            mismatches = np.sum(reordered_patterns != patterns)
            if mismatches:
                logging.info(f"Sanity check passed. Total mismatches: {mismatches}")
            else:
                logging.warning("Sanity check failed: Reordered patterns are identical to original patterns.")
	
        
        sanity_check_reordering(patterns, reordered_patterns, nRows, nCols, debug=debug)

        # Replace the dataset with reordered patterns
        del h5file[f"/{target_dataset_name}/EBSD/Data/Pattern"]
        h5file.create_dataset(f"/{target_dataset_name}/EBSD/Data/Pattern", data=reordered_patterns)


def modify_ang_file(file_path, file_suffix="_band_width", **kwargs):
    """
    Modifies the specified .ang file by updating specified columns with new values.
    The method reads an .ang file, extracts its `NCOLS_EVEN` and `NROWS` values from the header,
    and updates the specified columns in the data section. If values are not provided, the columns are left unchanged.

    Parameters:
    -----------
    file_path : str
        Path to the .ang file to be modified.
    kwargs : dict
        Key-value pairs where keys are column names (e.g., "IQ", "PRIAS_Bottom")
        and values are arrays to replace the respective column data.

    Returns:
    --------
    None
        The function saves the modified file in the same directory with "_modified" added to its original name.

    Notes:
    ------
    - The function expects the data section to start with lines that are indented with two spaces.
    - If provided column data do not match the number of pixels in the .ang file, an error is raised.
    """
    # Read the contents of the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract NCOLS_EVEN and NROWS from the file
    ncols_even = None
    nrows = None
    column_headers = []

    for line in lines:
        if line.startswith("# NCOLS_EVEN:"):
            ncols_even = int(line.split(":")[1].strip())
        elif line.startswith("# NROWS:"):
            nrows = int(line.split(":")[1].strip())
        elif line.startswith("# COLUMN_HEADERS:"):
            # Split column headers and normalize (replace spaces with underscores for internal processing)
            column_headers = [header.strip().replace(" ", "_") for header in line.split(":")[1].strip().split(", ")]
        if ncols_even is not None and nrows is not None and column_headers:
            break

    if ncols_even is None or nrows is None or not column_headers:
        logging.error("Error: NCOLS_EVEN, NROWS, or COLUMN_HEADERS not found in the header.")
        return

    # Ensure column names provided in kwargs are valid
    invalid_columns = [col for col in kwargs if col not in column_headers]
    if invalid_columns:
        logging.error(f"Invalid column names provided: {invalid_columns}")
        return

    # Verify sizes of provided column data
    n_pixels = nrows * ncols_even
    for col, data in kwargs.items():
        if len(data) != n_pixels:
            logging.error(f"The ang file has {n_pixels} pixels, but the {col} data has {len(data)} elements!")
            return

    # Determine the starting index of data lines
    data_lines_start = 0
    for i, line in enumerate(lines):
        if "# HEADER: End" in line:  # Data starts after the header
            data_lines_start = i+1
            break

    # Modify specified columns in the data section
    modified_lines = lines[:data_lines_start]
    data_lines = lines[data_lines_start:]
    col_indices = {col: column_headers.index(col) for col in kwargs}  # Map column names to indices

    for i, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) == len(column_headers):  # Ensure it's a valid data line
            for col, data in kwargs.items():
                col_idx = col_indices[col]
                parts[col_idx] = f"{data[i]:.2f}"  # Replace column with new value
            modified_line = "  ".join(parts) + "\n"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)  # Keep non-data lines unchanged

    # Create the new filename with "_modified"
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_modified_{file_suffix}{ext}"

    # Save the modified file
    with open(new_file_path, 'w') as f:
        f.writelines(modified_lines)

    logging.info(f"Modified file saved as: {new_file_path}")


def convert_results(obj):
    """
    Recursively convert numpy scalars/arrays into JSON-serializable types.

    Parameters:
        obj: Object to convert (dict, list, numpy scalar/array, or primitive).

    Returns:
        JSON-serializable object.
    """
    if isinstance(obj, dict):
        return {k: convert_results(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_results(i) for i in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results_to_json(results, path="bandOutputData.json"):
    """
    Serialize detection results to a JSON file.

    Parameters:
        results: Results list/dict to serialize.
        path: Destination JSON path.

    Returns:
        None.
    """
    with open(path, "w") as f:
        json.dump(convert_results(results), f, indent=4)
    logging.info("Results saved to %s", path)


def save_results_to_csv(results, raw_path="bandOutputData.csv", filtered_path="filtered_band_data.csv"):
    rows = []
    for res in results:
        xy = res.get("x,y")
        ind = res.get("ind")
        for band in res["bands"]:
            bw = band.get("bandWidth")
            cpp = band.get("centralPeak")
            valid = band.get("band_valid", False)
            peak = band.get("band_peak", 0)
            bkg = band.get("band_bkg", 0)
            psnr = band.get("psnr", 0)
            eff = band.get("efficientlineIntensity", 0)
            deff = band.get("defficientlineIntensity", 0)
            ratio = (
                eff / deff if deff and eff and np.isfinite(eff) and np.isfinite(deff) else 0
            )
            rows.append(
                {
                    "X,Y": xy,
                    "Ind": ind,
                    "hkl": band.get("hkl"),
                    "hkl_group": band.get("hkl_group", "unknown"),
                    "Central Line": band.get("central_line"),
                    "Line Distance": band.get("line_dist"),
                    "Band Width": bw,
                    "band_peak": peak,
                    "band_bkg": bkg,
                    "psnr": psnr,
                    "efficientlineIntensity": eff,
                    "defficientlineIntensity": deff,
                    "efficientDefficientRatio": np.round(ratio, 2),
                    "band_valid": valid,
                }
            )

    df = pd.DataFrame(rows).round(3)
    df.to_csv(raw_path, index=False)
    logging.info("Raw results saved to %s", raw_path)

    filt = df[df["band_valid"]]
    best = filt.loc[filt.groupby("Ind")["psnr"].idxmax()]
    best.to_csv(filtered_path, index=False)
    logging.info("Filtered results saved to %s", filtered_path)


def create_mock_ang_file(file_path, nrows, ncols_even, headers, column_headers):
    """
    Creates a mock .ang file for testing purposes.

    Parameters:
    -----------
    file_path : str
        Path to save the mock .ang file.
    nrows : int
        Number of rows in the data section.
    ncols_even : int
        Number of columns in the data section.
    headers : list of str
        Header lines to include in the file.
    column_headers : list of str
        Column headers to use in the data section.

    Returns:
    --------
    None
    """
    with open(file_path, 'w') as f:
        # Write headers
        for header in headers:
            f.write(header + "\n")
        f.write("# COLUMN_HEADERS: " + ", ".join(column_headers) + "\n")
        f.write("# NCOLS_EVEN: " + str(ncols_even) + "\n")
        f.write("# NROWS: " + str(nrows) + "\n")
        f.write("\n")  # Separate header and data

        # Generate mock data
        for _ in range(nrows):
            data_line = "  ".join([f"{np.random.uniform(0, 100):.2f}" for _ in range(ncols_even)])
            f.write(data_line + "\n")


def main():
    # Mock parameters
    test_file = r"C:\Users\kvman\Downloads\IS_Ni_ebsd_data\Nickel_modified.ang"
    nrows = 231
    ncols_even = 184
    headers = [
        "# HEADER: Start",
        "# Mock data for testing",
        "# HEADER: End"
    ]
    column_headers = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index", "SEM", "Fit", "PRIAS_Bottom",
                      "PRIAS_Middle", "PRIAS_Top"]

    # Create a mock .ang file
    #create_mock_ang_file(test_file, nrows, ncols_even, headers, column_headers)
    logging.info(f"Mock .ang file created: {test_file}")

    # Generate random replacement data
    iq_values = np.random.uniform(0, 100, nrows * ncols_even)
    prias_bottom_values = np.random.uniform(0, 2000, nrows * ncols_even)

    # Test modify_ang_file function
    modify_ang_file(test_file, IQ=iq_values, PRIAS_Bottom_Strip=prias_bottom_values)

    # Check for the modified file
    modified_file = os.path.splitext(test_file)[0] + "_modified.ang"
    if os.path.exists(modified_file):
        logging.info(f"Modified .ang file successfully created: {modified_file}")
    else:
        logging.error("Modified .ang file was not created.")


if __name__ == "__main__":
    main()
