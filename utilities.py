import os
import pandas as pd
import numpy as np
import logging

def modify_ang_file(file_path, IQ=None):
    """
    Modifies the specified .ang file by updating the IQ (Image Quality) values in the data section.
    The method reads an .ang file, extracts its `NCOLS_EVEN` and `NROWS` values from the header, and updates
    the IQ values in the sixth column of each data line. If IQ values are not provided, random values are generated.
    The modified file is saved with "_IQ_modified" appended to the original filename.

    Parameters:
    -----------
    file_path : str
        Path to the .ang file to be modified.
    IQ : array-like, optional
        Array of IQ values to use in place of the original ones. If not provided, random IQ values between 0 and 100
        are generated with two decimal precision.

    Returns:
    --------
    None
        The function saves the modified file in the same directory with "_IQ_modified" added to its original name.

    Notes:
    ------
    - The function expects the data section to start with lines that are indented with two spaces.
    - The sixth column (zero-indexed as column 5) in the data section is assumed to be the IQ values.
    - If `NCOLS_EVEN` or `NROWS` are not found in the header, the modification is aborted, and an error message is printed.
    """

    # Read the contents of the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract NCOLS_EVEN and NROWS from the file
    ncols_even = None
    nrows = None

    for line in lines:
        if line.startswith("# NCOLS_EVEN:"):
            ncols_even = int(line.split(":")[1].strip())
        elif line.startswith("# NROWS:"):
            nrows = int(line.split(":")[1].strip())
        if ncols_even is not None and nrows is not None:
            break

    if ncols_even is None or nrows is None:
        print("Error: NCOLS_EVEN or NROWS not found in the header.")
        return

    # Generate random IQ values (rounded to 2 decimal places)
    if IQ is None:
        iq_values = np.round(np.random.uniform(0, 100, size=ncols_even * nrows), 2)
    else:
        iq_values = IQ


    nPixels = nrows*ncols_even
    if not nPixels==iq_values.size:
        logging.error(f"The ang file has {nPixels} pixels but the IQ data has {iq_values.size} elements !!!")

    # Prepare to modify the file
    data_lines_start = 0
    for i, line in enumerate(lines):
        if line.startswith("  "):  # Data starts after header
            data_lines_start = i
            break

    # Modify column 6 (IQ values)
    modified_lines = lines[:data_lines_start]
    data_lines = lines[data_lines_start:]

    for i, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) == 13:  # Ensure it's a valid data line
            parts[5] = f"{iq_values[i]:.2f}"  # Replace column 6 (IQ) with new value
            modified_line = "  ".join(parts) + "\n"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)  # Keep non-data lines unchanged

    # Create the new filename with "_IQ_modified"
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_modified{ext}"

    # Save the modified file
    with open(new_file_path, 'w') as f:
        f.writelines(modified_lines)

    print(f"Modified file saved as: {new_file_path}")
    logging.info(f"New ang file is created at : {new_file_path}")
