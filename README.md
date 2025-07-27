# Kikuchi Pattern Processing and Band Width Detection

This script processes EBSD (Electron Backscatter Diffraction) data from Kikuchi patterns, using a configurable `.yml` file for specifying all necessary inputs. The program performs data loading, cropping (for debugging), pattern simulation, and Kikuchi band width detection, saving the results in CSV files.

## Features
- **Configurable Input**: All key parameters, paths, and settings are provided via `bandDetectorOptions.yml`, making the script adaptable for different datasets and materials without modifying the code.
- **Flexible Band Detection**: Detects Kikuchi bands based on specified parameters, outputs a detailed analysis of each detected band, and supports filtering and grouping of results based on PSNR.
- **Logging and Debugging**: Logs key steps, with debug options to speed up runs by cropping data when needed.
- **Modular Structure**: Designed for easy modification and extension.

---

## Installation

### Requirements

The script requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `yaml`
- `openpyxl`
- `kikuchipy`
- `orix`
- `diffpy.structure`

To install these libraries, use:
```bash
pip install pandas numpy matplotlib pyyaml openpyxl kikuchipy orix diffpy.structure

Files
`KikuchiBandWidthAutomator.py` is the main entry point for processing EBSD data and detecting Kikuchi bands.
Configuration File: bandDetectorOptions.yml - Contains all input parameters, including file paths, material information, detection thresholds, and settings for debug and visualization options.
Usage
Step 1: Configure the .yml File
The bandDetectorOptions.yml file contains all configurable parameters. Here’s an example structure of the file:

yaml
Copy code
# Configuration for input and processing
h5_file_path: "C:/path/to/your/Nickel.h5"  # Path to the .h5 EBSD dataset file
crop_start: 5     # Starting index for cropping, used only in debug mode
crop_end: 25      # Ending index for cropping, used only in debug mode
debug: false      # Enable cropping for faster debugging

# Material phase list configuration
phase_list:
  name: "Ni"                 # Phase name, e.g., "Ni" for Nickel
  space_group: 225           # Space group number for the crystal structure
  lattice: [3.5236, 3.5236, 3.5236, 90, 90, 90]  # Lattice parameters
  atoms:                      # List of atoms in the structure
    - element: "Ni"           # Element symbol
      position: [0, 0, 0]     # Position in the unit cell

# List of HKL indices for Kikuchi line detection
hkl_list:
  - [1, 1, 1]
  - [2, 0, 0]
  - [2, 2, 0]
  - [3, 1, 1]

# Detector configuration
pc: [0.545, 0.610, 0.6863]  # Pattern center for EBSD detector

# Kikuchi band width detection parameters
gradient_threshold: 50      # Threshold for edge detection gradients
perpendicular_line_length: 40  # Length of perpendicular lines to detected bands
smoothing_sigma: 2          # Sigma for Gaussian smoothing
strategy: "rectangular_area"  # Detection strategy (e.g., "rectangular_area")
rectWidth: 50               # Width of rectangle for detection
min_psnr: 1.1               # Minimum PSNR for accepting bands

plot_results: false         # Set to true to enable visualization
Step 2: Run the Script
Run the main script as follows:

bash
Copy code
python script_name.py
Step 3: View Output
Full Data Output: The script saves a complete analysis to `bandOutputData.csv` based on all detected bands.
Filtered Data: The filtered results, with PSNR-based grouping and other criteria, are saved to `filtered_band_data.csv`.
Code Walkthrough
The script has three main stages:

Data Loading:

Loads EBSD data from the .h5 file path specified in bandDetectorOptions.yml.
If debug is set to true, it crops the dataset between specified crop_start and crop_end indices to reduce processing time during development.
Kikuchi Pattern Simulation:

Configures a detector using parameters in the .yml file.
Simulates Kikuchi patterns using CustomKikuchiPatternSimulator, which generates markers and band labels based on the provided hkl_list.
Band Width Detection:

The `KikuchiBatchProcessor` class performs band detection using its `process()` method. Detection parameters are specified in the `.yml` file, such as `gradient_threshold`, `perpendicular_line_length`, `smoothing_sigma`, and `rectWidth`.
All floating-point values in the DataFrame are rounded to three decimal places before saving to CSV.
Filtering and Grouping Results:

Filters results for valid bands, groups by `Ind`, and extracts entries with the highest PSNR for each group, saving the final processed data to `filtered_band_data.csv`.
Example Output
`bandOutputData.csv`: Contains all detected band properties, including band width and PSNR values, with floating-point numbers rounded to three decimal places.
`filtered_band_data.csv`: Filtered data with the highest PSNR per band.
Logging and Debugging
The script uses logging to track key steps:

Loading Data: Logs the file being loaded.
Debug Mode: If enabled, logs that cropping is applied for faster execution.
Saving Results: Logs the path where results are saved.
Enable debug: true in the .yml file to crop data, making it easier to test modifications quickly.

Customization
For different materials or datasets, adjust the phase_list and hkl_list in bandDetectorOptions.yml. The file is structured to allow flexible adjustment of all major parameters, making it simple to extend or customize without modifying the main code.

Troubleshooting
Path Errors: Ensure paths use double backslashes (\\) or forward slashes (/) in bandDetectorOptions.yml.
Missing Libraries: Install required libraries using pip install commands listed in the Installation section.