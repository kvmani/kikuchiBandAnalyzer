# Kikuchi Band Analyzer

This project detects Kikuchi band widths from EBSD data. The pipeline is driven
by YAML configuration files and can run in a fast debug mode or a full
processing mode.

## Features
- **Config driven**: All parameters such as data paths, phase information and
  detection thresholds are defined in YAML files (e.g. `bandDetectorOptions.yml`).
- **Automated workflow**: `KikuchiBandWidthAutomator.py` loads the EBSD data,
  runs band detection and writes results to CSV/HDF5.
- **Band width detection**: `kikuchiBandWidthDetector.py` implements the core
  logic to detect bands in each pattern.
- **Logging**: Scripts use Python's logging module for progress reporting.

## Usage
### Normal Mode
1. Edit `bandDetectorOptions.yml` to point to your EBSD `.h5` file and adjust
   detection parameters.
2. Run the automator:
   ```bash
   python KikuchiBandWidthAutomator.py
   ```
   Results are written next to the input file as `<name>_bandOutputData.csv` and
   `<name>_filtered_band_data.csv`.

### Debug Mode
Set `debug: true` in the YAML file to process only a cropped region of the data.
Debug mode enables detailed `DEBUG` logging and is useful for quick tests.

## Script Overview
### KikuchiBandWidthAutomator.py
1. **prepare_dataset** – loads the HDF5 data and optionally crops it when debug
   mode is enabled.
2. **simulate_and_index** – simulates Kikuchi patterns and determines expected
   band positions using `kikuchipy` and `orix`.
3. **detect_band_widths** – delegates to `KikuchiBatchProcessor` to compute band
   widths for every pattern.
4. **export_results** – saves CSV summaries and writes computed arrays back to
   the HDF5/ANG files.

### kikuchiBandWidthDetector.py
- **BandDetector** – examines a single pattern and measures band widths based on
  marker points.
- **KikuchiBatchProcessor** – iterates over a grid of patterns and aggregates
  results.
- Helper functions load image stacks and JSON marker definitions.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Configuration File Fields
- `h5_file_path`: path to the EBSD dataset.
- `phase_list`: crystal phase description used for simulation.
- `hkl_list`: HKL lines to label and detect.
- `desired_hkl`: HKL group for band width statistics.
- `smoothing_sigma`, `rectWidth`, `min_psnr`: detection parameters.
- `debug`, `crop_start`, `crop_end`: enable and configure debug mode.

## Output
The automator creates two CSV files containing raw and filtered band
measurements. When run in normal mode it also writes band width, strain and
stress arrays back into the original HDF5 file.
