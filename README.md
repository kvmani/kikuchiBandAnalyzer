# Kikuchi Pattern Processing and Band Width Detection

This repository contains utilities for processing EBSD data and measuring Kikuchi band widths using configurable YAML options.

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Packages such as `pyebsdindex` and `openpyxl` are required and listed in the file.

## Usage

1. Adjust parameters in `bandDetectorOptions.yml` (or the debug variant).
2. Execute the automator script:

```bash
python KikuchiBandWidthAutomator.py
```

The program loads the EBSD dataset, simulates Kikuchi patterns and writes results to Excel. Updated `.ang` files are created in the same folder as the input data, e.g. `DA_modified_002_band_width.ang`.

## Code Structure

- `KikuchiBandWidthAutomator.py` – main driver of the workflow.
- `custom_simulation.py` – houses the `CustomGeometricalKikuchiPatternSimulation` and `CustomKikuchiPatternSimulator` classes. The module logs the HyperSpy version when creating text markers.
- `utilities.py` – common helpers for file handling and `.ang` modifications.

## Troubleshooting

Ensure all paths are correct in the YAML file. The script prints progress messages, including which HyperSpy version is active and where output files are saved.
