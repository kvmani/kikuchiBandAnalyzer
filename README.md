# Kikuchi Band Analyzer

Kikuchi Band Analyzer is a research Python toolkit for measuring Kikuchi band widths from EBSD diffraction patterns. The main entry point is a YAML‑driven batch pipeline (`KikuchiBandWidthAutomator.py`) designed to run non‑interactively in normal mode and on a smaller/cropped dataset in debug mode.

This repo also contains utilities for exporting EBSD patterns to images (useful for machine‑learning workflows) and reconstructing processed images back into HDF5.

## Quickstart (run on included test data)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Disable interactive plots if you are on a headless machine:
   - Set `skip_display_EBSDmap: true` in `bandDetectorOptionsHcp.yml`.

3. Run the pipeline:

   ```bash
   python KikuchiBandWidthAutomator.py
   ```

   In PyCharm, you can instead open `KikuchiBandWidthAutomator.py` and click the green **Run** triangle (or right‑click the file → **Run**).

By default the script uses `bandDetectorOptionsHcp.yml`, which points to `testData/Test_Ti.oh5` and `testData/Test_Ti.ang`.

## Run on your own data

### 1) Prepare input files

Place a matching pair in the same folder (base name must match):

- `sample.oh5` (or `sample.h5`)
- `sample.ang`

The code expects a common EDAX/TSL layout with patterns under `/<scan_name>/EBSD/Data/Pattern` (the scripts pick the first top‑level group that is not `Manufacturer` or `Version` and treat that as `<scan_name>`).

### 2) Choose and edit a YAML config

Pick an options file close to your material and update it (examples include `bandDetectorOptionsHcp.yml`, `bandDetectorOptionsMagnetite.yml`, and `bandDetectorOptionsDebug.yml`).

At minimum, set:

- `h5_file_path`: path to your `.oh5`/`.h5` file
- `phase_list`: crystal structure used for simulation/indexing
- `hkl_list`: reflectors to consider
- `desired_hkl`: target plane family for reporting band widths
- (optional) `debug`, `crop_start`, `crop_end`: faster iteration on a subset

### 3) Run with your chosen config

`KikuchiBandWidthAutomator.py` currently hard‑codes the default config in `main()`. To run a different YAML:

- Option A (simple): edit `KikuchiBandWidthAutomator.py` to pass your YAML:
  - `bwa = BandWidthAutomator(config_path="bandDetectorOptionsMagnetite.yml")`
- Option B (no edits): run from the command line using Python:

  ```bash
  python -c "from KikuchiBandWidthAutomator import BandWidthAutomator; BandWidthAutomator('bandDetectorOptionsMagnetite.yml').run()"
  ```

## Debug vs normal mode

- Normal mode: `debug: false` (processes the full dataset)
- Debug mode: `debug: true` (crops the dataset using `crop_start`/`crop_end` for faster turnaround)

Some options trigger plots (e.g. EBSD map display). For unattended runs, set `skip_display_EBSDmap: true` and disable plotting flags in your YAML.

## Outputs

For an input file `<stem>.oh5`/`<stem>.h5`, the pipeline writes outputs next to the input:

- CSV summaries:
  - `<stem>_bandOutputData.csv`
  - `<stem>_filtered_band_data.csv`
- An augmented HDF5 copy:
  - `<stem>_modified.h5`
  - This copy receives computed datasets under `/<scan_name>/EBSD/Data/` (e.g. `Band_Width`, `psnr`, `strain`, `stress`, …).
- Derived `.ang` files with additional columns:
  - `<stem>_modified_<suffix>.ang`

Notes:
- The pipeline does not overwrite your original `.oh5`/`.h5`; it works on copies.
- If your input is `.oh5`, the code may create an intermediate `.h5` copy with the same stem for processing.

## Optional: CycleGAN / ML preprocessing workflow

If you run a CycleGAN (or other model) to enhance patterns before band‑width analysis, see `HowToRunAnalysis.md` for a PyCharm‑first (Windows) step‑by‑step workflow (with terminal equivalents):

- Export patterns to PNG (`hdf5_image_export_and_validation.py`)
- Run CycleGAN inference (external repo)
- Reconstruct processed PNGs back into HDF5 (`hdf5_image_export_and_validation.py`)
- Run the band‑width pipeline on the reconstructed file

## EBSD Compare GUI (v1)

This repo includes an EBSD scan comparator GUI for aligned OH5 files. See the package README for full details: [`kikuchiBandAnalyzer/ebsd_compare/README.md`](kikuchiBandAnalyzer/ebsd_compare/README.md).

Common commands:

```bash
python scripts/make_noisy_oh5.py --config configs/ebsd_compare_config.yml
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml
python scripts/run_ebsd_compare_demo.py --config configs/ebsd_compare_config.yml
pytest -q
```

Note: generated demo artifacts (noisy OH5 and GUI screenshots) are created locally and are not committed.

## Repository layout (high level)

- `KikuchiBandWidthAutomator.py`: end‑to‑end batch pipeline (YAML‑driven)
- `kikuchiBandWidthDetector.py`: per‑pattern detection + batch processing
- `hdf5_image_export_and_validation.py`: export/reconstruct patterns for ML workflows
- `bandDetectorOptions*.yml`: example configuration files
- `testData/`: small example datasets and fixtures

## Contributing

See `contribute.md` and `AGENTS.md` for contribution guidelines (docstrings, logging, debug/normal run modes, and non‑interactive scripts).
