# Kikuchi Band Analyzer

Kikuchi Band Analyzer is a research Python toolkit for measuring Kikuchi band widths from EBSD diffraction patterns. The main entry point is a YAML‑driven batch pipeline (`KikuchiBandWidthAutomator.py`) designed to run non‑interactively in normal mode and on a smaller/cropped dataset in debug mode.

This repo also contains utilities for exporting EBSD patterns to images (useful for machine‑learning workflows) and reconstructing processed images back into HDF5.

New in this repo version:
- Band-profile exports now include bandwidth search indices (`band_start_idx`, `band_end_idx`, `central_peak_idx`, `profile_length`) in both JSON and OH5/HDF5 outputs.
- A visualization-first **Automator GUI** is available for running the pipeline from YAML without freezing the UI.
- EBSD Comparator can overlay and compare exported `band_profile` vectors from Scan A/B.

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

## Release 1.0.0

### What is new in this release

- Derived field registry with normalized band intensity difference (`band_intensity_diff_norm`) and HDF5 outputs for `band_intensity_ratio`.
- EBSD Comparator now supports YAML-driven scalar field lists via `fields`, with graceful missing-field warnings.
- Linked pan/zoom toggled by `sync_navigation`, plus reliable contrast updates across map panels.
- Shared versioning metadata with `VERSION`, `CHANGELOG.md`, and packaging hooks for the Windows installer.

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
  - This copy receives computed datasets under `/<scan_name>/EBSD/Data/` (e.g. `Band_Width`, `psnr`, `band_intensity_ratio`, `band_intensity_diff_norm`, `band_profile`, `central_line`, `strain`, `stress`, …).
- Derived `.ang` files with additional columns:
  - `<stem>_modified_<suffix>.ang`

Notes:
- The pipeline does not overwrite your original `.oh5`/`.h5`; it works on copies.
- If your input is `.oh5`, the code may create an intermediate `.h5` copy with the same stem for processing.

Derived field definitions:
- `band_intensity_ratio = I_eff / I_def`
- `band_intensity_diff_norm = 2*(I_eff - I_def)/(I_eff + I_def)`; values are set to NaN when `I_eff + I_def` is near zero.

JSON annotation details:
- See [`docs/ebsd_json_schema.md`](docs/ebsd_json_schema.md) for the input/output JSON schemas, `pattern_path` semantics, and mapping to CSV/HDF5 outputs.
- See [`docs/data_formats.md`](docs/data_formats.md) for the authoritative JSON + OH5/HDF5 dataset schema (paths, shapes, dtypes).

### Band-profile datasets (new)

When available, the pipeline writes the following additional datasets under `/<scan_name>/EBSD/Data/`:

- `band_profile`: `(nPixels, profile_len)` float32
- `central_line`: `(nPixels, 4)` float32
- `band_start_idx`, `band_end_idx`, `central_peak_idx`: `(nPixels,)` int32 (`-1` when unavailable)
- `profile_length`: `(nPixels,)` int32
- `band_valid`: `(nPixels,)` int8 (1 when a valid best-band profile is stored)

## Optional: CycleGAN / ML preprocessing workflow

If you run a CycleGAN (or other model) to enhance patterns before band‑width analysis, see `HowToRunAnalysis.md` for a PyCharm‑first (Windows) step‑by‑step workflow (with terminal equivalents):

- Export patterns to PNG (`hdf5_image_export_and_validation.py`)
- Run CycleGAN inference (external repo)
- Reconstruct processed PNGs back into HDF5 (`hdf5_image_export_and_validation.py`)
- Run the band‑width pipeline on the reconstructed file

## Partitioned EBSD pattern export

Use `export_ebsd_partition_patterns.py` to export EBSD patterns into multiple folders based on logical filters over scalar EBSD fields (CI, IQ, Phase, etc.). The script is safe by default (dry‑run) and produces scalar‑field statistics (min/max/mean/std/mode) plus partition summaries before writing any files.

Example config: `configs/ebsd_partition_export.yml`

Run a dry‑run (default):

```bash
python export_ebsd_partition_patterns.py --config configs/ebsd_partition_export.yml
```

Execute export:

```bash
python export_ebsd_partition_patterns.py --config configs/ebsd_partition_export.yml --execute
```

Notes:
- Conditions use canonical field names (e.g., `CI`, `IQ`, `Phase`) and rely on `field_aliases` in the YAML to map to dataset names inside the OH5/HQ5 file.
- Output images are 16‑bit grayscale PNGs (default) scaled per pattern.
- Expression rules: comparisons `> < >= <= == !=`, boolean `AND OR NOT`, parentheses for grouping, and identifier‑only field names (letters/digits/underscore). Chained comparisons are not supported.

Example conditions:
- `CI > 0.1`
- `IQ < 400`
- `CI > 0.1 AND Phase == 1`
- `(CI > 0.15 AND IQ > 300) OR Phase == 2`

## EBSD Compare GUI (v2)

This repo includes an EBSD scan comparator GUI that supports aligned or mismatched OH5 grids. When grids differ, a registration dialog helps align scan B to scan A via human-picked control points and RANSAC. Use the `fields` list in the YAML config to select which scalar maps to compare, and `sync_navigation` to toggle linked pan/zoom. See the package README for full details: [`kikuchiBandAnalyzer/ebsd_compare/README.md`](kikuchiBandAnalyzer/ebsd_compare/README.md).

The GUI also provides:
- A **band profile comparison** panel that plots `band_profile` from Scan A/B on shared axes and overlays `central_line` on patterns when those datasets exist.
- An **Export Comparison OH5** button which writes `{stemA}_{stemB}_comparison.oh5` next to scan A. The export copies scan A as a template, overwrites scalar maps with the chosen delta/ratio result (A/B for ratio), skips Phase-like fields, and embeds alignment metadata for traceability.

User guide:
- [`docs/ebsd_comparator_band_profiles.md`](docs/ebsd_comparator_band_profiles.md)

Common commands:

```bash
python scripts/make_noisy_oh5.py --config configs/ebsd_compare_config.yml
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml --debug
python scripts/run_ebsd_compare_demo.py --config configs/ebsd_compare_config.yml
python -m kikuchiBandAnalyzer.automator_gui.main_window --config bandDetectorOptionsHcp.yml
python scripts/run_automator_gui_demo.py --debug
pytest -q
```

Note: proof screenshots used by the documentation live in `docs/screenshots/`.

## Automator GUI

The Automator GUI runs the same analysis engine as `KikuchiBandWidthAutomator.py`, but provides a visualization-first workflow (map/pattern/profile) and progress monitoring.

User guide:
- [`docs/automator_gui.md`](docs/automator_gui.md)

## Windows installer (EBSD Scan Comparator)

To build a professional Windows installer (single setup EXE that bundles the GUI and dependencies), follow the step-by-step guide:

- `docs/windows_installer_guide.md`

## Versioning and releases

- Current version is stored in `VERSION` and mirrored in `app_metadata.py` at build time.
- Python API exposes `kikuchiBandAnalyzer.__version__` when the `VERSION` file is present.
- Release notes live in `CHANGELOG.md` (append a new section when you bump the version).
- Windows packaging pulls the version via `packaging/generate_installer_vars.py` and `packaging/ebsd_gui.spec`.

## Repository layout (high level)

- `KikuchiBandWidthAutomator.py`: end‑to‑end batch pipeline (YAML‑driven)
- `kikuchiBandWidthDetector.py`: per‑pattern detection + batch processing
- `hdf5_image_export_and_validation.py`: export/reconstruct patterns for ML workflows
- `bandDetectorOptions*.yml`: example configuration files
- `VERSION`: single source of truth for the repo version
- `CHANGELOG.md`: release notes
- `testData/`: small example datasets and fixtures

## Contributing

See `contribute.md` and `AGENTS.md` for contribution guidelines (docstrings, logging, debug/normal run modes, and non‑interactive scripts).
