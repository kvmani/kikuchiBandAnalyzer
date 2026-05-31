# Kikuchi Band Analyzer

Kikuchi Band Analyzer is a research Python toolkit for measuring Kikuchi band widths from EBSD diffraction patterns. The main entry point is a YAML‑driven batch pipeline (`KikuchiBandWidthAutomator.py`) designed to run non‑interactively in normal mode and on a smaller/cropped dataset in debug mode.

This repo also contains utilities for exporting EBSD patterns to images (useful for machine‑learning workflows) and reconstructing processed images back into HDF5.

Supported EBSD scan sources are being unified behind package readers. EDAX/TSL
`.oh5`/`.h5` files are supported today, and HKL/Oxford `.ctf` metadata plus an
external pattern-image folder is now supported by the comparator reader layer
and by the band-width automator through direct CTF Euler-angle line simulation.
See [`docs/hkl_ctf_support.md`](docs/hkl_ctf_support.md).

## Documentation

The authoritative documentation is a Sphinx site under `docs/site`. It covers
installation, debug and normal workflows, TSL `.oh5`/`.h5` processing, HKL
`.ctf` plus pattern-folder processing, exports, tutorials, mathematical
formulations, developer standards, and generated API reference pages.

Build it locally with:

```bash
python -m sphinx -b html docs/site docs/site/_build/html
```

Then open `docs/site/_build/html/index.html`.

New in this repo version:
- Band-profile exports now include bandwidth search indices (`band_start_idx`, `band_end_idx`, `central_peak_idx`, `profile_length`) in both JSON and OH5/HDF5 outputs.
- A visualization-first **Automator GUI** is available for running the pipeline from YAML without freezing the UI.
- EBSD Comparator can overlay and compare exported `band_profile` vectors from Scan A/B.
- A dedicated **OH5 to ANG Exporter GUI** supports mapping OH5 scalar fields into ANG columns with sanity checks and live logging.
- A **Single Pattern Solver** can debug one EBSP from either OH5/H5 or CTF+pattern-folder input with YAML-configured phase, PC convention, detector geometry, simulated Kikuchi overlays, and a chosen `{111}` band profile.

## Single Pattern Solver

Use this before changing batch or GUI automation when debugging pattern center,
Euler convention, or HKL/Oxford versus EDAX/TSL detector geometry.

Render one configured pattern non-interactively:

```bash
python -m kikuchiBandAnalyzer.single_pattern_solver.solver --config configs/single_pattern_ctf.yml --json outputs/single_pattern_ctf/solution.json --png outputs/single_pattern_ctf/solution.png
```

Launch the live PC-adjustment GUI:

```bash
python -m kikuchiBandAnalyzer.single_pattern_solver.gui --config configs/single_pattern_ctf.yml
```

The same GUI is also exposed as `indexing-debug-gui`. It supports CTF plus
pattern-folder input, OH5/H5 scan input, and a single EBSP image. For scan
inputs, use **Load Source / Middle Pixel** to read the scan dimensions and
default the selected pixel to the middle of the map before tuning PC and
detector geometry. Enable **Run kikuchipy Hough indexing** and **Overlay indexed
orientation** when you want the overlay to come from a fresh single-pattern
indexing result rather than the Euler angles stored in the input file.

For a student-facing PC calibration workflow, see
[`docs/single_pattern_pc_calibration.md`](docs/single_pattern_pc_calibration.md).

The bundled examples are:

- `configs/single_pattern_ctf.yml` for `testData/hkl_ctf_test_data/Subset.ctf` plus `Binned_2x2`.
- `configs/single_pattern_da.yml` for `testData/DA.oh5`.

## Quickstart (run on included test data)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Disable interactive plots if you are on a headless machine:
   - Set `skip_display_EBSDmap: true` in `bandDetectorOptionsHcp.yml`.

3. Run the pipeline:

   ```bash
   python KikuchiBandWidthAutomator.py --config bandDetectorOptionsHcp.yml
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

For HKL/Oxford CTF input, use:

- `sample.ctf`
- a pattern folder containing one image per pixel

The code expects a common EDAX/TSL layout with patterns under `/<scan_name>/EBSD/Data/Pattern` (the scripts pick the first top‑level group that is not `Manufacturer` or `Version` and treat that as `<scan_name>`).

For CTF comparator workflows, configure the pattern folder under the `ctf`
section in `configs/ebsd_compare_config.yml`.

### 2) Choose and edit a YAML config

Pick an options file close to your material and update it (examples include `bandDetectorOptionsHcp.yml`, `bandDetectorOptionsMagnetite.yml`, and `bandDetectorOptionsDebug.yml`).

At minimum, set:

- `h5_file_path`: path to your `.oh5`/`.h5` file
- `phase_list`: crystal structure used for simulation/indexing
- `hkl_list`: reflectors to consider
- `desired_hkl`: target plane family for reporting band widths
- (optional) `debug`, `crop_start`, `crop_end`: faster iteration on a subset

### 3) Run with your chosen config

Run any YAML configuration non-interactively from the command line:

```bash
python KikuchiBandWidthAutomator.py --config bandDetectorOptionsMagnetite.yml
```

If installed as a package, the equivalent console command is:

```bash
kikuchi-band-width --config bandDetectorOptionsMagnetite.yml
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
- Companion ANG export for TSL loading:
  - `<stem>_modified.ang`
  - Header is copied verbatim from the original `<stem>.ang`.
  - PRIAS columns are overwritten from `<stem>_modified.h5` as:
    - `PRIAS Bottom Strip` <- `Band_Width`
    - `PRIAS Center Square` <- `psnr`
    - `PRIAS Top Strip` <- `band_intensity_ratio`

Notes:
- The pipeline does not overwrite your original `.oh5`/`.h5`; it works on copies.
- If your input is `.oh5`, the code may create an intermediate `.h5` copy with the same stem for processing.

Derived field definitions:
- `band_intensity_ratio = I_eff / I_def`
- `band_intensity_diff_norm = 2*(I_eff - I_def)/(I_eff + I_def)`; values are set to NaN when `I_eff + I_def` is near zero.

JSON annotation details:
- See [`docs/ebsd_json_schema.md`](docs/ebsd_json_schema.md) for the input/output JSON schemas, `pattern_path` semantics, and mapping to CSV/HDF5 outputs.
- See [`docs/data_formats.md`](docs/data_formats.md) for the authoritative JSON + OH5/HDF5 dataset schema (paths, shapes, dtypes).
- See [`docs/testing_strategy.md`](docs/testing_strategy.md) for fixture, golden-data, stress-case, and quality-gate expectations.
- See [`docs/hkl_ctf_support.md`](docs/hkl_ctf_support.md) for HKL/Oxford CTF plus pattern-folder support.
- See [`docs/hkl_ctf_test_data_request.md`](docs/hkl_ctf_test_data_request.md) for the small FCC HKL/Oxford fixture request used by tests and tutorials.
- For the bundled HKL fixture, run `python scripts/convert_hkl_ctf_test_data.py`
  to create DA-compatible `Subset_HKL_TSL.ang`, `.h5`, and `.oh5` files, then
  run `python scripts/render_hkl_ctf_tsl_validation.py` to generate the IPF-X
  comparison slide against `AcquisitionDetails.pptx`.
- For an end-to-end GUI that prepares either HKL `.ctf` + pattern folders or
  TSL/EDAX `.oh5`/`.h5` + `.ang` inputs and then runs indexing/band-width
  analysis, run `python -m kikuchiBandAnalyzer.workflow_gui.main_window`.
- Tutorial notebooks:
  - [`docs/notebooks/single_pattern_debug_workflow.ipynb`](docs/notebooks/single_pattern_debug_workflow.ipynb)
  - [`docs/notebooks/end_to_end_scan_workflow.ipynb`](docs/notebooks/end_to_end_scan_workflow.ipynb)

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
- Use `max_patterns_per_partition` to cap exports per partition (defaults to 1000) and `random_seed` for reproducible sampling when a partition has more matches than the cap.
- Expression rules: comparisons `> < >= <= == !=`, boolean `AND OR NOT`, parentheses for grouping, and identifier‑only field names (letters/digits/underscore). Chained comparisons are not supported.

Example conditions:
- `CI > 0.1`
- `IQ < 400`
- `CI > 0.1 AND Phase == 1`
- `(CI > 0.15 AND IQ > 300) OR Phase == 2`

## Unmixing research notes (working)

This repo includes living notes for planning and documenting EBSD pattern unmixing approaches (intended to later inform code upgrades and publication-quality discussion):

- [`docs/unmixing_algorithms_and_issues.md`](docs/unmixing_algorithms_and_issues.md) — index of algorithm families, common degeneracies, evaluation ideas, and open questions.
- [`docs/mixed_pattern_unmixing.md`](docs/mixed_pattern_unmixing.md) — case study: avoiding the trivial `A = B = C` collapse in `C ≈ xA + (1-x)B`.

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
python -m kikuchiBandAnalyzer.band_width.detector_cli --source testData/Med_Mn_10k_4x4_00995.png --annotations testData/Med_Mn_10k_4x4_00995.json --config bandDetectorOptionsMagnetiteAccuracyTesting.yml --tile-from-single --debug --json-output outputs/single_pattern_debug/bandOutputData.json
python KikuchiBandWidthAutomator.py --config bandDetectorOptionsHcp.yml
python -m kikuchiBandAnalyzer.band_width.cli --config bandDetectorOptionsHcp.yml
python -m kikuchiBandAnalyzer.band_width.detector_cli --source testData/Med_Mn_10k_4x4_00995.png --annotations testData/Med_Mn_10k_4x4_00995.json --config bandDetectorOptionsMagnetiteAccuracyTesting.yml --tile-from-single
python KikuchiBandWidthAutomator.py --config configs/ctf_ni_band_width_example.yml
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml --debug
python scripts/run_ebsd_compare_demo.py --config configs/ebsd_compare_config.yml
python -m kikuchiBandAnalyzer.automator_gui.main_window --config bandDetectorOptionsHcp.yml
python scripts/run_automator_gui_demo.py --debug
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.gui
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.cli --config configs/oh5_to_ang_exporter.yml
pytest -q
```

Note: proof screenshots used by the documentation live in `docs/screenshots/`.

## Automator GUI

The Automator GUI runs the same analysis engine as `KikuchiBandWidthAutomator.py`, but provides a visualization-first workflow (map/pattern/profile) and progress monitoring.

User guide:
- [`docs/automator_gui.md`](docs/automator_gui.md)

## OH5 to ANG Exporter

This GUI builds a new ANG file by combining:

- a modified OH5/HDF5 file (source of scalar/derived values), and
- a source ANG file (header template + baseline row layout).

It performs sanity checks on pixel counts before export, enforces locked mappings for `phi1/PHI/phi2`, supports user-defined source->target mappings (or formula->target mappings) for other columns, and can optionally write one ASCII mapping note line in the ANG header.

Each user mapping can also apply:
- optional formula expressions using OH5 scalar fields and numeric constants (for example `Band_Width * 120 + CI`),
- optional linear scaling from source min/max to a user-specified target range,
- optional output type conversion (`float`, `int` with nearest rounding, or `auto` inference from target column tokens).

User guide:
- [`docs/howto_oh5_to_ang_exporter.md`](docs/howto_oh5_to_ang_exporter.md)

Launch:

```bash
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.gui
```

CLI (non-interactive):

```bash
python -m kikuchiBandAnalyzer.oh5_to_ang_exporter.cli --config configs/oh5_to_ang_exporter.yml
```

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
- `kikuchiBandAnalyzer/band_width/`: package APIs and CLIs for the band-width pipeline
- `kikuchiBandAnalyzer/io/`: HDF5/OH5/ANG compatibility helpers
- `kikuchiBandAnalyzer/fields/`: derived-field registry exports
- `hdf5_image_export_and_validation.py`: export/reconstruct patterns for ML workflows
- `bandDetectorOptions*.yml`: example configuration files
- `VERSION`: single source of truth for the repo version
- `CHANGELOG.md`: release notes
- `testData/`: small example datasets and fixtures; generated outputs should go under ignored `outputs/`, `tmp/`, or external scratch folders

## Contributing

See `contribute.md` and `AGENTS.md` for contribution guidelines (docstrings, logging, debug/normal run modes, and non‑interactive scripts).
