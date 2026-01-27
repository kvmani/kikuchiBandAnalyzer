# EBSD Compare GUI (v2)

This package provides a field-discovery-driven comparator for EBSD scans stored in `.oh5` (HDF5) files. It discovers scalar maps and pattern stacks under `/<scan>/EBSD/Data/`, supports lazy pattern loading, and uses a YAML configuration to drive probe panel contents, diff modes, and alignment workflows.

## Release 1.0.0 highlights

- YAML `fields` list enables comparison of any scalar map (not just IQ/CI).
- Missing fields are skipped with warnings in both logs and the GUI.
- Linked pan/zoom is controlled via `sync_navigation`.
- Map contrast controls update reliably across all panels.

## Field discovery

- The reader locates the first top-level scan group that is **not** `Manufacturer` or `Version`.
- Scalar maps are detected by their shape matching `(nRows, nColumns)` or `(nRows * nColumns,)`.
- Vector-per-pixel datasets are detected by shapes like `(nRows * nColumns, k)` or `(nRows, nColumns, k)` (for example `band_profile` and `central_line`).
- Pattern stacks are detected by shapes like `(nRows * nColumns, height, width)` or `(nRows, nColumns, height, width)`.
- Field aliasing is supported via the `field_aliases` section of the YAML config.

## GUI usage

```bash
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml
```

You can also pass scans directly:

```bash
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window \
  --config configs/ebsd_compare_config.yml \
  --scan-a testData/Test_Ti.oh5 \
  --scan-b testData/Test_Ti_noisy.oh5
```

After loading both scans, the GUI automatically probes the middle pixel to populate the probe table and patterns panel.

The GUI uses a compact three-row control strip (file paths, display/coordinate controls, and auto-scan/status) so the map and pattern viewers remain dominant.

### Selection + auto-scan

- Coordinate convention: **X = column**, **Y = row** (zero-based indices).
- You can click any map or type X/Y in the always-visible inputs. Press Enter or tab out to update the selection.
- The auto-scan controls play a raster animation from `(0, 0)` across X, then Y, updating patterns live.
- Use the Speed control (milliseconds per step) to slow down or speed up the raster.

### Map + pattern controls

- Each map preview includes Home/Zoom/Pan controls; zooming or panning one map syncs the other two when `sync_navigation` is enabled.
- Map previews render in grayscale with per-map contrast percentiles (low/high) for brightness/contrast tuning.
- Pattern panels include Home/Zoom/Pan controls and stay synchronized while inspecting patterns when `sync_navigation` is enabled.
- The top controls are compact to keep the map and pattern viewers dominant, with toolbars embedded inside each viewer.

### Band profile comparison (new)

If your OH5 files contain the exported band datasets (`band_profile`, `central_line`, `band_start_idx`, `band_end_idx`, ...), the GUI enables a profile comparison plot:

- Plots `band_profile` from Scan A and Scan B on shared axes (optional normalization).
- Overlays `central_line` on patterns (toggleable per pattern).
- Adds probe-table rows for key band scalar fields when present.

User guide: `docs/ebsd_comparator_band_profiles.md`

### Export comparison OH5

The GUI includes an **Export Comparison OH5** button that writes an OH5 file designed to be loaded by downstream TSL/OIM tooling.

- Output path: written next to scan A as `{stemA}_{stemB}_comparison.oh5`
- Mode: choose **Delta (A - B)**, **Absolute Delta**, or **Ratio (A / B)** at export time
- Fields: exports all common scalar maps discovered under `/<scan>/EBSD/Data/` (skips Phase-like fields)
- Layout: copies scan A as a template and overwrites each exported scalar dataset with the comparison map so dataset names remain unchanged
- Metadata: stores export context and alignment details under `/<scan>/EBSD/Compare/` (including an `alignment_yaml` payload when alignment is active)

### Registration + alignment

If the scans do not share the same grid shape, the GUI launches a registration dialog. The registration tool is designed for research-grade alignment workflows and includes:

- Field selection (IQ/CI or any shared scalar map) for registration.
- Contrast adjustment with percentile clipping (optionally linked between scans).
- Zoom/pan controls via the Matplotlib navigation toolbar.
- Linked view toggles to keep both images in sync during zooming.
- Point-pair picking (click A then B), on-image annotations, and editable tables.
- RANSAC alignment with inlier/outlier labeling, residuals, and RMS summary.
- Optional alignment export to YAML for reuse in CLI workflows.

Once alignment is accepted, all downstream comparisons use the aligned scan B data.

## Logging

The GUI includes a log console docked at the bottom of the main window. Logs are timestamped and leveled, include colored icons, and can be filtered by level and searched by substring. Use auto-scroll to follow live updates, Clear Logs to reset the view, Copy Selected/Copy All to share logs in bug reports, and Save… to export the visible log rows to a text file.

## Debug mode

Both the GUI and CLI support `--debug` to load a small simulated dataset when scan paths are omitted:

```bash
python -m kikuchiBandAnalyzer.ebsd_compare.gui.main_window --config configs/ebsd_compare_config.yml --debug
python -m kikuchiBandAnalyzer.ebsd_compare.cli --config configs/ebsd_compare_config.yml --debug
```

## CLI exports

```bash
python -m kikuchiBandAnalyzer.ebsd_compare.cli \
  --config configs/ebsd_compare_config.yml \
  --scan-a testData/Test_Ti.oh5 \
  --scan-b testData/Test_Ti_noisy.oh5
```

Exports land in `tmp/compare_exports` by default.

If your scans are misaligned, supply a precomputed alignment in the YAML configuration (see below), or use the GUI registration tool to generate one and point the CLI to it.

## Generate a noisy OH5 file

```bash
python scripts/make_noisy_oh5.py --config configs/ebsd_compare_config.yml
```

This produces `testData/Test_Ti_noisy.oh5` by adding deterministic Gaussian noise to the IQ and CI datasets only. The noisy file is generated locally and is not committed to version control.

## Demo + proof screenshot

```bash
python scripts/run_ebsd_compare_demo.py --config configs/ebsd_compare_config.yml
```

The demo script generates a small simulated dataset in debug mode, launches the GUI, auto-loads scans A/B, auto-probes the center pixel, and saves a proof screenshot to `docs/screenshots/ebsd_compare_band_profile_proof.png`.

## Configuration

See `configs/ebsd_compare_config.yml` for:

- `default_map_field`
- `fields` (preferred scalar field list) or `compare_fields.scalars` (legacy)
- `compare_fields.patterns`
- `display.map_diff_mode` and `display.pattern_diff_mode`
- `field_aliases` for alternate dataset names
- `sync_navigation` to toggle linked pan/zoom across correlated viewers
- `alignment.*` for registration/warping configuration, control points, and saved alignment paths
- `logging.*` for GUI log level, format, and file logging
- `auto_scan.delay_ms`, `auto_scan.min_delay_ms`, `auto_scan.max_delay_ms` for auto-scan playback speed limits
- `debug.*` for simulated data parameters
- `noisy_generation` and `demo` sections used by scripts

If a configured field is missing in a scan, the GUI logs a warning, shows a UI warning, and skips that field. When none of the configured fields are present, the GUI falls back to all common scalar fields.

Example scalar field selection:

```yaml
ebsd_compare:
  default_map_field: IQ
  fields:
    - IQ
    - CI
    - Band_Width
    - psnr
    - band_intensity_ratio
    - band_intensity_diff_norm
  sync_navigation: true
  compare_fields:
    patterns:
      - Pattern
```
