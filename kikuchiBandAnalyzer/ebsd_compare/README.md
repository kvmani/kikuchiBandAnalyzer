# EBSD Compare GUI (v2)

This package provides a field-discovery-driven comparator for EBSD scans stored in `.oh5` (HDF5) files. It discovers scalar maps and pattern stacks under `/<scan>/EBSD/Data/`, supports lazy pattern loading, and uses a YAML configuration to drive probe panel contents, diff modes, and alignment workflows.

## Field discovery

- The reader locates the first top-level scan group that is **not** `Manufacturer` or `Version`.
- Scalar maps are detected by their shape matching `(nRows, nColumns)` or `(nRows * nColumns,)`.
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

The GUI includes a log panel at the bottom of the main window. All intermediate results, warnings, and alignment diagnostics are logged via the standard `logging` package and streamed to the panel.

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

The demo script generates the noisy file (if needed), launches the GUI, auto-loads scans A/B, auto-probes the center pixel, and saves a proof screenshot to `tmp/ebsd_compare_gui_proof.png` (the screenshot is generated locally and not committed).

## Configuration

See `configs/ebsd_compare_config.yml` for:

- `default_map_field`
- `compare_fields.scalars` and `compare_fields.patterns`
- `display.map_diff_mode` and `display.pattern_diff_mode`
- `field_aliases` for alternate dataset names
- `alignment.*` for registration/warping configuration, control points, and saved alignment paths
- `logging.*` for GUI log level, format, and file logging
- `debug.*` for simulated data parameters
- `noisy_generation` and `demo` sections used by scripts
