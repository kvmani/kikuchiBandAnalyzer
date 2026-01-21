# EBSD Compare GUI (v1)

This package provides a field-discovery-driven comparator for two aligned EBSD scans stored in `.oh5` (HDF5) files. It discovers scalar maps and pattern stacks under `/<scan>/EBSD/Data/`, supports lazy pattern loading, and uses a YAML configuration to drive the probe panel contents and diff modes.

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

## CLI exports

```bash
python -m kikuchiBandAnalyzer.ebsd_compare.cli \
  --config configs/ebsd_compare_config.yml \
  --scan-a testData/Test_Ti.oh5 \
  --scan-b testData/Test_Ti_noisy.oh5
```

Exports land in `tmp/compare_exports` by default.

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
- `noisy_generation` and `demo` sections used by scripts
