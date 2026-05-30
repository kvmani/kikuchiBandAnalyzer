# HKL/Oxford CTF Support

HKL/Oxford CTF acquisitions differ from EDAX/TSL OH5/HDF5 scans:

- The `.ctf` file is a text table containing indexing metadata such as phase,
  X/Y position, Euler angles, MAD/fit-like values, band count, and contrast.
- The diffraction patterns are stored separately as image files in a folder,
  one image per pixel.

The repository now treats this as a first-class source shape through the common
`ScanDataset` reader abstraction:

- `.oh5`, `.h5`, and `.hdf5` files use `OH5ScanFileReader`.
- `.ctf` files use `CtfPatternScanFileReader`.
- `open_scan_dataset()` selects the reader by file extension.

## Pattern Folder Mapping

For CTF scans, configure the pattern folder in `configs/ebsd_compare_config.yml`:

```yaml
ebsd_compare:
  ctf:
    pattern_dirs:
      scan_a: path/to/scan_a_patterns
      scan_b: path/to/scan_b_patterns
    pattern_template: "pattern_x{x}_y{y}.png"
```

If `pattern_template` is omitted, images in the pattern folder are sorted by
filename and mapped in row-major order: `index = y * nx + x`.

If no pattern directory is configured, the reader still exposes CTF scalar maps
for comparison. Pattern panels simply have no `Pattern` field.

For the bundled `testData/hkl_ctf_test_data` fixture, the TSL compatibility
converter uses an explicit coordinate mapping:

```text
CTF pixel (x, y) -> Binned_2x2/{x}_{y}.tiff
```

The CTF grid is 15 x 21. The pattern folder contains additional images from the
larger HKL acquisition; images outside the CTF subset are ignored.

## CTF Fields

All numeric CTF columns are exposed as scalar fields. Common fields include:

- `Phase`
- `X`, `Y`
- `Bands`
- `Error`
- `Euler1`, `Euler2`, `Euler3`
- `MAD`
- `BC`, `BS`, or other vendor-exported contrast fields

Aliases in the compare config can map project terminology onto these names. For
example, `IQ` can alias to `BC`, and `Fit` can alias to `MAD`.

## Band-Width Automator Scope

The band-width automator can now prepare `.ctf` plus pattern-folder acquisitions
into the same internal HDF5/CSV output flow used by OH5/HDF5 runs. When no
precomputed line annotations are supplied, the automator uses the CTF Euler
angles to simulate Kikuchi line positions directly.

Minimum CTF config keys:

```yaml
ctf_file_path: path/to/sample.ctf
pattern_folder: path/to/pattern_images
desired_hkl: "110"
desired_hkl_ref_width: 1.0
elastic_modulus: 1.0
rectWidth: 20
min_psnr: 1.0
hkl_list:
  - [1, 1, 0]
phase_list:
  name: Ni
  space_group: 225
  lattice: [1, 1, 1, 90, 90, 90]
  atoms:
    - element: Ni
      position: [0, 0, 0]
ctf_detector:
  convention: oxford
  pc: [0.5, 0.5, 0.5]
  sample_tilt: 70.0
  tilt: 0.0
  azimuthal: 0.0
ctf_euler_direction: lab2crystal
```

An FCC Ni starter config is provided at
[`configs/ctf_ni_band_width_example.yml`](../configs/ctf_ni_band_width_example.yml).

The requested small FCC validation fixture is specified in
[`docs/hkl_ctf_test_data_request.md`](hkl_ctf_test_data_request.md). It defines
the expected `.ctf` plus pattern-folder layout, HKL reference IPF exports,
acquisition metadata, and repository size limits.

The automator writes:

- `<ctf_stem>_bandOutputData.csv`
- `<ctf_stem>_filtered_band_data.csv`
- `<ctf_stem>_modified.h5`
- `<ctf_stem>_modified.oh5`
- `<ctf_stem>_modified.ang`

For CTF sources, the ANG export uses canonical FCC/Ni-compatible columns:
`phi1`, `PHI`, `phi2`, `x`, `y`, `IQ`, `CI`, `Phase index`, `Fit`, and PRIAS
metric columns populated from `Band_Width`, `psnr`, and `band_intensity_ratio`.

If a trusted annotation JSON is already available, set
`band_annotation_json_path` or `line_annotation_json_path`. In that case, the
automator skips CTF Euler simulation and uses the provided annotations.

## FCC Fixture Conversion To TSL ANG/H5/OH5

The project includes a limited FCC-only converter for the bundled HKL fixture:

```bash
python scripts/convert_hkl_ctf_test_data.py
```

It writes:

- `testData/hkl_ctf_test_data/converted_tsl/Subset_HKL_TSL.ang`
- `testData/hkl_ctf_test_data/converted_tsl/Subset_HKL_TSL.h5`
- `testData/hkl_ctf_test_data/converted_tsl/Subset_HKL_TSL.oh5`

The converter copies the DA/Ni FCC phase metadata and hkl families, then writes
the fixture phase as `Cr` with cubic lattice constants `a=b=c=2.91`. It maps CTF
fields into DA-compatible names:

- `Euler1`, `Euler2`, `Euler3` -> `Phi1`, `Phi`, `Phi2` in radians
- `BC` -> `IQ`
- `MAD` -> `Fit`
- `CI = 1 - MAD / max(MAD)`, clipped to `[0, 1]`
- `BS` -> `SEM Signal`
- valid CTF phase pixels -> TSL phase `0`; unindexed CTF phase `0` -> `-1`

Run the validation renderer after conversion:

```bash
python scripts/render_hkl_ctf_tsl_validation.py
```

It extracts the HKL IPF-X reference from `AcquisitionDetails.pptx`, renders
CTF/ANG/H5 IPF-X maps with the same orientation-color backend, validates that
the generated HDF5 loads through `kikuchipy`, and writes a one-slide comparison
deck under `testData/hkl_ctf_test_data/validation/`.

The same preparation path is available from the workflow GUI:

```bash
python -m kikuchiBandAnalyzer.workflow_gui.main_window
```

Choose `HKL/Oxford CTF + patterns`, select the `.ctf` file and pattern folder,
set the pattern mapping to `{x}_{y}.tiff`, and keep the detector convention as
`oxford`. For conventional TSL/EDAX inputs, choose `TSL/EDAX OH5/H5 + ANG` and
keep the detector convention as `edax` unless the acquisition metadata states a
different convention.

## Production Checks

The direct CTF path performs these checks before detection:

- CTF must contain finite `Euler1`, `Euler2`, and `Euler3` columns.
- Euler row count must match `XCells * YCells` and the pattern-folder grid.
- Every pattern image must exist and all patterns must have the same 2D shape.
- `ctf_detector.pc` must contain three values. Unusual PC values trigger a
  warning with a corrective suggestion.
- Missing detector geometry values are allowed only with warnings. For
  production analysis, set `sample_tilt`, `tilt`, `azimuthal`, `pc`, and
  `convention` explicitly.
- If simulated line annotations are sparse, the automator pads missing pixels
  with empty annotations and warns to check detector geometry, PC, sample tilt,
  Euler convention, and `desired_hkl`.
