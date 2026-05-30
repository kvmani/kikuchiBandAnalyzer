# Configuration

Normal mode is non-interactive and config driven. A configuration file should
describe inputs, phase assumptions, detector options, export targets, and debug
behavior explicitly.

```yaml
input:
  format: ctf
  ctf_path: testData/fixtures/ni_scan.ctf
  pattern_folder: testData/fixtures/ni_patterns
  pattern_glob: "*.tif"

phase:
  name: Ni
  crystal_structure: FCC
  lattice_parameter_angstrom: 3.52
  hkls:
    - [1, 1, 1]
    - [2, 0, 0]
    - [2, 2, 0]

detector:
  band_width_threshold: 0.5
  debug: false
  output_intermediate_plots: false

export:
  ang_path: outputs/ni_band_widths.ang
  h5_path: outputs/ni_band_widths.h5
  oh5_path: outputs/ni_band_widths.oh5
```

Configuration validation should fail early with messages that describe the
incorrect key, why it is invalid, and how to fix it. Required checks include:

- input paths exist and point to the expected file or directory type;
- grid dimensions are consistent with the number of patterns;
- phase definitions include an explicit phase name and HKL list;
- crop bounds and profile settings are inside pattern dimensions;
- export paths are writable and do not silently overwrite protected data.

Debug mode may synthesize input data. Normal mode must never require source-code
edits or interactive prompts.
