# Testing Data Strategy

This project uses three fixture levels so scientific behavior can be tested
without mixing generated research artifacts into the source tree.

## Synthetic Fixtures

Synthetic fixtures are generated inside tests or stored as tiny files under
`testData/fixtures/` when reuse is valuable.

- Minimal OH5/HDF5 files with scalar maps, vector fields, pattern datasets,
  missing fields, malformed shapes, integer maps, and mismatched grids.
- Minimal ANG files with valid headers and a small number of data rows.
- Deterministic Kikuchi-like grayscale patterns with known central lines,
  expected edge indices, known band widths, and controlled noise.

## Golden Fixtures

Golden fixtures are approved regression files under `testData/golden/`.

- One cropped real OH5/ANG pair for debug-mode end-to-end pipeline checks.
- Expected JSON, CSV, HDF5/OH5, and ANG outputs for tolerance-based comparison.
- Comparison OH5 outputs for `delta`, `abs_delta`, and `ratio` modes.

Golden files should be small enough for normal Git review. Large generated data
must stay outside the repo or under ignored scratch/output directories.

## Stress Fixtures

Stress fixtures cover failure modes and edge cases.

- Alignment-required scans with mismatched grid sizes.
- Missing optional datasets such as `band_profile`, `central_line`,
  `Pattern Height`, and `Pattern Width`.
- Zero denominators, NaN/Inf values, phase-like fields, malformed JSON,
  invalid YAML, non-square patterns, and unsupported dataset shapes.

## Quality Gates

The default local gate is:

```bash
python -m pytest -q
python -m pytest --collect-only -q
```

Package code must remain non-interactive and logging-based. Tests enforce no
`print()` or `input()` calls inside `kikuchiBandAnalyzer/`. Legacy top-level
scripts should be converted into package CLIs as they are touched.
