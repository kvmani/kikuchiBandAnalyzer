# Testing Strategy

The test suite should cover three fixture levels.

## Synthetic Fixtures

Synthetic fixtures are generated, deterministic, and small. They should cover:

- tiny OH5/HDF5 files with scalar maps, vector fields, and missing fields;
- tiny CTF files with valid and invalid columns;
- deterministic Kikuchi-like images with known central lines and band widths;
- malformed YAML/JSON configurations.

## Golden Fixtures

Golden fixtures are approved regression samples. They should be cropped to a few
pixels and include expected `.json`, `.csv`, `.h5`, `.oh5`, and `.ang` outputs
with numeric tolerances.

## Stress Fixtures

Stress fixtures exercise failure modes:

- mismatched grids;
- missing optional datasets;
- zero denominators;
- NaN and infinite values;
- integer-only datasets;
- non-square patterns;
- invalid phase definitions.

Quality gates should include `pytest`, coverage, docstring checks, import smoke
tests, no package-level `print(...)`, and no interactive `input(...)` in normal
mode.
