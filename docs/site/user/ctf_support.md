# HKL CTF Support

HKL CTF support is a first-class migration goal. The immediate production target
is FCC Ni, with general phase support designed into the configuration and data
model.

## Required Inputs

- one `.ctf` file with indexing data;
- one folder containing exactly one pattern image for each scan pixel;
- an explicit phase section identifying Ni/FCC and the HKL families to measure;
- a mapping rule when filenames do not follow raster acquisition order.

## Validation Rules

The CTF workflow should fail early when:

- required CTF columns are missing;
- the number of patterns does not match the CTF grid;
- pattern images have mixed dimensions or unsupported bit depth;
- Euler angles contain NaN or non-finite values;
- coordinates are not compatible with a rectangular scan grid;
- the phase is not FCC Ni and no explicit phase adapter is available.

## Corrective Error Messages

Errors should tell the user what to fix. For example, a pattern-count mismatch
should report the expected count from the CTF grid, the discovered count in the
folder, the glob pattern used, and whether hidden or unsupported files were
ignored.
