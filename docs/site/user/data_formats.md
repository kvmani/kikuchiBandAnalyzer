# Data Formats

## TSL `.oh5` And `.h5`

TSL-style HDF5 files keep indexing maps, metadata, and patterns in one container.
The reader discovers scalar maps, vector fields, grid dimensions, phase fields,
pattern dimensions, and optional band-profile datasets.

## HKL `.ctf` Plus Pattern Folder

HKL workflows separate metadata from images:

- the `.ctf` file stores EBSD indexing data such as `X`, `Y`, Euler angles,
  phase, bands, error, MAD, BC, and BS;
- a folder stores one Kikuchi pattern image for each scan pixel;
- filenames may encode scan order, coordinates, or acquisition order depending
  on the microscope export settings.

The foundational reader normalizes CTF columns into the same scan model used by
`.ang` and `.oh5` readers. Pattern folder acquisition validates image count,
shape consistency, coordinate mapping, and missing-file conditions before the
analysis starts.

## `.ang`

The `.ang` export is the minimum interoperable text output. For FCC Ni, the
pipeline preserves Euler angles, coordinates, phase-related columns, and appends
band-width fields using deterministic column names.

## Derived HDF5 And OH5 Outputs

Derived `.h5` and `.oh5` files are used for structured downstream analysis. They
store scan grids, band-width arrays, quality metadata, line/profile diagnostics
where requested, and enough provenance to reproduce the run.
