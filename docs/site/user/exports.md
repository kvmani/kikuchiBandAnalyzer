# Exports

## `.ang`

The `.ang` export is intended for compatibility with EBSD tools. It should keep
the original scan geometry and append derived band-width fields without changing
the meaning of the existing Euler angle and coordinate columns.

## `.h5`

The `.h5` export is the structured scientific output. It is suitable for Python,
MATLAB, and downstream batch analysis.

Recommended groups:

- `/scan`: coordinates, dimensions, phase ids, and original indexing fields;
- `/band_width`: measured widths, per-HKL maps, quality flags, and edge metadata;
- `/provenance`: package version, configuration hash, input paths, and run time.

## `.oh5`

The `.oh5` export keeps compatibility with workflows that expect OH5-like field
organization. Comparison exports should include delta, absolute-delta, and ratio
fields with clear units and source-field metadata.

## Provenance

Every export should include enough metadata to answer:

- which input files were used;
- which phase and HKL definitions were active;
- which detector settings were used;
- which software version created the file;
- whether debug or normal mode produced the result.
