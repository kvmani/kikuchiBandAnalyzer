# Tutorial: End-To-End Scan Processing

This tutorial describes the full scan workflow for both TSL and HKL-style data.

## TSL Input

```powershell
kikuchi-band-width --config configs/tsl_band_width_example.yml
```

The configuration identifies the `.oh5` or `.h5` input file and requested export
paths.

## HKL CTF Input

```powershell
kikuchi-band-width --config configs/ctf_ni_band_width_example.yml
```

The configuration identifies the `.ctf` file, the pattern folder, the file glob,
and the FCC Ni phase definition.

## Expected Outputs

At minimum, the workflow should be able to create:

- an `.ang` file with preserved scan geometry and appended band-width fields;
- an `.h5` file with structured arrays and provenance;
- an `.oh5` file when OH5-compatible export is requested;
- optional debug figures for small scans or selected pixels.

## Reproducibility Checklist

- commit small synthetic fixtures and golden outputs;
- keep large generated outputs outside version control;
- compare floating point outputs with explicit tolerances;
- record package version and configuration in every structured export.
