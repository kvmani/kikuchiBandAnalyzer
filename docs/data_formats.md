# Data Formats (JSON + OH5/HDF5)

This document defines the required data formats that flow end-to-end through:

1) **Intermediate JSON** (per-pixel / per-band results)
2) **Final OH5/HDF5 outputs** written under `/<scan_name>/EBSD/Data/`

The goal is that downstream GUIs (Automator GUI and EBSD Comparator) can reliably discover and visualize band profiles and band-width metadata.

## Intermediate JSON (per-band fields)

Each EBSD pixel entry contains a `bands` list. For each detected band, the following keys must be present for downstream visualization:

### Required

- `band_profile` (list[float])
  - 1D intensity profile vector for the detected band.
- `central_line` (list[float] of length 4)
  - `[x1, y1, x2, y2]` endpoints of the band center line in **pattern pixel coordinates**.
- `band_start_idx` (int)
  - Index into `band_profile` corresponding to the **left local minimum** used for bandwidth calculation (`-1` when unavailable).
- `band_end_idx` (int)
  - Index into `band_profile` corresponding to the **right local minimum** used for bandwidth calculation (`-1` when unavailable).

### Strongly recommended

- `central_peak_idx` (int)
  - Index into `band_profile` of the central peak used to split the left/right minima search (`-1` when unavailable).
- `profile_length` (int)
  - Expected profile length, used for validation (typically `rectWidth * 4`).

### Legacy compatibility

Older keys are preserved for backward compatibility:

- `bandStart`, `bandEnd`, `centralPeak`

Downstream code should prefer the snake_case variants.

## OH5/HDF5 datasets (per-pixel outputs)

Final outputs are stored under:

`/<scan_name>/EBSD/Data/`

### Vector datasets

These datasets are *per pixel* and store a fixed-length vector:

- `band_profile`
  - **Shape:** `(nPixels, profile_len)`
  - **Dtype:** `float32`
  - **Missing/invalid pixels:** filled with `NaN`
- `central_line`
  - **Shape:** `(nPixels, 4)`
  - **Dtype:** `float32`
  - **Missing/invalid pixels:** filled with `NaN`

### Index datasets

These datasets are *per pixel* and store indices into `band_profile`:

- `band_start_idx`
  - **Shape:** `(nPixels,)`
  - **Dtype:** `int32`
  - **Missing/invalid pixels:** `-1`
- `band_end_idx`
  - **Shape:** `(nPixels,)`
  - **Dtype:** `int32`
  - **Missing/invalid pixels:** `-1`
- `central_peak_idx`
  - **Shape:** `(nPixels,)`
  - **Dtype:** `int32`
  - **Missing/invalid pixels:** `-1`
- `profile_length`
  - **Shape:** `(nPixels,)`
  - **Dtype:** `int32`
  - **Value:** typically `rectWidth * 4` for all pixels

### Validity dataset

- `band_valid`
  - **Shape:** `(nPixels,)`
  - **Dtype:** `int8`
  - **Semantics:** `1` when a valid best-band profile was stored for the pixel, `0` otherwise

### Notes

- `profile_len` is expected to be constant across the scan (guarded in code; mismatches log warnings).
- Band profiles are selected **per pixel** using the pipeline’s “best band by PSNR” rule.
- The EBSD Comparator and Automator GUI treat `band_profile`/`central_line` as *vector fields* and will gracefully disable the feature when these datasets are absent (older OH5 files).

