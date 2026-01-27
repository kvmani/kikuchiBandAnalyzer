# EBSD JSON Annotation & Output Schemas

This document describes the JSON formats consumed and produced by the Kikuchi Band Analyzer pipeline, including the new optional `pattern_path` input field and the `band_profile` output vector. It also maps JSON fields to CSV columns and HDF5 datasets.

## Input JSON schema (annotations)

Each entry corresponds to a pixel/pattern in the EBSD scan. The pipeline expects a list of entries unless the configuration enables tiling from a single entry.

```json
[
  {
    "x,y": [12, 34],
    "ind": 1234,
    "pattern_path": "patterns/pattern_12_34.png",
    "points": [
      {
        "hkl": "1,1,1",
        "central_line": [25.0, 40.0, 75.0, 40.0],
        "line_mid_xy": [50.0, 40.0],
        "line_dist": 3.5,
        "hkl_group": "111"
      }
    ]
  }
]
```

### Required fields

- `points` (list): Annotation list containing band candidates. Each entry must include:
  - `hkl` (string): Miller indices for the line.
  - `central_line` (list[float]): `[x1, y1, x2, y2]` endpoints in pixels.
  - `line_mid_xy` (list[float]): midpoint of the line.
  - `line_dist` (float): distance from pattern center.

### Optional fields

- `pattern_path` (string): Path to a standalone pattern image or a string like `"(row,col)"` when the pattern is stored inside an OH5/H5 dataset.

### Legacy/deprecated fields

- `x,y` and `ind` are accepted on input but re-derived during processing. They are retained for compatibility with legacy annotation files.

## Output JSON schema (after detection)

After processing, the pipeline adds a `bands` list to each entry. Each band records scalar metrics plus the `band_profile` vector and profile index metadata (`band_start_idx`, `central_peak_idx`, `band_end_idx`, and `profile_length`).

```json
{
  "x,y": [12, 34],
  "ind": 1234,
  "pattern_path": "patterns/pattern_12_34.png",
  "points": [...],
  "bands": [
    {
      "hkl": "1,1,1",
      "hkl_group": "111",
      "central_line": [25.1, 40.2, 74.8, 40.1],
      "line_mid_xy": [50.0, 40.0],
      "line_dist": 3.5,
      "bandWidth": 12.3,
      "psnr": 5.4,
      "band_peak": 200.0,
      "band_bkg": 40.0,
      "bandStart": 32,
      "bandEnd": 58,
      "centralPeak": 45,
      "band_start_idx": 32,
      "central_peak_idx": 45,
      "band_end_idx": 58,
      "profile_length": 80,
      "efficientlineIntensity": 41.2,
      "defficientlineIntensity": 39.8,
      "band_valid": true,
      "band_profile": [0.0, 1.2, 3.4, "..."]
    }
  ]
}
```

### `band_profile` meaning

`band_profile` is the summed intensity profile across the rectangular region centered on the detected band. The profile length is always `rectWidth * 4` samples (as configured in the YAML file). Values are in arbitrary intensity units derived from the source patterns.

### Profile index metadata

These values refer to indices into `band_profile`:

- `band_start_idx`: left local minimum used for bandwidth calculation (`-1` when unavailable)
- `central_peak_idx`: peak index used to split left/right minima search (`-1` when unavailable)
- `band_end_idx`: right local minimum used for bandwidth calculation (`-1` when unavailable)
- `profile_length`: expected length of `band_profile` for validation (typically `rectWidth * 4`)

## CSV output mapping

The CSV exports include scalar values per detected band:

| JSON field | CSV column |
| --- | --- |
| `x,y` | `X,Y` |
| `ind` | `Ind` |
| `central_line` | `Central Line` |
| `line_dist` | `Line Distance` |
| `bandWidth` | `Band Width` |
| `band_peak` | `band_peak` |
| `band_bkg` | `band_bkg` |
| `psnr` | `psnr` |
| `efficientlineIntensity` | `efficientlineIntensity` |
| `defficientlineIntensity` | `defficientlineIntensity` |
| `band_valid` | `band_valid` |

`band_profile` is not currently exported to CSV because it is a vector.

## HDF5 output mapping

The augmented HDF5 file receives derived datasets under `/<scan_name>/EBSD/Data/`:

| Output dataset | Description |
| --- | --- |
| `Band_Width` | Selected band width per pixel (float32). |
| `psnr` | Selected band PSNR per pixel (float32). |
| `efficientlineIntensity` | Efficient line intensity per pixel (float32). |
| `defficientlineIntensity` | Defficient line intensity per pixel (float32). |
| `band_intensity_ratio` | `I_eff / I_def` per pixel (float32). |
| `band_profile` | Summed band profile per pixel (float32, shape `n_pixels × (rectWidth*4)`). |
| `central_line` | Selected band central line per pixel (float32, shape `n_pixels × 4`). |
| `band_start_idx` | Left local minimum index per pixel (int32, `-1` when unavailable). |
| `band_end_idx` | Right local minimum index per pixel (int32, `-1` when unavailable). |
| `central_peak_idx` | Central peak index per pixel (int32, `-1` when unavailable). |
| `profile_length` | Expected profile length per pixel (int32). |
| `band_valid` | 1 when a valid best-band profile was stored (int8). |

`band_profile` and `central_line` are populated using the highest-PSNR valid band per pixel. Missing or invalid bands are stored as NaNs.
