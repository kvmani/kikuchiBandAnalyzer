# Unified EBSD Band-Width Workflow GUI

The unified workflow GUI is the recommended interactive entry point for running
Kikuchi band-width analysis on a full EBSD scan. It prepares either TSL/EDAX
OH5/H5 plus ANG data or HKL/Oxford CTF data with an external pattern folder,
lets the user tune detector geometry on individual patterns, and then runs the
complete legacy band-width pipeline with live progress and map displays.

Scientifically, this workflow measures changes in selected Kikuchi band widths
from experimentally acquired EBSPs. Those widths are connected to changes in
interplanar spacing and can be used as local crystallographic/strain-sensitive
contrast after the EBSD acquisition orientations are known. The GUI is designed
to make detector calibration and pattern-quality checks visible before a full
scan is processed, while preserving the original acquisition Euler angles in all
exported files.

## Launch

Start with defaults:

```powershell
python -m kikuchiBandAnalyzer.workflow_gui
```

Load a YAML configuration into editable controls:

```powershell
python -m kikuchiBandAnalyzer.workflow_gui --config configs\unified_workflow_da.yml
```

The GUI accepts both the nested single-pattern configuration schema and the
flat batch configuration schema. The resolved settings used for a full run are
written to `workflow_band_width_config.yml` in the selected output directory.

## Inputs

For the conventional TSL/EDAX route, provide:

- **Source**: `.oh5`, `.h5`, or `.hdf5` file containing the EBSD scan and EBSP
  patterns.
- **ANG**: matching `.ang` file with the same scan grid and original Euler
  angles.
- **Output dir**: folder where prepared copies, modified files, maps, logs, and
  summaries will be written.

For the HKL/Oxford CTF route, provide:

- **Source**: `.ctf` text file containing scan coordinates, Euler angles, MAD,
  phase, and map metadata.
- **Patterns**: folder containing the pattern images referenced by the naming
  template.
- **CTF pattern map**: filename template, for example `{x}_{y}.tiff`, used to
  map scan pixel coordinates to pattern images.
- **Output dir**: folder where the CTF data will first be translated into
  DA-compatible `.h5`, `.oh5`, and `.ang` files before analysis.

For both routes, check these analysis inputs before running:

- **Phase** and **Space group**: currently intended for FCC phases in this GUI
  workflow.
- **Lattice**: six values `a, b, c, alpha, beta, gamma`.
- **HKL list**: families used for simulated Kikuchi overlays.
- **Desired HKL**: band family used for the reported band-width measurement,
  for example `1,1,1`.
- **PC**: pattern center as `pcx, pcy, pcz`.
- **PC convention**: use `edax` for TSL/EDAX data and `oxford` for HKL/Oxford
  CTF data unless you intentionally converted conventions elsewhere.
- **Sample tilt**, **Detector tilt**, and **Azimuthal**: detector geometry used
  by kikuchipy simulation and diagnostic indexing.
- **rectWidth** and **min_psnr**: band-profile extraction width and acceptance
  threshold.

## Recommended Workflow

1. Select OH5/H5 plus its companion ANG, or select CTF plus its pattern folder.
2. Verify the phase, lattice, PC convention, PC values, tilts, and HKL families.
3. Click **Prepare / Preview**. The middle scan pixel is selected automatically.
4. Inspect the IQ/IPF preview maps. Click a few map pixels or type X/Y values
   in **Selected Pattern**.
5. Click **Solve Selected Pattern** for each diagnostic pixel.
6. Drag the PC marker in the diagnostic pattern. With automatic re-solving
   enabled, the Hough transform and indexed overlay update after the drag.
7. Compare the dotted experimental Hough bands, solid simulated Kikuchi lines,
   and highlighted profile band. They should be geometrically consistent when
   PC and detector geometry are reasonable.
8. Repeat on patterns from different parts of the scan before accepting the PC.
9. Click **Run Full Band-Width Analysis** to process the complete scan.

During full processing, the Batch Result inspector displays throttled live
pattern overlays and profiles. The map tabs show linked IQ and IPF-X/IPF-Y/IPF-Z,
band-width, PSNR, validity, strain, and stress maps. Clicking either map selects
the same pixel and marks it with a bold plus sign.

## Inspecting Results in the GUI

After a run completes, click any pixel in an IQ or result map. The GUI updates:

- the selected-pixel marker on all map tabs;
- the EBSP pattern image for that pixel;
- solid simulated Kikuchi lines from the original acquisition orientation;
- the highlighted band used for the selected band-profile measurement;
- the band profile plot with start, peak, and end markers when available;
- scalar metrics such as `Band_Width`, `psnr`, `band_valid`, strain, stress,
  and `band_intensity_ratio`.

Use the map tabs to compare different outputs:

- **IPF-X**, **IPF-Y**, **IPF-Z**: orientation color maps generated from the
  original Euler angles.
- **Band Width**: measured width for the configured desired HKL family.
- **PSNR**: quality metric from the band-profile fit/detection.
- **Validity**: mask showing where the selected band measurement passed
  validity criteria.
- **Strain** and **Stress**: derived maps based on the configured reference
  width and elastic modulus.

The Matplotlib toolbars provide pan, zoom, and reset controls. The contrast
controls adjust the displayed percentile range for grayscale/scalar maps and
patterns without changing the saved data.

## Orientation Preservation

Hough indexing in this GUI is diagnostic only. It helps assess the detector PC
and geometry for an individual pattern, but its Euler solution is never written
to ANG, H5, or OH5 output.

Production simulated bands and IPF maps use the original acquisition Euler
angles. Generated ANG files preserve the original Euler columns and replace only
the legacy PRIAS metric columns:

- `Band_Width` -> `PRIAS Bottom Strip`
- `psnr` -> `PRIAS Center Square`
- `band_intensity_ratio` -> `PRIAS Top Strip`

## Outputs

The full run writes these files into the selected output directory:

- `<scan>_modified.h5`: HDF5 output with computed band metrics added under the
  EBSD data group.
- `<scan>_modified.oh5`: OH5-compatible copy for tools that expect the `.oh5`
  extension.
- `<scan>_modified.ang`: legacy-compatible ANG export with original Euler
  columns preserved and PRIAS metric columns updated.
- `<scan>_bandOutputData.csv`: raw per-pixel/per-band results.
- `<scan>_filtered_band_data.csv`: filtered best-band style output.
- `workflow_band_width_config.yml`: resolved configuration actually used for
  the run.
- `<scan>_run_summary.json`: compact summary of processed pixels and quality
  statistics.
- map PNG files for IQ and result tabs.

The modified HDF5/OH5 data group includes fields such as:

- `Band_Width`
- `psnr`
- `band_intensity_ratio`
- `band_intensity_diff_norm`
- `band_profile`
- `central_line`
- `band_start_idx`
- `central_peak_idx`
- `band_end_idx`
- `profile_length`
- `band_valid`
- `strain`
- `stress`

The bottom log console records preparation, simulation, per-stage progress,
live processing status, export paths, warnings, and failures. Use **Save...** in
the log console when you need a text record of a run for debugging or reporting.

## Beginner Checks and Common Problems

- If the pattern overlay is shifted, first confirm the **PC convention**. TSL
  and HKL pattern-center conventions are different, and kikuchipy must receive
  the correct convention.
- If the CTF route shows missing patterns, check the pattern folder and filename
  template. The template must reproduce the image names from pixel coordinates.
- If the band profile is shown but marked invalid, the detector found a finite
  profile but it did not pass the configured quality criteria. Inspect `psnr`,
  `min_psnr`, and the overlay geometry.
- If maps look structurally wrong, compare IPF maps from the source, generated
  ANG, and generated OH5. They should preserve the original microstructure
  because exported Euler angles are not replaced by diagnostic Hough solutions.
- If processing is slow, use a tiny fixture or a copied subset while tuning PC.
  For original-orientation production runs, avoid cropping only the pattern
  stack because it must remain aligned with the original Euler grid.
