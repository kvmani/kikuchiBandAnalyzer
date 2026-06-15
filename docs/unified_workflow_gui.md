# Unified EBSD Band-Width Workflow GUI

The unified workflow GUI prepares either TSL/EDAX OH5/H5 plus ANG data or
HKL/Oxford CTF data with an external pattern folder. It combines interactive
single-pattern calibration with full-scan band-width processing.

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

## Recommended Workflow

1. Select OH5/H5 plus its companion ANG, or select CTF plus its pattern folder.
2. Verify the phase, lattice, PC convention, PC values, tilts, and HKL families.
3. Click **Prepare / Preview**. The middle scan pixel is selected automatically.
4. Choose arbitrary X/Y pixels and click **Solve Selected Pattern**.
5. Drag the PC marker in the diagnostic pattern. With automatic re-solving
   enabled, the Hough transform and indexed overlay update after the drag.
6. Repeat on patterns from different parts of the scan before accepting the PC.
7. Click **Run Full Band-Width Analysis** to process the complete scan.

During full processing, the Batch Result inspector displays throttled live
pattern overlays and profiles. The map tabs show linked IQ and IPF-X/IPF-Y/IPF-Z,
band-width, PSNR, validity, strain, and stress maps. Clicking either map selects
the same pixel and marks it with a bold plus sign.

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

The full run writes the legacy modified H5/OH5 and ANG files, raw and filtered
CSV files, stored band profiles and line coordinates, derived metric datasets,
and the resolved YAML configuration. The log console records preparation,
simulation, processing, export, warnings, and failures.
