# Single-Pattern PC Calibration Debugger

This guide explains how to use the single-pattern indexing GUI to tune pattern
center (PC) and detector geometry before running full-scan indexing or band-width
analysis. It is intended for students working with both TSL/EDAX `.oh5` files
and HKL/Oxford `.ctf` files with external pattern images.

## Purpose

The GUI lets you inspect one EBSP at a time and compare three overlays:

- **Dotted red lines**: experimental bands detected by kikuchipy/PyEBSDIndex
  Hough indexing.
- **Solid colored lines**: simulated Kikuchi bands recomputed from the solved or
  file-provided orientation.
- **Thick yellow line**: the specific band used for the band-profile plot.

For correct PC and detector geometry, the dotted experimental Hough bands, solid
simulated bands, and visible Kikuchi bands in the pattern should overlap as well
as possible.

## Environment Setup

Use a Python environment with all project dependencies installed:

```powershell
python -m pip install -r requirements.txt
```

Hough indexing requires `pyebsdindex`. If the Hough panel says that
`pyebsdindex` is missing, install it into the same environment used to launch
the GUI:

```powershell
python -m pip install "pyebsdindex>=0.3.9,<0.4"
```

Confirm which interpreter is being used:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip show pyebsdindex
```

## Launch Commands

Run these commands from the repository root.

For the bundled TSL/EDAX Ni reference:

```powershell
python -m kikuchiBandAnalyzer.single_pattern_solver.gui --config configs\single_pattern_da.yml
```

For the bundled HKL/Oxford CTF example:

```powershell
python -m kikuchiBandAnalyzer.single_pattern_solver.gui --config configs\single_pattern_ctf.yml
```

## Recommended CTF Debug Workflow

1. Select **HKL/Oxford CTF + pattern folder** in the input mode.
2. Set **Source** to the `.ctf` file.
3. Set **Pattern dir** to the folder containing the pattern images.
4. Set **Pattern map** to the filename convention. For the bundled toy data,
   this is:

   ```text
   {x}_{y}.tiff
   ```

5. Click **Load Source / Middle Pixel**. The GUI reads the scan dimensions and
   defaults to the middle pixel.
6. Keep **Run kikuchipy Hough indexing** enabled.
7. Keep **Overlay indexed orientation** enabled when you want the solid
   simulated lines to use the newly indexed single-pattern solution.
8. Adjust **PC x\***, **PC y\***, and **PC z\*** until the overlays agree.
9. Use the red draggable **PC** marker on the pattern to tune `PC x*` and
   `PC y*` interactively. Releasing the marker reruns the solution.
10. Inspect the log console at the bottom after each solve.

## What Each Plot Means

### EBSP Overlay

The central plot shows the experimental pattern with overlays:

- The red `PC` marker shows the current pattern-center location. Drag it to
  change `PC x*` and `PC y*`.
- Dotted red numbered lines are the top Hough-detected experimental bands.
  Their labels match the numbered points in the Hough plot.
- Solid colored lines are simulated Kikuchi lines for the configured HKL
  families.
- The thick yellow line labeled `profile {111}` marks the exact band used for
  the band-profile plot.

### Hough Transform

The lower-right plot shows the Hough/Radon image from the backend. The top five
selected peaks are numbered by descending detected intensity. Good indexing
starts with these selected peaks landing on clear local maxima.

### Band Profile

The upper-right plot shows the intensity profile sampled across the selected
profile band. The vertical markers indicate the estimated start, central peak,
and end positions.

## PC Convention Notes

The **PC convention** field matters:

- Use `edax` for TSL/EDAX `.oh5` and `.h5` data.
- Use `oxford` for HKL/Oxford `.ctf` data.

Using the wrong convention can shift the simulated overlay even when the Euler
angles and phase information are reasonable.

## Practical Signs Of A Good Calibration

A good PC/detector setup usually shows:

- Hough peak labels in the Hough panel lie on strong local maxima.
- Dotted red Hough lines follow visible experimental Kikuchi bands.
- Solid simulated Kikuchi lines overlap the same visible bands.
- The yellow profile band crosses a real, isolated band suitable for measuring
  width.
- Re-solving nearby pixels gives a similar visual agreement without large PC
  changes.

## Common Problems

### Hough panel says pyebsdindex is missing

Install dependencies into the same environment used to run the GUI:

```powershell
python -m pip install -r requirements.txt
```

### Hough peaks appear, but simulated lines are shifted

Check these settings:

- PC convention: `oxford` for CTF, `edax` for OH5/H5.
- PC values, especially `PC z*`.
- Sample tilt and detector tilt.
- Phase lattice parameter and space group.

### CTF pattern is missing

Check that the pattern template matches the file names. For example, if files
are named `7_10.tiff`, use:

```text
{x}_{y}.tiff
```

### Too many overlays are visible

The GUI displays only the top five Hough bands by default. You can reduce the
configured HKL families in the **HKL families** field if the solid simulated
overlay is too dense.

## Saving Results

Use **Save JSON** to export the current solution, including:

- pixel coordinate,
- Euler angles,
- detector geometry,
- Hough diagnostics,
- selected band profile metadata.

Use **Save PNG** to save the current overlay and profile figure for lab notes or
reports.
