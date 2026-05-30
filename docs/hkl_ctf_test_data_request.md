# HKL CTF Test Data Request

Use this note as the email/request text for preparing the small HKL/Oxford CTF
fixture used by tests, documentation, and tutorials.

## Email To Student

Subject: Request for a small HKL/Oxford EBSD CTF fixture for repository tests

Please prepare a small EBSD test dataset from HKL/Oxford acquisition software.
The purpose is not scientific analysis of the sample; it is to give the code a
compact, trusted input that verifies the full workflow from HKL `.ctf` plus
pattern images through indexing checks, IPF map plotting, band-width
calculation, and export to `.h5`, `.oh5`, and `.ang`.

The dataset must stay small enough to live in the Git repository. Please follow
the steps below exactly and keep notes of any software settings you use.

### 1. Select A Simple Dataset

1. Use a single-phase FCC material, preferably Ni or another well-indexed cubic
   FCC sample.
2. Select a small scan area containing 3 to 4 grains. The IPF maps should show
   multiple distinct orientations, but the scan must not be large.
3. Keep the grid intentionally small. A target of roughly 20 x 20 to 40 x 40
   points is enough. Do not export a full production-sized map.
4. Use 4x4 pattern binning to reduce pattern dimensions and repository size.
5. Export patterns as 8-bit grayscale images. Do not use 12-bit, 16-bit, TIFF
   stacks, or floating-point images for this fixture.

### 2. Export The HKL Files

1. Export the indexed scan as a `.ctf` file.
2. Export one pattern image per scan point into a separate folder.
3. Use a deterministic pattern naming scheme. Preferred names are either:
   `pattern_000000.png`, `pattern_000001.png`, ... in row-major scan order, or
   `pattern_x0_y0.png`, `pattern_x1_y0.png`, ... with explicit coordinates.
4. Do not rename the `.ctf` columns manually. We need the original HKL/Oxford
   column names and header content for parser validation.
5. If HKL software offers options for Euler angle convention, detector
   geometry, or pattern center export, keep the default Oxford/HKL convention
   and record exactly what was used.

### 3. Record Acquisition And Geometry Metadata

Create a small text or YAML file named `acquisition_notes.yml` beside the `.ctf`
file. Include the following values as reported by the microscope or HKL
software:

```yaml
sample_name: ""
material: "Ni or other FCC phase"
crystal_structure: "FCC"
space_group: 225
scan_grid:
  x_cells:
  y_cells:
  x_step:
  y_step:
pattern_export:
  image_format: "png"
  bit_depth: 8
  binning: "4x4"
  pattern_width_px:
  pattern_height_px:
detector_geometry:
  hkl_pattern_center:
  hkl_pattern_center_definition:
  sample_tilt_deg:
  detector_tilt_deg:
  detector_azimuth_deg:
  camera_length_or_dd:
software:
  hkl_or_aztec_version:
  export_date:
notes:
  - ""
```

The pattern center information is especially important. The code must handle
the difference between HKL/Oxford and EDAX/TSL pattern-center definitions
accurately and without the user manually editing output files. Please include
both the numeric values and the definition used by the software, for example
whether the origin is top-left or bottom-left, whether PC coordinates are
normalized by detector width/height, and how detector distance is defined.

### 4. Export HKL Reference Images

From HKL/Oxford software, export reference images for visual cross-checking:

1. `hkl_ipf_x.png`
2. `hkl_ipf_y.png`
3. `hkl_ipf_z.png`
4. Any HKL phase, band contrast, MAD, or fit map that is easy to export.

These reference images are not the truth for every number, but they let us
quickly confirm that our parsed orientations and plotted IPF maps are not
rotated, transposed, mirrored, or using the wrong convention.

### 5. Expected Repository Folder Layout

Please package the fixture with this structure:

```text
testData/hkl_ctf_fcc_small/
  README.md
  acquisition_notes.yml
  scan.ctf
  patterns/
    pattern_000000.png
    pattern_000001.png
    ...
  hkl_reference/
    hkl_ipf_x.png
    hkl_ipf_y.png
    hkl_ipf_z.png
    hkl_band_contrast.png
    hkl_mad_or_fit.png
  expected_outputs/
    scan_modified.h5
    scan_modified.oh5
    scan_modified.ang
```

If the generated `.h5`, `.oh5`, or `.ang` files are too large, keep only the
smallest files needed for automated tests and share the larger files separately
for manual validation. The primary goal is a Git-friendly fixture.

### 6. Validation The Dataset Must Enable

The fixture should allow us to start with:

```text
scan.ctf
patterns/
```

and end with:

```text
scan_modified.h5
scan_modified.oh5
scan_modified.ang
```

without manual intervention.

The tests and tutorials should be able to verify:

1. The `.ctf` reader correctly parses grid size, step size, phase, Euler
   angles, MAD/fit, band count, band contrast, and any other scalar columns.
2. Pattern images map to the correct scan pixels.
3. The orientation maps produce IPF-X, IPF-Y, and IPF-Z plots consistent with
   the HKL software exports.
4. FCC phase information and HKL indexing are handled end to end.
5. Band-width calculation runs on the exported 8-bit patterns.
6. The computed band-width fields are exported into `.h5` and `.oh5`.
7. The `.ang` export contains the expected Euler angles, x/y positions, phase,
   IQ/CI/fit-like fields, and PRIAS-style derived fields.
8. HKL/Oxford pattern-center definitions are converted internally to the
   detector convention required by the code and to any TSL/ANG-compatible output
   convention. The workflow should warn clearly if metadata is missing, but it
   should not require hand editing for a normal, well-described fixture.

### 7. Size Limits

Please keep the compressed folder preferably below 20 MB and absolutely below
50 MB unless we agree otherwise. To control size:

1. Use 4x4 binning.
2. Use 8-bit PNG patterns.
3. Keep the scan area small.
4. Avoid including raw vendor project folders, temporary files, screenshots of
   the whole application, or duplicate exports.

### 8. Final Checklist Before Sending

Before sending the data, please confirm:

1. The `.ctf` opens in HKL/Oxford software.
2. The pattern folder contains exactly one image per scan point.
3. The first few pattern filenames match the intended row-major or x/y naming
   convention.
4. The scan contains 3 to 4 visible grains.
5. IPF-X, IPF-Y, and IPF-Z reference images were exported.
6. `acquisition_notes.yml` includes the pattern center and detector geometry.
7. All pattern images are 8-bit.
8. The folder is small enough for a repository fixture.

Please do not crop or edit the `.ctf` manually after export. If you need to
reduce the dataset, do it in the acquisition/export software so the grid,
pattern folder, and metadata remain consistent.

## Repository Acceptance Criteria

When the fixture is received, add or update tests so `pytest -q` verifies the
CTF fixture can be read and the band-width workflow can produce `.h5`, `.oh5`,
and `.ang` outputs. The tests should compare parsed dimensions, scalar maps,
pattern mapping, output file existence, output dataset shapes, and a small set
of representative Euler/PC-derived values.
