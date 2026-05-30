# HKL CTF FCC Small Fixture

This folder is reserved for the small HKL/Oxford FCC CTF fixture described in
[`docs/hkl_ctf_test_data_request.md`](../../docs/hkl_ctf_test_data_request.md).

Expected layout:

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

Keep this fixture compact enough for normal Git checkout. Use 4x4 pattern
binning, 8-bit grayscale pattern images, and a small scan region containing 3 to
4 grains.
