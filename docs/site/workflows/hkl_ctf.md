# HKL CTF Workflow

HKL CTF processing combines text indexing data with a folder of pattern images.

```powershell
kikuchi-band-width --config configs/ctf_ni_band_width_example.yml
```

For FCC Ni validation, compare one sample represented as `.ang`, `.h5`, and
`.ctf` plus pattern folder. The expected result is equivalent scan geometry and
consistent band-width fields across output formats within the configured numeric
tolerance.

Store only small cropped fixtures in the repository. Large microscope exports
should remain outside version control and be referenced from local configuration
files.

For the FCC HKL/Oxford fixture used by tests and tutorials, follow the request
template in `docs/hkl_ctf_test_data_request.md`. The fixture should contain a
small `.ctf` file, an 8-bit 4x4-binned pattern folder, HKL IPF-X/IPF-Y/IPF-Z
reference images, acquisition notes with pattern-center geometry, and compact
expected `.h5`, `.oh5`, and `.ang` outputs when size allows.
