# TSL OH5/H5 Workflow

TSL `.oh5` and `.h5` files are single-container inputs. The normal-mode
configuration should point to the input file, choose phase/HKL settings, and
declare all export paths.

```powershell
kikuchi-band-width --config configs/tsl_band_width_example.yml
```

Before running a long scan, collect metadata only and verify:

- scan width and height;
- pattern height and width;
- available phase fields;
- writable output directory;
- selected HKL families.
