# Quickstart

## Single Pattern Debug Run

Debug mode is the smallest reproducible workflow. It measures one pattern and can
show publication-style intermediate plots.

```powershell
python -m kikuchiBandAnalyzer.band_width.detector_cli `
  --debug `
  --interactive `
  --json-output outputs/debug_single_pattern.json
```

The JSON output records the selected line, edge locations, band width, and
diagnostic metadata. Interactive figures show the image, the sampled intensity
profile, the detected shoulders, and the measured width.

## TSL Scan Processing

Create a YAML configuration that points to an `.oh5` or `.h5` file and then run:

```powershell
kikuchi-band-width --config configs/tsl_band_width_example.yml
```

## HKL CTF Processing

HKL systems provide indexing data in a text `.ctf` file and store patterns in a
separate folder. The configured workflow joins these sources by scan coordinate
or deterministic raster order.

```powershell
kikuchi-band-width --config configs/ctf_ni_band_width_example.yml
```

The current CTF implementation targets FCC Ni as the production reference phase.
The data model is phase-aware so additional phases can be added without changing
the scan orchestration contract.
