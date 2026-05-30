# Single Pattern Debug Mode

Single-pattern debug mode is the primary developer and user diagnostic workflow.
It answers the question: can one Kikuchi pattern be indexed, measured, explained,
and exported before a full scan is attempted?

Expected behavior:

- load a pattern from a file or generate a deterministic synthetic pattern;
- simulate or read candidate Kikuchi lines for the selected FCC Ni phase;
- sample intensity profiles perpendicular to each line;
- locate the band shoulders and compute width in pixel units;
- show interactive diagnostic figures when requested;
- export JSON and optional figures with complete provenance.

The diagnostic plot should make the measured geometry clear enough for a report:
central line, profile direction, edge markers, selected width, uncertainty or
quality flags when available, and readable axis labels.

```powershell
python -m kikuchiBandAnalyzer.band_width.detector_cli `
  --debug `
  --interactive `
  --json-output outputs/single_pattern_debug.json `
  --debug-plot outputs/single_pattern_debug.png
```

When the detector cannot produce a trustworthy width, it should explain whether
the issue is missing line metadata, invalid pattern dimensions, a flat profile,
ambiguous shoulders, or an out-of-bounds crop.
