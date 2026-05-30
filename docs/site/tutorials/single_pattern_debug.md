# Tutorial: Single Pattern Debug Analysis

This tutorial walks through the smallest meaningful analysis.

## Goal

Measure one Kikuchi band width and inspect every intermediate result.

## Steps

1. Activate the development environment.
2. Run the detector in debug mode.
3. Inspect the interactive figure.
4. Read the exported JSON.
5. Promote the case into a regression test only after the expected width is
   deterministic.

```powershell
python -m kikuchiBandAnalyzer.band_width.detector_cli `
  --debug `
  --interactive `
  --json-output outputs/tutorial_single_pattern.json
```

## What To Inspect

The central line should pass through the band center. The sampled profile should
show two shoulders. The exported width should equal the distance between the
selected shoulder locations in the profile coordinate system.

## Failure Interpretation

A flat profile usually indicates poor contrast, an incorrect line, or the wrong
pattern region. Out-of-bounds line metadata usually indicates a mismatch between
the indexing geometry and the image shape.
