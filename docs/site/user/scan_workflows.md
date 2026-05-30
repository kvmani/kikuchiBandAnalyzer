# Scan Workflows

The scan workflow has five stages:

```{mermaid}
flowchart LR
  A["Input discovery"] --> B["Indexing map normalization"]
  B --> C["Pattern acquisition"]
  C --> D["Band-width measurement"]
  D --> E["ANG/HDF5/OH5 export"]
```

## Input Discovery

The reader factory identifies whether the input is TSL HDF5/OH5, HKL CTF plus
patterns, or another supported scan representation. Unsupported inputs should
fail before any long-running computation starts.

## Indexing Map Normalization

Euler angles, coordinates, phase information, and scan shape are converted into
a common model. This allows the detector and exporters to be format-independent.

## Pattern Acquisition

TSL files read patterns from HDF5 datasets. CTF workflows read images from a
folder and validate that the pattern count matches the scan grid.

## Band-Width Measurement

Each pixel pattern is processed with the configured HKL set. The detector returns
widths, edges, profile metadata, and warnings. Batch processing records partial
results and cancellation state explicitly.

## Export

The exporter writes `.ang`, `.h5`, and `.oh5` products. Output files include the
measured fields, scan coordinates, phase metadata, and provenance.
