# Kikuchi Band Analyzer

Kikuchi Band Analyzer is a production-oriented toolkit for measuring Kikuchi band
widths, comparing EBSD scans, and exporting enriched EBSD data products. The
current migration keeps legacy workflows available while moving the main
capabilities into importable, testable package modules.

The documentation is organized for two audiences:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Start Using The Tools
:link: user/index
:link-type: doc

Install the package, run debug mode, configure normal scan processing, and export
TSL `.oh5`/`.h5`, HKL `.ctf` plus pattern-folder data, `.ang`, and `.h5` outputs.
:::

:::{grid-item-card} Understand The Algorithms
:link: theory/index
:link-type: doc

Review the geometry, band-width model, edge localization, CTF scan mapping, and
quality checks used by the pipeline.
:::

:::{grid-item-card} Follow Tutorials
:link: tutorials/index
:link-type: doc

Work through single-pattern debug analysis and end-to-end scan processing using
the same structure as the included notebooks.
:::

:::{grid-item-card} Extend The Code
:link: dev/index
:link-type: doc

Use the API reference, testing strategy, documentation standards, and migration
notes when adding new readers, exporters, phases, or algorithms.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

user/index
workflows/index
tutorials/index
theory/index
reference/index
dev/index
```
