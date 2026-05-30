# Migration Goals

The migration goal is not only to move files into a package. The goal is a
modular, production-grade scientific codebase with preserved legacy behavior.

Required properties:

- compatibility wrappers keep legacy user-visible workflows available;
- package modules expose testable APIs;
- normal mode is config driven and non-interactive;
- debug mode is deterministic and visually explanatory;
- TSL `.oh5`/`.h5` and HKL `.ctf` plus pattern-folder workflows share a common
  scan model;
- FCC Ni CTF processing is the first supported HKL production path;
- exports include `.ang`, `.h5`, and `.oh5` products with provenance;
- documentation, tests, and examples change together.
