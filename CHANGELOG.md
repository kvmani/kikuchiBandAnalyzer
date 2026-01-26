# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-01-26

### Added
- Derived field registry with normalized band intensity difference (`band_intensity_diff_norm`).
- HDF5 outputs for `band_intensity_ratio` and `band_intensity_diff_norm`.
- YAML `fields` list in the EBSD comparator for arbitrary scalar map selection.
- Missing-field warnings surfaced in GUI and logs.
- `sync_navigation` toggle for linked pan/zoom in the EBSD comparator.
- Versioning framework with `VERSION`, `CHANGELOG.md`, and shared metadata.

### Changed
- EBSD comparator contrast controls now re-apply limits reliably after view resets.
- CLI exports honor configured scalar field selection and missing-field handling.
