"""Factory helpers for vendor-neutral EBSD scan readers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from kikuchiBandAnalyzer.ebsd_compare.model import ScanDataset
from kikuchiBandAnalyzer.ebsd_compare.readers.ctf_reader import CtfPatternScanFileReader
from kikuchiBandAnalyzer.ebsd_compare.readers.oh5_reader import OH5ScanFileReader


def open_scan_dataset(
    path: str | Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    role: Optional[str] = None,
    field_aliases: Optional[dict[str, list[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> ScanDataset:
    """Open an EBSD scan dataset from a supported vendor format.

    Parameters:
        path: Path to the scan file.
        config: Optional comparison configuration.
        role: Optional role such as ``scan_a`` or ``scan_b`` for resolving
            role-specific CTF pattern directories.
        field_aliases: Optional field alias mapping.
        logger: Optional logger instance.

    Returns:
        ScanDataset backed by the matching reader.
    """

    scan_path = Path(path)
    suffix = scan_path.suffix.lower()
    if suffix in {".oh5", ".h5", ".hdf5"}:
        return OH5ScanFileReader.from_path(
            scan_path,
            field_aliases=field_aliases,
            logger=logger,
        )
    if suffix == ".ctf":
        pattern_dir, pattern_template = resolve_ctf_pattern_source(scan_path, config, role)
        return CtfPatternScanFileReader.from_path(
            scan_path,
            pattern_dir=pattern_dir,
            pattern_template=pattern_template,
            field_aliases=field_aliases,
            logger=logger,
        )
    raise ValueError(f"Unsupported scan file extension '{scan_path.suffix}'.")


def resolve_ctf_pattern_source(
    ctf_path: Path,
    config: Optional[Mapping[str, Any]] = None,
    role: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    """Resolve a CTF pattern folder and optional filename template.

    Parameters:
        ctf_path: Path to the CTF file.
        config: Optional comparison configuration.
        role: Optional role such as ``scan_a`` or ``scan_b``.

    Returns:
        Tuple of pattern directory and filename template.
    """

    ctf_config = dict((config or {}).get("ctf", {}) or {})
    template = ctf_config.get("pattern_template")
    pattern_dir = None
    pattern_dirs = ctf_config.get("pattern_dirs", {})
    if role and isinstance(pattern_dirs, Mapping):
        pattern_dir = pattern_dirs.get(role)
    if pattern_dir is None:
        pattern_dir = ctf_config.get("pattern_dir")
    if pattern_dir is None:
        sibling = ctf_path.with_name(f"{ctf_path.stem}_patterns")
        if sibling.exists():
            pattern_dir = sibling
    if pattern_dir is None:
        return None, template
    resolved = Path(pattern_dir)
    if not resolved.is_absolute():
        resolved = ctf_path.parent / resolved
    return resolved, template
