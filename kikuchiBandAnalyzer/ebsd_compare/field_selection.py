"""Field selection helpers for EBSD compare configuration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from kikuchiBandAnalyzer.ebsd_compare.model import FieldCatalog


def extract_scalar_fields(config: Dict[str, Any]) -> List[str]:
    """Extract configured scalar field names from a config dictionary.

    Parameters:
        config: Configuration dictionary for EBSD compare.

    Returns:
        List of scalar field names (may be empty).
    """

    fields = config.get("fields")
    if isinstance(fields, (list, tuple)):
        return [str(field) for field in fields if str(field).strip()]
    compare_fields = config.get("compare_fields", {})
    scalars = compare_fields.get("scalars", [])
    if isinstance(scalars, (list, tuple)):
        return [str(field) for field in scalars if str(field).strip()]
    return []


def extract_pattern_fields(config: Dict[str, Any]) -> List[str]:
    """Extract configured pattern field names from a config dictionary.

    Parameters:
        config: Configuration dictionary for EBSD compare.

    Returns:
        List of pattern field names (may be empty).
    """

    compare_fields = config.get("compare_fields", {})
    patterns = compare_fields.get("patterns", [])
    if isinstance(patterns, (list, tuple)):
        return [str(field) for field in patterns if str(field).strip()]
    return []


def resolve_scalar_fields(
    config: Dict[str, Any],
    catalog_a: FieldCatalog,
    catalog_b: FieldCatalog,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str]]:
    """Resolve scalar field names that exist in both scans.

    Parameters:
        config: Configuration dictionary for EBSD compare.
        catalog_a: Field catalog for scan A.
        catalog_b: Field catalog for scan B.
        logger: Optional logger for warning messages.

    Returns:
        Tuple of (resolved fields, warning messages).
    """

    warnings: List[str] = []
    desired = extract_scalar_fields(config)
    common = sorted(set(catalog_a.scalars) & set(catalog_b.scalars))
    if not desired:
        return common, warnings
    resolved: List[str] = []
    seen = set()
    for field in desired:
        if field in seen:
            continue
        seen.add(field)
        missing_a = field not in catalog_a.scalars
        missing_b = field not in catalog_b.scalars
        if missing_a:
            warnings.append(
                f"Field '{field}' not found in scan A; skipping."
            )
        if missing_b:
            warnings.append(
                f"Field '{field}' not found in scan B; skipping."
            )
        if not missing_a and not missing_b:
            resolved.append(field)
    if not resolved and common:
        warnings.append(
            "Configured scalar fields missing; using all common scalar fields."
        )
        resolved = common
    if logger:
        for message in warnings:
            logger.warning(message)
    return resolved, warnings


def resolve_pattern_fields(
    config: Dict[str, Any],
    catalog_a: FieldCatalog,
    catalog_b: FieldCatalog,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str]]:
    """Resolve pattern field names that exist in both scans.

    Parameters:
        config: Configuration dictionary for EBSD compare.
        catalog_a: Field catalog for scan A.
        catalog_b: Field catalog for scan B.
        logger: Optional logger for warning messages.

    Returns:
        Tuple of (resolved fields, warning messages).
    """

    warnings: List[str] = []
    desired = extract_pattern_fields(config)
    common = sorted(set(catalog_a.patterns) & set(catalog_b.patterns))
    if not desired:
        return common, warnings
    resolved: List[str] = []
    seen = set()
    for field in desired:
        if field in seen:
            continue
        seen.add(field)
        missing_a = field not in catalog_a.patterns
        missing_b = field not in catalog_b.patterns
        if missing_a:
            warnings.append(
                f"Pattern field '{field}' not found in scan A; skipping."
            )
        if missing_b:
            warnings.append(
                f"Pattern field '{field}' not found in scan B; skipping."
            )
        if not missing_a and not missing_b:
            resolved.append(field)
    if not resolved and common:
        warnings.append(
            "Configured pattern fields missing; using all common patterns."
        )
        resolved = common
    if logger:
        for message in warnings:
            logger.warning(message)
    return resolved, warnings


def resolve_sync_navigation(config: Dict[str, Any]) -> bool:
    """Resolve the sync_navigation flag from config.

    Parameters:
        config: Configuration dictionary for EBSD compare.

    Returns:
        True if navigation sync is enabled, otherwise False.
    """

    return bool(config.get("sync_navigation", True))
