"""Tests for EBSD compare field selection helpers."""

from __future__ import annotations

from kikuchiBandAnalyzer.ebsd_compare.field_selection import (
    resolve_pattern_fields,
    resolve_scalar_fields,
    resolve_sync_navigation,
)
from kikuchiBandAnalyzer.ebsd_compare.model import FieldCatalog, FieldRef


def _make_catalog(scalars: list[str], patterns: list[str]) -> FieldCatalog:
    """Build a FieldCatalog with placeholder refs.

    Parameters:
        scalars: Scalar field names.
        patterns: Pattern field names.

    Returns:
        FieldCatalog with the provided fields.
    """

    scalar_refs = {
        name: FieldRef(name=name, path=f"/{name}", kind="scalar", shape=(2, 2), dtype="float32")
        for name in scalars
    }
    pattern_refs = {
        name: FieldRef(name=name, path=f"/{name}", kind="pattern", shape=(2, 2, 2), dtype="float32")
        for name in patterns
    }
    return FieldCatalog(scalars=scalar_refs, patterns=pattern_refs)


def test_resolve_scalar_fields_prefers_fields_list() -> None:
    """Ensure the new fields list overrides legacy compare_fields entries."""

    config = {"fields": ["IQ"], "compare_fields": {"scalars": ["CI"]}}
    catalog_a = _make_catalog(["IQ", "CI"], [])
    catalog_b = _make_catalog(["IQ", "CI"], [])
    fields, warnings = resolve_scalar_fields(config, catalog_a, catalog_b)
    assert fields == ["IQ"]
    assert not warnings


def test_resolve_scalar_fields_warns_and_filters_missing() -> None:
    """Warn and skip missing scalar fields."""

    config = {"fields": ["IQ", "CI", "Missing"]}
    catalog_a = _make_catalog(["IQ", "CI"], [])
    catalog_b = _make_catalog(["IQ"], [])
    fields, warnings = resolve_scalar_fields(config, catalog_a, catalog_b)
    assert fields == ["IQ"]
    assert any("CI" in message for message in warnings)
    assert any("Missing" in message for message in warnings)


def test_resolve_pattern_fields_falls_back_to_common() -> None:
    """Fall back to common patterns when configured ones are missing."""

    config = {"compare_fields": {"patterns": ["PatternA"]}}
    catalog_a = _make_catalog([], ["PatternB"])
    catalog_b = _make_catalog([], ["PatternB"])
    fields, warnings = resolve_pattern_fields(config, catalog_a, catalog_b)
    assert fields == ["PatternB"]
    assert warnings


def test_resolve_sync_navigation_defaults_true() -> None:
    """Return true when sync_navigation is not set."""

    assert resolve_sync_navigation({}) is True
    assert resolve_sync_navigation({"sync_navigation": False}) is False
