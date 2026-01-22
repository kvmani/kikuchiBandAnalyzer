"""Tests for GUI validation helpers."""

from __future__ import annotations

from kikuchiBandAnalyzer.ebsd_compare.gui.validation import (
    validate_int_in_range,
    validate_speed_ms,
)


def test_validate_int_in_range_valid() -> None:
    """Validate a correct integer input."""

    result = validate_int_in_range("5", 0, 10, "X")
    assert result.is_valid()
    assert result.value == 5


def test_validate_int_in_range_invalid() -> None:
    """Validate an out-of-range integer input."""

    result = validate_int_in_range("999", 0, 10, "X")
    assert not result.is_valid()
    assert result.error


def test_validate_speed_ms() -> None:
    """Validate the auto-scan speed helper."""

    assert validate_speed_ms(100, 25, 200).is_valid()
    assert not validate_speed_ms(5, 25, 200).is_valid()
