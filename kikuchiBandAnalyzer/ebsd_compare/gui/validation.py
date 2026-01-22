"""Validation helpers for the EBSD compare GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ValidationResult(Generic[T]):
    """Result container for validation routines.

    Parameters:
        value: Parsed value when valid.
        error: Error message when invalid.
    """

    value: Optional[T]
    error: Optional[str]

    def is_valid(self) -> bool:
        """Return True when the validation result is valid.

        Returns:
            True if the result includes a value and no error.
        """

        return self.value is not None and self.error is None


def validate_int_in_range(
    raw_value: str, min_value: int, max_value: int, label: str
) -> ValidationResult[int]:
    """Validate an integer string within a range.

    Parameters:
        raw_value: Raw input string to parse.
        min_value: Minimum allowed integer value.
        max_value: Maximum allowed integer value.
        label: Label used in error messages.

    Returns:
        ValidationResult with parsed integer or error message.
    """

    stripped = raw_value.strip()
    if not stripped:
        return ValidationResult(value=None, error=f'{label} is required.')
    try:
        value = int(stripped)
    except ValueError:
        return ValidationResult(
            value=None, error=f'{label} must be an integer in [{min_value}, {max_value}].'
        )
    if value < min_value or value > max_value:
        return ValidationResult(
            value=None, error=f'{label} must be in [{min_value}, {max_value}].'
        )
    return ValidationResult(value=value, error=None)


def validate_speed_ms(
    delay_ms: int, min_ms: int, max_ms: int
) -> ValidationResult[int]:
    """Validate an auto-scan delay value in milliseconds.

    Parameters:
        delay_ms: Delay in milliseconds.
        min_ms: Minimum allowed delay.
        max_ms: Maximum allowed delay.

    Returns:
        ValidationResult with validated delay or error message.
    """

    if delay_ms < min_ms or delay_ms > max_ms:
        return ValidationResult(
            value=None,
            error=f"Speed must be between {min_ms} and {max_ms} ms.",
        )
    return ValidationResult(value=delay_ms, error=None)
