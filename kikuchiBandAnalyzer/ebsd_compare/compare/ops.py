"""Numeric comparison operations for EBSD maps and patterns."""

from __future__ import annotations

import numpy as np


def delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the difference map A - B.

    Parameters:
        a: First array.
        b: Second array.

    Returns:
        Difference array.
    """

    return a - b


def abs_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the absolute difference map |A - B|.

    Parameters:
        a: First array.
        b: Second array.

    Returns:
        Absolute difference array.
    """

    return np.abs(a - b)


def ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute the ratio map A / B with safe division.

    Parameters:
        a: Numerator array.
        b: Denominator array.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Ratio array with invalid divisions masked as NaN.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b + eps)
    result[~np.isfinite(result)] = np.nan
    return result
