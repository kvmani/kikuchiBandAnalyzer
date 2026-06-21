"""Scientific scalar-map display settings and Matplotlib normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import colormaps, colors


@dataclass
class MapDisplaySettings:
    """Display-only settings for a scalar scientific map.

    Parameters:
        range_mode: ``auto`` percentile limits or ``manual`` limits.
        percentile_low: Automatic lower percentile.
        percentile_high: Automatic upper percentile.
        minimum: Manual minimum.
        maximum: Manual maximum.
        scale: ``linear``, ``log``, or ``symlog``.
        symmetric: Whether limits are symmetric around zero.
        linthresh: Linear threshold used by symmetric-log scaling.
        colormap: Matplotlib colormap name.
        reversed: Whether to reverse the colormap.
        invalid_color: Color used for NaN/masked values.
    """

    range_mode: str = "auto"
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    minimum: float = 0.0
    maximum: float = 1.0
    scale: str = "linear"
    symmetric: bool = False
    linthresh: float = 1.0e-3
    colormap: str = "viridis"
    reversed: bool = False
    invalid_color: str = "#808080"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None, **defaults: Any) -> MapDisplaySettings:
        """Create settings from YAML-compatible values.

        Parameters:
            values: Optional source mapping.
            defaults: Field defaults applied before mapping values.

        Returns:
            Validated settings instance.
        """

        merged = dict(defaults)
        merged.update(dict(values or {}))
        allowed = set(cls.__dataclass_fields__)
        settings = cls(**{key: value for key, value in merged.items() if key in allowed})
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate settings and raise a clear error for invalid combinations."""

        if self.range_mode not in {"auto", "manual"}:
            raise ValueError("range_mode must be 'auto' or 'manual'.")
        if not 0 <= self.percentile_low < self.percentile_high <= 100:
            raise ValueError("Automatic percentiles must satisfy 0 <= low < high <= 100.")
        if self.range_mode == "manual" and not self.minimum < self.maximum:
            raise ValueError("Manual map minimum must be less than maximum.")
        if self.scale not in {"linear", "log", "symlog"}:
            raise ValueError("scale must be 'linear', 'log', or 'symlog'.")
        if self.linthresh <= 0:
            raise ValueError("symlog linthresh must be positive.")
        if self.colormap not in colormaps:
            raise ValueError(f"Unknown Matplotlib colormap: {self.colormap!r}.")
        if not colors.is_color_like(self.invalid_color):
            raise ValueError(f"Invalid NaN/masked color: {self.invalid_color!r}.")

    def to_mapping(self) -> dict[str, Any]:
        """Return a YAML-compatible mapping.

        Returns:
            Serialized settings dictionary.
        """

        return {
            "range_mode": self.range_mode,
            "percentile_low": float(self.percentile_low),
            "percentile_high": float(self.percentile_high),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
            "scale": self.scale,
            "symmetric": bool(self.symmetric),
            "linthresh": float(self.linthresh),
            "colormap": self.colormap,
            "reversed": bool(self.reversed),
            "invalid_color": self.invalid_color,
        }

    def render_parameters(self, data: np.ndarray) -> tuple[colors.Normalize, colors.Colormap]:
        """Build normalization and colormap without modifying source data.

        Parameters:
            data: Scalar map array.

        Returns:
            Matplotlib normalization and copied colormap.
        """

        self.validate()
        array = np.asarray(data, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        elif self.range_mode == "manual":
            vmin, vmax = float(self.minimum), float(self.maximum)
        elif self.scale == "log":
            positive = finite[finite > 0]
            if positive.size == 0:
                raise ValueError("Log map scale requires at least one positive finite value.")
            vmin, vmax = np.percentile(
                positive, [self.percentile_low, self.percentile_high]
            ).astype(float)
        else:
            vmin, vmax = np.percentile(
                finite, [self.percentile_low, self.percentile_high]
            ).astype(float)
        if self.symmetric:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        if vmin == vmax:
            vmax = vmin + max(1.0, abs(vmin) * 1.0e-6)
        if self.scale == "log":
            if vmin <= 0 or vmax <= 0:
                raise ValueError("Log map limits must both be positive.")
            norm: colors.Normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
        elif self.scale == "symlog":
            norm = colors.SymLogNorm(
                linthresh=float(self.linthresh), vmin=vmin, vmax=vmax
            )
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_name = self.colormap + ("_r" if self.reversed and not self.colormap.endswith("_r") else "")
        cmap = colormaps[cmap_name].copy()
        cmap.set_bad(self.invalid_color)
        return norm, cmap
