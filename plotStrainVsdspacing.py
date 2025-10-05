"""
Cubic Lattice d(hkl) vs. Strain (Publication-Ready Plot)
--------------------------------------------------------
Intent:
    - Given a stress-free cubic lattice parameter a0 (e.g., 3.4 Å), vary the lattice parameter by
      engineering tensile strain from 0% to 10% in 1% increments.
    - Compute interplanar spacings d_hkl(a) for planes (100), (110), (111): d_hkl = a / sqrt(h^2+k^2+l^2).
    - Plot Strain (%) vs. d_hkl (Å) with publication-ready quality (fonts, labels, legend, markers).

Structure:
    - LatticeStudy: Encapsulates data and computations for lattice parameter and d_hkl vs. strain.
    - Plotter: Handles figure creation and styling without specifying colors (per constraints).

Design decisions:
    - Strain is engineering strain: eps = (a - a0)/a0; a = a0 * (1 + eps)
    - Include 0% strain as requested.
    - Default input via an in-code dictionary for early development. YAML support can be added on request.

Usage:
    - Run this file directly. Outputs a PNG figure 'd_hkl_vs_strain.png' in the working directory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


HKL = Tuple[int, int, int]


@dataclass
class LatticeStudy:
    """Compute lattice parameter changes and interplanar spacings for cubic crystals.

    Attributes
    ----------
    a0 : float
        Stress-free lattice parameter (Å).
    strain_steps : Iterable[float]
        Engineering strain values (fractional, e.g., 0.00, 0.01, ..., 0.10).
    planes : List[HKL]
        List of Miller indices to evaluate.
    """

    a0: float
    strain_steps: Iterable[float]
    planes: List[HKL]

    def __post_init__(self) -> None:
        self._validate_inputs()
        self.strain_steps = np.array(list(self.strain_steps), dtype=float)

    def _validate_inputs(self) -> None:
        if not isinstance(self.a0, (int, float)) or self.a0 <= 0:
            raise ValueError("a0 must be a positive number (Å).")
        if not isinstance(self.planes, (list, tuple)) or len(self.planes) == 0:
            raise ValueError("planes must be a non-empty list of (h,k,l) tuples.")
        for hkl in self.planes:
            if (
                not isinstance(hkl, (list, tuple))
                or len(hkl) != 3
                or not all(isinstance(v, int) for v in hkl)
            ):
                raise ValueError(f"Invalid Miller index: {hkl}. Use integer triplets like (1,1,1).")
            if all(v == 0 for v in hkl):
                raise ValueError("(0,0,0) is not a valid Miller index.")

    # --------- Core computations ---------
    def a_values(self) -> np.ndarray:
        """Return the lattice parameter values a = a0 * (1 + eps) for each strain step."""
        return self.a0 * (1.0 + self.strain_steps)

    @staticmethod
    def _d_for_plane_given_a(a: np.ndarray, hkl: HKL) -> np.ndarray:
        h, k, l = hkl
        denom = math.sqrt(h * h + k * k + l * l)
        return a / denom

    def d_for_plane(self, hkl: HKL) -> np.ndarray:
        """Compute d_hkl(a) for all a-values for a given (hkl)."""
        a_vals = self.a_values()
        return self._d_for_plane_given_a(a_vals, hkl)

    def compute_all(self) -> Dict[HKL, np.ndarray]:
        """Compute d_hkl arrays for each plane.

        Returns
        -------
        Dict[HKL, np.ndarray]
            Mapping from (h,k,l) -> d_hkl(a) over all strain steps.
        """
        results: Dict[HKL, np.ndarray] = {}
        a_vals = self.a_values()  # compute once
        for hkl in self.planes:
            results[hkl] = self._d_for_plane_given_a(a_vals, hkl)
        return results


class Plotter:
    """Create publication-ready matplotlib plots (without specifying colors)."""

    def __init__(self, font_size: int = 13):
        # Configure global style (avoid color specifications to respect constraints)
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.labelsize": font_size + 1,
                "axes.titlesize": font_size + 2,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
                "font.size": font_size,
            }
        )

    @staticmethod
    def _hkl_to_label(hkl: HKL) -> str:
        h, k, l = hkl
        return fr"$({h}{k}{l})$"

    def plot_d_vs_strain(
        self,
        strain_steps: np.ndarray,
        d_map: Dict[HKL, np.ndarray],
        a0: float,
        output_path: str = "d_hkl_vs_strain.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        x_percent = 100.0 * strain_steps

        # Choose different markers for up to a few curves (no color specification)
        markers = ["o", "s", "^", "D", "v", ">", "<"]

        for i, (hkl, d_vals) in enumerate(d_map.items()):
            marker = markers[i % len(markers)]
            ax.plot(
                x_percent,
                d_vals,
                linewidth=2.0,
                marker=marker,
                markersize=5.5,
                markeredgewidth=0.9,
                label=self._hkl_to_label(hkl),
            )

        # Labels & title
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel(r"Interplanar spacing $d_{hkl}$ (Å)")
        ax.set_title("Cubic crystal: $d_{hkl}$ vs. tensile strain")

        # Ticks & grid
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle=":", linewidth=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.6)

        # Legend
        leg = ax.legend(frameon=True, fancybox=True, borderpad=0.6)
        leg.set_title("Planes")

        # Helpful annotation: linear proportionality
        ax.annotate(
            r"$d_{hkl} = \dfrac{a}{\sqrt{h^2+k^2+l^2}}\;\propto\;a$",
            xy=(0.55, 0.15),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=1.0),
        )

        # Tight layout and save
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")


# --- Additional analysis classes appended ---
class KikuchiAnalysis:
    """Compute Kikuchi band widths from d-spacing maps with a scaling factor.

    Band width definition: BW_hkl = scaling_factor / d_hkl.
    With scaling_factor = 1, units are 1/Å.
    """

    def __init__(self, scaling_factor: float = 1.0) -> None:
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive.")
        self.scaling_factor = scaling_factor

    def compute(self, d_map: Dict[HKL, np.ndarray]) -> Dict[HKL, np.ndarray]:
        bw_map: Dict[HKL, np.ndarray] = {}
        for hkl, d_vals in d_map.items():
            d_vals = np.asarray(d_vals, dtype=float)
            if np.any(d_vals <= 0):
                raise ValueError("All d-spacings must be positive to compute bandwidth.")
            bw_map[hkl] = self.scaling_factor / d_vals
        return bw_map


class PlotterDual:
    """Two-panel plotter: (left) d_hkl vs strain, (right) Kikuchi band width vs strain (side-by-side)."""

    def __init__(self, font_size: int = 13):
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.labelsize": font_size + 1,
                "axes.titlesize": font_size + 2,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
                "font.size": font_size,
            }
        )

    @staticmethod
    def _hkl_to_label(hkl: HKL) -> str:
        h, k, l = hkl
        return f"({h}{k}{l})"

    def plot(
        self,
        strain_steps: np.ndarray,
        d_map: Dict[HKL, np.ndarray],
        bw_map: Dict[HKL, np.ndarray],
        output_path: str = "d_hkl_and_kikuchi_bw_vs_strain.png",
    ) -> None:
        # Side-by-side panels, shared x-axis
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.0, 4.8), sharex=True)

        x_percent = 100.0 * np.asarray(strain_steps, dtype=float)
        markers = ["o", "s", "^", "D", "v", ">", "<"]

        # Left: d_hkl vs strain
        for i, (hkl, d_vals) in enumerate(d_map.items()):
            ax1.plot(
                x_percent,
                d_vals,
                linewidth=2.0,
                marker=markers[i % len(markers)],
                markersize=5.5,
                markeredgewidth=0.9,
                label=self._hkl_to_label(hkl),
            )
        ax1.set_xlabel("Strain (%)")
        ax1.set_ylabel("Interplanar spacing d_hkl (Å)")
        ax1.set_title("d_hkl vs tensile strain")
        ax1.minorticks_on()
        ax1.grid(True, which="major", linestyle=":", linewidth=0.8)
        ax1.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.6)
        ax1.legend(frameon=True, fancybox=True, borderpad=0.6, title="Planes")
        # ax1.annotate(
        #     "d_hkl = a / sqrt(h^2+k^2+l^2)  →  ∝ a",
        #     xy=(0.55, 0.15), xycoords="axes fraction",
        #     ha="center", va="center", fontsize=12,
        #     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=1.0),
        # )

        # Right: band width vs strain
        for i, (hkl, bw_vals) in enumerate(bw_map.items()):
            ax2.plot(
                x_percent,
                bw_vals,
                linewidth=2.0,
                marker=markers[i % len(markers)],
                markersize=5.5,
                markeredgewidth=0.9,
                label=self._hkl_to_label(hkl),
            )
        ax2.set_xlabel("Strain (%)")
        ax2.set_ylabel("Kikuchi band width (1/d_hkl) (1/Å)")
        ax2.set_title("Kikuchi band width vs tensile strain")
        ax2.minorticks_on()
        ax2.grid(True, which="major", linestyle=":", linewidth=0.8)
        ax2.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.6)
        ax2.legend(frameon=True, fancybox=True, borderpad=0.6, title="Planes")
        # ax2.annotate(
        #     "Band width = scaling / d_hkl  →  ∝ 1/a  (scaling=1)",
        #     xy=(0.55, 0.85), xycoords="axes fraction",
        #     ha="center", va="center", fontsize=12,
        #     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=1.0),
        # )

        # Overall figure title
        fig.suptitle("Cubic crystal: d_hkl and Kikuchi band width vs tensile strain", y=0.98)

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.show()
        print(f"Saved figure to: {output_path}")


# ------------------- Example run -------------------
if __name__ == "__main__":
    # Default input dictionary (can be replaced by YAML loading if requested)
    config = {
        "a0": 3.4,  # Å
        # Include 0% to 10% in 1% steps
        "strain_steps": [i / 100.0 for i in range(0, 5)],
        "planes": [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        "scaling_factor": 1.0,  # band width scaling; units become (scaling)/Å
    }

    study = LatticeStudy(
        a0=config["a0"],
        strain_steps=config["strain_steps"],
        planes=config["planes"],
    )

    # Compute results
    d_map = study.compute_all()

    # Band width from d-map
    bw_map = KikuchiAnalysis(scaling_factor=config["scaling_factor"]).compute(d_map)

    # Optional: quick numeric sanity checks at 0% strain
    for hkl in config["planes"]:
        d0 = d_map[hkl][0]
        denom = math.sqrt(sum(v * v for v in hkl))
        assert abs(d0 - config["a0"] / denom) < 1e-12, "d_hkl at 0% should equal a0/sqrt(h^2+k^2+l^2)"
        assert abs(bw_map[hkl][0] - (config["scaling_factor"] / d0)) < 1e-12, "band width must be inverse of d"

    # Plot two panels
    plotter = PlotterDual(font_size=13)
    plotter.plot(
        strain_steps=np.array(config["strain_steps"], dtype=float),
        d_map=d_map,
        bw_map=bw_map,
        output_path="d_hkl_and_kikuchi_bw_vs_strain.png",
    )
