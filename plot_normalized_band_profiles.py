#!/usr/bin/env python3
"""
plot_profiles_config.py – Plot several two-column (x,y) CSV files on one graph,
driven entirely by a CONFIG dict, and export the scaled data to Excel.

Behavior mirrors the CLI-based version:
- Per-series x-scaling
- Cycling colors/styles/widths
- Auto-detect single header row
- Save figure to 'output' if provided, else show()
- Export Excel with paired columns [Scaled X (label), Y (label)]

Usage:
1) Edit the CONFIG dict at the bottom of this file and run:
   python plot_profiles_config.py

2) Or import plot_profiles_from_config(CONFIG) from another module.
"""

from __future__ import annotations
import os
import itertools
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _has_header(path: str) -> bool:
    """Return True if the first non-empty line appears to be a header (non-digit start)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return not line[0].isdigit()
    return False


def load_profile(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column profile (x, y) from CSV."""
    skip = 1 if _has_header(path) else 0
    x, y = np.loadtxt(path, delimiter=",", skiprows=skip, unpack=True)
    return x, y


def _cycle(seq):
    """Yield elements of seq in an endless cycle."""
    return itertools.cycle(seq)


def _validate_and_prepare_config(cfg: dict) -> dict:
    """Validate inputs and fill defaults without altering original behavior."""
    if "files" not in cfg or not cfg["files"]:
        raise ValueError("CONFIG['files'] must be a non-empty list of CSV paths.")

    files: List[str] = cfg["files"]

    # x_scales: if provided, must match number of files; else default 1.0 per file
    x_scales: Optional[List[float]] = cfg.get("x_scales")
    if x_scales is not None:
        if len(x_scales) != len(files):
            raise ValueError("Length of CONFIG['x_scales'] must match CONFIG['files'].")
    else:
        x_scales = [1.0] * len(files)

    font_size: int = int(cfg.get("font_size", 12))

    default_colors = ["C0", "C1", "C2", "C3", "C4"]
    default_styles = ["solid", "dashed", "dotted", "dashdot", (0, (1, 1))]
    default_widths = [1.5] * len(default_colors)

    colors = cfg.get("colors") or default_colors
    styles = cfg.get("styles") or default_styles
    widths = cfg.get("widths") or default_widths

    # output: None/"" -> show(); else savefig to this path
    out_path = cfg.get("output")
    if isinstance(out_path, str) and not out_path.strip():
        out_path = None

    return {
        "files": files,
        "x_scales": x_scales,
        "font_size": font_size,
        "colors": colors,
        "styles": styles,
        "widths": widths,
        "output": out_path,
    }


def plot_profiles_from_config(CONFIG: dict) -> str:
    """
    Execute plotting and Excel export as per CONFIG.
    Returns the Excel path written (for convenience).
    """
    cfg = _validate_and_prepare_config(CONFIG)

    files = cfg["files"]
    x_scales = cfg["x_scales"]
    font_size = cfg["font_size"]
    colors = cfg["colors"]
    styles = cfg["styles"]
    widths = cfg["widths"]
    out_path = cfg["output"]

    # Matplotlib font sizing baseline
    plt.rcParams.update({"font.size": font_size})

    # Cycle styling if shorter than the number of series
    col_iter = _cycle(colors)
    sty_iter = _cycle(styles)
    wid_iter = _cycle(widths)

    fig, ax = plt.subplots()

    # Collect columns for Excel export; each series contributes 2 columns
    series_list: List[pd.Series] = []

    for file, scale in zip(files, x_scales):
        x, y = load_profile(file)
        x_scaled = x * scale
        label = os.path.basename(file)
        if label.lower().endswith(".csv"):
            label = label[:-4]

        ax.plot(
            x_scaled,
            y,
            color=next(col_iter),
            linestyle=next(sty_iter),
            linewidth=next(wid_iter),
            label=label,
        )

        series_list.append(pd.Series(x_scaled, name=f"Scaled X ({label})"))
        series_list.append(pd.Series(y,        name=f"Y ({label})"))

    # Axes labels and ticks
    ax.set_xlabel("Distance in Pixels", fontsize=font_size + 8)
    ax.set_ylabel("Normalized Profile", fontsize=font_size + 8)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=font_size + 4)
    ax.tick_params(axis="both", which="minor", labelsize=font_size + 2)
    ax.legend(fontsize=font_size + 2)
    fig.tight_layout()

    # Save/show and decide Excel path
    if out_path:
        # Ensure folder exists
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        excel_path = os.path.splitext(out_path)[0] + "_export.xlsx"
    else:
        plt.show()
        excel_path = "profiles_export.xlsx"

    # Concatenate into a single DataFrame (handles differing lengths)
    df = pd.concat(series_list, axis=1)
    df.index.name = "Index"
    df.to_excel(excel_path, index=True)
    print(f"[INFO] Exported scaled profile data to: {excel_path}")

    return excel_path


# ------------------------------
# Example CONFIG and entry point
# ------------------------------
if __name__ == "__main__":
    CONFIG = {
        "files": [
            "ground_truth.csv",
            "noisy.csv",
            "ml_processed.csv",
        ],
        "x_scales": [0.25, 0.25, 0.25],
        "font_size": 12,
        "colors": ["C0", "C1", "C2"],
        "styles": ["solid", "dashed", "dotted"],
        "widths": [1.5, 1.5, 2.0],
        "output": "aiprocessed_accuracy_testing_profiles.png",  # set to None or "" to show() instead
    }

    # Run with the example config
    plot_profiles_from_config(CONFIG)
