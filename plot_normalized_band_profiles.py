#!/usr/bin/env python3
"""
plot_profiles.py – Plot several two-column (x,y) CSV files on one graph and export the scaled data to Excel.

Example:
python plot_profiles.py 4X4_Raw.csv 1X1_GroundTruth.csv 1X1_ML_Processed.csv \
       --x-scales "1.7,1.0,1.0" --font-size 12 \
       --colors "C0,C1,C2" --styles "solid,dashed,dotted" \
       --widths "1.5,1.5,2" --output profiles.png
"""

import argparse
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _has_header(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return not line[0].isdigit()
    return False


def load_profile(path: str):
    skip = 1 if _has_header(path) else 0
    x, y = np.loadtxt(path, delimiter=",", skiprows=skip, unpack=True)
    return x, y


def infinite_cycle(seq):
    return itertools.cycle(seq)


def plot_profiles(
    files,
    x_scales=None,
    font_size=12,
    colors=None,
    styles=None,
    widths=None,
    out_path=None,
):
    plt.rcParams.update({"font.size": font_size})

    default_colors = ["C0", "C1", "C2", "C3", "C4"]
    default_styles = ["solid", "dashed", "dotted", "dashdot", (0, (1, 1))]
    default_widths = [1.5] * len(default_colors)

    colors = colors or default_colors
    styles = styles or default_styles
    widths = widths or default_widths
    x_scales = x_scales or [1.0] * len(files)

    fig, ax = plt.subplots()

    col_iter = itertools.cycle(colors)
    sty_iter = itertools.cycle(styles)
    w_iter = itertools.cycle(widths)

    series_list = []

    for file, scale in zip(files, x_scales):
        x, y = load_profile(file)
        x_scaled = x * scale
        label = os.path.basename(file)[:-4]
        ax.plot(
            x_scaled,
            y,
            color=next(col_iter),
            linestyle=next(sty_iter),
            linewidth=next(w_iter),
            label=label,
        )
        # Prepare labeled Series for Excel
        series_list.append(pd.Series(x_scaled, name=f"Scaled X ({label})"))
        series_list.append(pd.Series(y, name=f"Y ({label})"))

    ax.set_xlabel("Distance in Pixels", fontsize=font_size + 8)
    ax.set_ylabel("Normalized Profile", fontsize=font_size + 8)

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    ax.tick_params(axis="both", which="major", labelsize=font_size + 4)
    ax.tick_params(axis="both", which="minor", labelsize=font_size + 2)

    ax.legend(fontsize=font_size + 2)
    fig.tight_layout()

    # Save figure if requested
    if out_path:
        fig.savefig(out_path, dpi=300)
        excel_path = os.path.splitext(out_path)[0] + "_export.xlsx"
    else:
        plt.show()
        excel_path = "profiles_export.xlsx"

    # Export to Excel using concatenation (handles varying lengths)
    df = pd.concat(series_list, axis=1)
    df.index.name = "Index"
    df.to_excel(excel_path, index=True)
    print(f"[INFO] Exported scaled profile data to: {excel_path}")


def parse_cli():
    p = argparse.ArgumentParser(description="Plot multiple x-y CSV profiles.")
    p.add_argument("files", nargs="+", help="CSV files with two columns: x,y")
    p.add_argument(
        "--x-scales",
        default="",
        help="Comma-separated list of x scale factors (one per file, default 1)",
    )
    p.add_argument("--font-size", type=int, default=12, help="Base font size")
    p.add_argument("--colors", default="", help='Comma-sep list of colors (e.g. "C0,#ff0000,C2")')
    p.add_argument("--styles", default="", help='Comma-sep list of line styles (e.g. "solid,dashed,dotted")')
    p.add_argument("--widths", default="", help='Comma-sep list of line widths (e.g. "1.5,2,1.5")')
    p.add_argument("--output", "-o", help="Save figure instead of showing (file name)")
    return p.parse_args()


def main():
    args = parse_cli()

    def to_list(s, cast=str):
        return [cast(v) for v in s.split(",")] if s else None

    x_scales = to_list(args.x_scales, float)
    colors = to_list(args.colors, str)
    styles = to_list(args.styles, str)
    widths = to_list(args.widths, float)

    if x_scales and len(x_scales) != len(args.files):
        raise ValueError("Number of x-scales must match number of input files.")

    plot_profiles(
        args.files,
        x_scales=x_scales,
        font_size=args.font_size,
        colors=colors,
        styles=styles,
        widths=widths,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
