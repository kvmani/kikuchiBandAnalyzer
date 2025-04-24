#!/usr/bin/env python3
"""
plot_profiles.py – Plot several two-column (x,y) CSV files on one graph.

Example
-------
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

# ───────────────────────── helper utilities ─────────────────────────
def _has_header(path: str) -> bool:
    """Detect if the first non-blank row is textual (header)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:                          # first non-blank
                return not line[0].isdigit()
    return False


def load_profile(path: str):
    """Load two-column CSV, automatically skipping one header row (if any)."""
    skip = 1 if _has_header(path) else 0
    x, y = np.loadtxt(path, delimiter=",", skiprows=skip, unpack=True)
    return x, y


def infinite_cycle(seq):
    """Yield items of *seq* forever (useful for styling lists)."""
    return itertools.cycle(seq)


# ─────────────────────────── plotting core ──────────────────────────
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

    fig, ax = plt.subplots()                   # use explicit Axes object

    col_iter = itertools.cycle(colors)
    sty_iter = itertools.cycle(styles)
    w_iter = itertools.cycle(widths)
    scale_iter = itertools.cycle(x_scales)

    for f in files:
        x, y = load_profile(f)
        scale = next(scale_iter)
        ax.plot(
            x * scale,
            y,
            color=next(col_iter),
            linestyle=next(sty_iter),
            linewidth=next(w_iter),
            label=os.path.basename(f)[:-4],
        )

    # axis labels with larger font
    ax.set_xlabel("Distance in Pixels", fontsize=font_size + 8)
    ax.set_ylabel("Normalized Profile", fontsize=font_size + 8)

    # ax.xaxis.set_major_locator(mticker.LinearLocator(4))
    # ax.yaxis.set_major_locator(mticker.LinearLocator(4))

    # Set minor ticks (4 intervals between major ticks)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    # tick labels larger and minor ticks every 1/5th of major interval
    ax.tick_params(axis="both", which="major", labelsize=font_size + 4)
    ax.tick_params(axis="both", which="minor", labelsize=font_size + 2)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))   # 4 minors between majors
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    ax.legend(fontsize=font_size + 2)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
    else:
        plt.show()

# ────────────────────────── command-line I/O ────────────────────────
def parse_cli():
    p = argparse.ArgumentParser(description="Plot multiple x-y CSV profiles.")
    p.add_argument("files", nargs="+", help="CSV files with two columns: x,y")
    p.add_argument(
        "--x-scales",
        default="",
        help="Comma-separated list of x scale factors (one per file, default 1)",
    )
    p.add_argument("--font-size", type=int, default=12, help="Base font size")
    p.add_argument(
        "--colors",
        default="",
        help='Comma-sep list of colors (e.g. "C0,#ff0000,C2")',
    )
    p.add_argument(
        "--styles",
        default="",
        help='Comma-sep list of line styles (e.g. "solid,dashed,dotted")',
    )
    p.add_argument(
        "--widths",
        default="",
        help='Comma-sep list of line widths (e.g. "1.5,2,1.5")',
    )
    p.add_argument(
        "--output", "-o", help="Save figure instead of showing (file name)"
    )
    return p.parse_args()


def main():
    args = parse_cli()

    # Convert comma-separated CLI strings to lists (allow empty string)
    def to_list(s, cast=str):
        return [cast(v) for v in s.split(",")] if s else None

    x_scales = to_list(args.x_scales, float)
    colors = to_list(args.colors, str)
    styles = to_list(args.styles, str)
    widths = to_list(args.widths, float)

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
