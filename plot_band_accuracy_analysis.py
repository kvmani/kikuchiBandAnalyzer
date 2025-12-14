# This script reads the uploaded Excel file and reproduces the two plots
# using the same style and semantics as in the user's simulator/plotter.
# It saves PNGs to /mnt/data and also displays the figures inline.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Paths ----------
xlsx_path = Path(r"C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\band_width_accuracy_analysis.xlsx")
assert xlsx_path.exists(), f"Excel not found at {xlsx_path}"

# ---------- Read Excel ----------
df = pd.read_excel(xlsx_path)

# Standardize column names by an explicit mapping (robust to small header variations)
colmap = {
    "band_width_true (pixels)": "bw_true",
    "ground_truth_bw (pixels)": "gt_bw",
    "ground_truth_bw_err (pixels)": "gt_bw_err",
    "noisy_bw (pixels)": "noisy_bw",
    "noisy_bw_err_low (pixels)": "noisy_bw_err_low",
    "noisy_bw_err_high (pixels)": "noisy_bw_err_high",
    "ml_bw (pixels)": "ml_bw",
    "ml_bw_err (pixels)": "ml_bw_err",
    "K (Å·px)": "K",
    "d_true (Å)": "d_true",
    "d_gt (Å)": "d_gt",
    "d_gt_err (Å)": "d_gt_err",
    "d_noisy (Å)": "d_noisy",
    "d_noisy_err_low (Å)": "d_noisy_err_low",
    "d_noisy_err_high (Å)": "d_noisy_err_high",
    "d_ml (Å)": "d_ml",
    "d_ml_err (Å)": "d_ml_err",
}

# Try to coerce headers exactly; if workbook has extra spaces, trim and match
trimmed = {c: c.strip() for c in df.columns}
df.columns = [trimmed[c] for c in df.columns]

# Verify all required columns exist
missing = [src for src in colmap.keys() if src not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in Excel: {missing}")

# Rename
df = df.rename(columns=colmap)

# ---------- Arrays ----------
x_true = df["bw_true"].to_numpy(dtype=float)

y_gt    = df["gt_bw"].to_numpy(dtype=float)
y_noisy = df["noisy_bw"].to_numpy(dtype=float)
y_ml    = df["ml_bw"].to_numpy(dtype=float)

yerr_gt     = df["gt_bw_err"].to_numpy(dtype=float)
yerr_noisy  = np.vstack([
    df["noisy_bw_err_low"].to_numpy(dtype=float),
    df["noisy_bw_err_high"].to_numpy(dtype=float),
])
yerr_ml     = df["ml_bw_err"].to_numpy(dtype=float)

dx_true = df["d_true"].to_numpy(dtype=float)
dy_gt    = df["d_gt"].to_numpy(dtype=float)
dy_noisy = df["d_noisy"].to_numpy(dtype=float)
dy_ml    = df["d_ml"].to_numpy(dtype=float)

dyerr_gt    = df["d_gt_err"].to_numpy(dtype=float)
dyerr_noisy = np.vstack([
    df["d_noisy_err_low"].to_numpy(dtype=float),
    df["d_noisy_err_high"].to_numpy(dtype=float),
])
dyerr_ml    = df["d_ml_err"].to_numpy(dtype=float)

# ---------- Style (same as user's code) ----------
label_font      = {"size": 16}
title_font      = {"size": 16, "weight": "bold"}
tick_label_size = 14
legend_font     = {"size": 14}
grid_alpha      = 0.01

c_gt    = "#1f77b4"   # blue
c_noisy = "#ff7f0e"   # orange
c_ml    = "#2ca02c"   # green

# ---------- Figure 1: Band width ----------
plt.figure(figsize=(12, 5), dpi=120)
ax1 = plt.gca()

ax1.errorbar(x_true, y_gt,    yerr=yerr_gt,    fmt='o', ms=4, lw=1.5, color=c_gt,    ecolor=c_gt,    capsize=3, label="ground_truth (px)")
ax1.errorbar(x_true, y_noisy, yerr=yerr_noisy, fmt='^', ms=4, lw=1.5, color=c_noisy, ecolor=c_noisy, capsize=3, label="noisy (px)")
ax1.errorbar(x_true, y_ml,    yerr=yerr_ml,    fmt='s', ms=4, lw=1.5, color=c_ml,    ecolor=c_ml,    capsize=3, label="ML_processed (px)")

ax1.minorticks_on()
ax1.set_xlabel("band_width_true (px)", fontdict=label_font)
ax1.set_ylabel("Estimated band_width (px)", fontdict=label_font)
ax1.set_title("Band width: truth vs estimates", fontdict=title_font, pad=10)
ax1.tick_params(axis='both', labelsize=tick_label_size)
ax1.grid(True, ls='--', alpha=grid_alpha)
ax1.legend(prop=legend_font)

bw_png = "band_width_truth_vs_estimates.png"
plt.tight_layout()
plt.savefig(bw_png, bbox_inches="tight")
plt.show()

# ---------- Figure 2: d-spacing ----------
plt.figure(figsize=(12, 5), dpi=120)
ax2 = plt.gca()

ax2.errorbar(dx_true, dy_gt,    yerr=dyerr_gt,    fmt='o', ms=4, lw=1.5, color=c_gt,    ecolor=c_gt,    capsize=3, label="d_gt (Å)")
ax2.errorbar(dx_true, dy_noisy, yerr=dyerr_noisy, fmt='^', ms=4, lw=1.5, color=c_noisy, ecolor=c_noisy, capsize=3, label="d_noisy (Å)")
ax2.errorbar(dx_true, dy_ml,    yerr=dyerr_ml,    fmt='s', ms=4, lw=1.5, color=c_ml,    ecolor=c_ml,    capsize=3, label="d_ml (Å)")

ax2.minorticks_on()
ax2.set_xlabel("d_true (Å)", fontdict=label_font)
ax2.set_ylabel("Estimated d (Å)", fontdict=label_font)
ax2.set_title("d-spacing: truth vs estimates", fontdict=title_font, pad=10)
ax2.tick_params(axis='both', labelsize=tick_label_size)
ax2.grid(True, ls='--', alpha=grid_alpha)
ax2.legend(prop=legend_font)

d_png = "d_spacing_truth_vs_estimates.png"
plt.tight_layout()
plt.savefig(d_png, bbox_inches="tight")
plt.show()

bw_png, d_png
