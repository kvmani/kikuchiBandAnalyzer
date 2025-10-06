#!/usr/bin/env python3
"""
Interactive Line Intensity Profiler (Standalone)
- Draw lines on an image and see matching intensity profiles.
- Simulated 100x100 image or load from path (auto-grayscale via Pillow).
- Bilinear sampling, pixel-distance x-axis.
- Edit endpoints (toggle 'e'), undo ('u'), clear ('c'), save PNG+CSV ('s'), toggle stats ('t').
- Publication-friendly fonts/legends. Minimal external deps: numpy, pillow, matplotlib.

User settings live in __main__ as a CONFIG dict.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------- Utilities ---------------------------

def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """2D float image in [0,1] from grayscale/RGB/RGBA."""
    if arr.ndim == 2:
        img = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        im = Image.fromarray(arr).convert("L")
        img = np.array(im, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def fmt_sig(x: float, sig: int = 2) -> str:
    try:
        return f"{float(x):.{sig}g}"
    except Exception:
        return str(x)

# --------------------------- Data ---------------------------

@dataclass
class LineSpec:
    idx: int
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    color: str
    label: str
    artist: Optional[Line2D] = None
    profile_x: Optional[np.ndarray] = None
    profile_y: Optional[np.ndarray] = None
    stats_text_artist: Optional[matplotlib.text.Text] = None

# --------------------------- Core ---------------------------

class InteractiveLineProfiler:
    def __init__(
        self,
        image_source: str,
        simulate: bool = False,
        seed: Optional[int] = None,
        dpi: int = 150,
        csv_path: Optional[str] = None,
        png_path: Optional[str] = None,
        show_stats: bool = False,
        figsize: Tuple[float, float] = (11, 5),
        font_size: int = 11,
    ) -> None:
        self.image_source = image_source
        self.simulate = simulate
        self.seed = seed
        self.dpi = dpi
        self.csv_path = csv_path
        self.png_path = png_path
        self.show_stats = show_stats
        self.figsize = figsize
        self.font_size = font_size

        self.img: np.ndarray = self._load_image()
        self.h, self.w = self.img.shape

        # Matplotlib state
        self.fig: Optional[matplotlib.figure.Figure] = None
        self.ax_img: Optional[matplotlib.axes.Axes] = None
        self.ax_prof: Optional[matplotlib.axes.Axes] = None

        # Interaction state
        self.pending_point: Optional[Tuple[float, float]] = None
        self.lines: List[LineSpec] = []
        self.edit_mode: bool = False
        self.dragging: bool = False
        self.drag_target: Optional[Tuple[int, str]] = None

        # Connectors
        self.cid_click = None
        self.cid_motion = None
        self.cid_release = None
        self.cid_key = None

        self._setup_figure()
        self._connect_events()

    # ----------------------- Load & Figure -----------------------

    def _load_image(self) -> np.ndarray:
        if self.simulate:
            rng = np.random.default_rng(self.seed)
            return rng.random((100, 100), dtype=np.float32)
        if not os.path.isfile(self.image_source):
            raise FileNotFoundError(f"Image not found: {self.image_source}")
        pil = Image.open(self.image_source)
        arr = np.array(pil)
        return to_grayscale(arr)

    def _setup_figure(self) -> None:
        matplotlib.rcParams.update({
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "font.size": self.font_size,
            "axes.titlesize": self.font_size + 1,
            "axes.labelsize": self.font_size,
            "legend.fontsize": self.font_size - 1,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
        })
        self.fig, (self.ax_img, self.ax_prof) = plt.subplots(1, 2, figsize=self.figsize, constrained_layout=True)

        # (a) Image
        self.ax_img.imshow(self.img, cmap="gray", interpolation="nearest", origin="upper")
        self.ax_img.set_title(self._title_for_image())
        self.ax_img.set_xlabel("X (px)")
        self.ax_img.set_ylabel("Y (px)")
        self.ax_img.set_xlim(0, self.w - 1)
        self.ax_img.set_ylim(self.h - 1, 0)

        # (b) Profiles
        self.ax_prof.set_title("(b) Intensity Profiles")
        self.ax_prof.set_xlabel("Distance (px)")
        self.ax_prof.set_ylabel("Intensity (0–1)")
        self.ax_prof.set_xlim(0, max(self.w, self.h))
        self.ax_prof.set_ylim(0, 1)

        # Compact help at left edge of the figure (small text)
        help_txt = "Add: 2×LMB  |  Edit: e+drag  |  Undo: u  |  Clear: c  |  Save: s  |  Stats: t  |  Quit: q"
        self.fig.text(0.005, 0.5, help_txt, rotation=90, va="center", ha="left",
                      fontsize=max(self.font_size - 3, 6), alpha=0.75)

    def _title_for_image(self) -> str:
        base = "simulated" if self.simulate else (os.path.splitext(os.path.basename(self.image_source))[0] or "image")
        return f"(a) Image: {base}"

    # ----------------------- Events -----------------------

    def _connect_events(self) -> None:
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _inside_image_axes(self, event) -> bool:
        return event.inaxes is self.ax_img

    def _clamp_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return float(np.clip(x, 0, self.w - 1)), float(np.clip(y, 0, self.h - 1))

    def _on_click(self, event):
        if not self._inside_image_axes(event) or event.button != 1:
            return
        x, y = self._clamp_to_image(event.xdata, event.ydata)
        if self.edit_mode:
            target = self._pick_endpoint(x, y, tol_px=8)
            if target is not None:
                self.dragging = True
                self.drag_target = target
            return
        # Creation mode
        if self.pending_point is None:
            self.pending_point = (x, y)
            self._draw_temporary_marker(x, y)
        else:
            p0, p1 = self.pending_point, (x, y)
            if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1.0:
                self.pending_point = None
                self._clear_temporary_marker()
                return
            self._add_line(p0, p1)
            self.pending_point = None
            self._clear_temporary_marker()

    def _on_motion(self, event):
        if not (self.dragging and self._inside_image_axes(event) and self.drag_target):
            return
        x, y = self._clamp_to_image(event.xdata, event.ydata)
        line_idx, which = self.drag_target
        ls = self.lines[line_idx]
        if which == 'p0':
            ls.p0 = (x, y)
        else:
            ls.p1 = (x, y)
        if ls.artist is not None:
            ls.artist.set_data([ls.p0[0], ls.p1[0]], [ls.p0[1], ls.p1[1]])
        self._update_line_profile(ls)
        self._plot_profiles()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.drag_target = None

    def _on_key(self, event):
        if event.key == 'e':
            self.edit_mode = not self.edit_mode
            self._set_status(f"EDIT: {'ON' if self.edit_mode else 'OFF'}")
        elif event.key == 'u':
            self._undo_last()
        elif event.key == 'c':
            self._clear_all()
        elif event.key == 's':
            self._save_all()
        elif event.key == 't':
            self.show_stats = not self.show_stats
            self._refresh_stats_overlay()
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)

    # ----------------------- Line ops -----------------------

    def _pick_endpoint(self, x: float, y: float, tol_px: float = 8.0) -> Optional[Tuple[int, str]]:
        best, best_dist = None, float('inf')
        for i, ls in enumerate(self.lines):
            for which, p in [('p0', ls.p0), ('p1', ls.p1)]:
                d = np.hypot(x - p[0], y - p[1])
                if d < best_dist and d <= tol_px:
                    best_dist, best = d, (i, which)
        return best

    def _draw_temporary_marker(self, x: float, y: float) -> None:
        size = 4
        self._clear_temporary_marker()
        self.temp_marker = self.ax_img.plot([x - size, x + size], [y, y], lw=1, color='white', alpha=0.8)[0]
        self.temp_marker2 = self.ax_img.plot([x, x], [y - size, y + size], lw=1, color='white', alpha=0.8)[0]
        self.fig.canvas.draw_idle()

    def _clear_temporary_marker(self) -> None:
        for attr in ('temp_marker', 'temp_marker2'):
            artist = getattr(self, attr, None)
            if artist is not None:
                artist.remove()
                setattr(self, attr, None)

    def _add_line(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> None:
        color = self._next_color()
        idx = len(self.lines) + 1
        label = f"Line {idx}"
        artist, = self.ax_img.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', lw=2, color=color, label=label)
        ls = LineSpec(idx=idx, p0=p0, p1=p1, color=color, label=label, artist=artist)
        self._update_line_profile(ls)
        self.lines.append(ls)
        self._update_image_legend()
        self._plot_profiles()
        self.fig.canvas.draw_idle()

    def _update_line_profile(self, ls: LineSpec) -> None:
        d, vals = self._sample_profile(ls.p0, ls.p1)
        ls.profile_x, ls.profile_y = d, vals
        if self.show_stats:
            self._draw_stats_for_line(ls)
        else:
            self._remove_stats_for_line(ls)

    def _undo_last(self) -> None:
        if not self.lines:
            return
        ls = self.lines.pop()
        if ls.artist is not None:
            ls.artist.remove()
        self._remove_stats_for_line(ls)
        for i, l in enumerate(self.lines, start=1):
            l.idx = i
            l.label = f"Line {i}"
            if l.artist is not None:
                l.artist.set_label(l.label)
        self._update_image_legend()
        self._plot_profiles()
        self.fig.canvas.draw_idle()

    def _clear_all(self) -> None:
        for ls in self.lines:
            if ls.artist is not None:
                ls.artist.remove()
            self._remove_stats_for_line(ls)
        self.lines.clear()
        self._update_image_legend()
        self._plot_profiles()
        self.fig.canvas.draw_idle()

    def _update_image_legend(self) -> None:
        leg = self.ax_img.get_legend()
        if leg is not None:
            leg.remove()
        if self.lines:
            self.ax_img.legend(loc='upper right', frameon=True, framealpha=0.8)

    def _plot_profiles(self) -> None:
        self.ax_prof.cla()
        self.ax_prof.set_title("(b) Intensity Profiles")
        self.ax_prof.set_xlabel("Distance (px)")
        self.ax_prof.set_ylabel("Intensity (0–1)")
        xlim = 0
        for ls in self.lines:
            if ls.profile_x is not None and ls.profile_y is not None:
                self.ax_prof.plot(ls.profile_x, ls.profile_y, '-', lw=2, color=ls.color, label=ls.label)
                xlim = max(xlim, float(ls.profile_x[-1]) if len(ls.profile_x) else 0)
        if self.lines:
            self.ax_prof.legend(loc='best', frameon=True, framealpha=0.8)
        self.ax_prof.set_xlim(0, max(xlim, 10))
        self.ax_prof.set_ylim(0, 1)

    def _draw_stats_for_line(self, ls: LineSpec) -> None:
        self._remove_stats_for_line(ls)
        if ls.profile_y is None or ls.artist is None:
            return
        vals = ls.profile_y
        text = f"μ={fmt_sig(np.mean(vals))}, σ={fmt_sig(np.std(vals))}, min={fmt_sig(np.min(vals))}, max={fmt_sig(np.max(vals))}"
        midx = 0.5 * (ls.p0[0] + ls.p1[0])
        midy = 0.5 * (ls.p0[1] + ls.p1[1])
        ls.stats_text_artist = self.ax_img.text(
            midx, midy, text, color=ls.color, fontsize=max(self.font_size - 2, 6),
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='black', ec=ls.color, alpha=0.35),
        )

    def _remove_stats_for_line(self, ls: LineSpec) -> None:
        if ls.stats_text_artist is not None:
            try:
                ls.stats_text_artist.remove()
            except Exception:
                pass
            ls.stats_text_artist = None

    def _refresh_stats_overlay(self) -> None:
        for ls in self.lines:
            if self.show_stats:
                self._draw_stats_for_line(ls)
            else:
                self._remove_stats_for_line(ls)

    def _next_color(self) -> str:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
        return colors[len(self.lines) % len(colors)]

    # ----------------------- Sampling -----------------------

    def _sample_profile(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        n = int(round(max(abs(dx), abs(dy)))) + 1
        if n <= 1:
            return np.array([0.0], dtype=np.float32), np.array([self._bilinear(x0, y0)], dtype=np.float32)
        ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
        xs = x0 + ts * dx
        ys = y0 + ts * dy
        vals = np.array([self._bilinear(x, y) for x, y in zip(xs, ys)], dtype=np.float32)
        dists = np.hypot(xs - x0, ys - y0)
        return dists, vals

    def _bilinear(self, x: float, y: float) -> float:
        x = np.clip(x, 0, self.w - 1)
        y = np.clip(y, 0, self.h - 1)
        x0 = int(np.floor(x)); x1 = min(x0 + 1, self.w - 1)
        y0 = int(np.floor(y)); y1 = min(y0 + 1, self.h - 1)
        dx, dy = x - x0, y - y0
        I00 = self.img[y0, x0]; I10 = self.img[y0, x1]
        I01 = self.img[y1, x0]; I11 = self.img[y1, x1]
        I0 = I00 * (1 - dx) + I10 * dx
        I1 = I01 * (1 - dx) + I11 * dx
        return float(I0 * (1 - dy) + I1 * dy)

    # ----------------------- Save -----------------------

    def _save_all(self) -> None:
        base = "simulated" if self.simulate else os.path.splitext(os.path.basename(self.image_source))[0]
        csv_path = self.csv_path or f"{base}_profiles.csv"
        png_path = self.png_path or f"{base}_figure.png"
        try:
            self._save_csv(csv_path)
            self.fig.savefig(png_path, bbox_inches='tight')
            self._set_status(f"Saved: {csv_path}, {png_path}")
        except Exception as e:
            self._set_status(f"Save failed: {e}")

    def _save_csv(self, path: str) -> None:
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["line_id", "x0", "y0", "x1", "y1", "distance_px", "intensity"])
            for ls in self.lines:
                if ls.profile_x is None or ls.profile_y is None:
                    continue
                for d, I in zip(ls.profile_x, ls.profile_y):
                    w.writerow([ls.idx, f"{ls.p0[0]:.3f}", f"{ls.p0[1]:.3f}", f"{ls.p1[0]:.3f}", f"{ls.p1[1]:.3f}", f"{d:.3f}", f"{I:.6f}"])

    def _set_status(self, msg: str) -> None:
        self.ax_prof.set_title(f"(b) Intensity Profiles — {msg}")
        self.fig.canvas.draw_idle()

    # ----------------------- Public -----------------------

    def show(self) -> None:
        plt.show()

# --------------------------- __main__ ---------------------------

if __name__ == "__main__":
    # All user settings here
    CONFIG = {
        # Pick one of the two sources below:
        "simulate": True,                 # True -> use 100x100 random image
        "image_path": None,               # e.g., r"/path/to/image.png" when simulate=False
        "seed": 123,                      # RNG seed for simulation

        # Outputs (optional). If None, auto-names based on image base name
        "csv_path": None,                 # e.g., "profiles.csv"
        "png_path": None,                 # e.g., "figure.png"

        # Figure appearance & behavior
        "dpi": 150,
        "show_stats": True,               # overlay μ/σ/min/max on image lines
        "figsize": (11, 5),
        "font_size": 11,
    }

    simulate = bool(CONFIG.get("simulate", False))
    image_source = "simulated" if simulate else (CONFIG.get("image_path") or "")
    profiler = InteractiveLineProfiler(
        image_source=image_source,
        simulate=simulate,
        seed=CONFIG.get("seed"),
        dpi=int(CONFIG.get("dpi", 150)),
        csv_path=CONFIG.get("csv_path"),
        png_path=CONFIG.get("png_path"),
        show_stats=bool(CONFIG.get("show_stats", False)),
        figsize=tuple(CONFIG.get("figsize", (11, 5))),
        font_size=int(CONFIG.get("font_size", 11)),
    )

    profiler.show()
