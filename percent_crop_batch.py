#!/usr/bin/env python3
"""
Batch Image Crop & Resize Pipeline (OOP, Python)

Features
- Input via a folder and/or explicit file list
- Optional simulation mode (default True): create a 1500x1500 synthetic image
- Grayscale normalization
- Percentage-based crops computed from ORIGINAL dimensions (not cumulative)
- Resize each crop to a target size (default 460x460)
- Debug visualization: 1x3 panel (original, cropped, resized)
- Logging for crucial steps
- Configurable via a config dict in __main__

Dependencies: Pillow, numpy, matplotlib, tqdm (optional)

Author: CodeBuilder
"""
from __future__ import annotations

import os
import sys
import math
import logging
import glob
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterable

from PIL import Image
import numpy as np

# Matplotlib is imported lazily inside DebugVisualizer to avoid overhead if debug is False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


SUPPORTED_EXTS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
}


# -----------------------------
# Logger Factory
# -----------------------------
class LoggerFactory:
    @staticmethod
    def configure(level: str = "INFO") -> logging.Logger:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger("ImageCropper")
        logger.debug("Logger configured with level %s", level)
        return logger


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class ImageSource:
    """Represents a source of an image, either from disk or simulated in-memory."""
    display_name: str
    path: Optional[str] = None
    _pil_image: Optional[Image.Image] = field(default=None, repr=False)

    def is_simulated(self) -> bool:
        return self.path is None and self._pil_image is not None

    def load(self) -> Image.Image:
        if self._pil_image is not None:
            return self._pil_image.copy()
        if self.path is None:
            raise ValueError("ImageSource has neither path nor in-memory image.")
        return Image.open(self.path)


@dataclass
class CropRange:
    start: int = 2
    stop: int = 50
    step: int = 3

    def generate(self) -> List[int]:
        # Inclusive stop if it lands exactly; otherwise highest less than stop
        vals = list(range(self.start, self.stop + 1, self.step))
        # Ensure stop included if it is not aligned but should be per spec (here spec wants 50 included)
        if vals and vals[-1] != self.stop and self.stop > vals[-1]:
            vals.append(self.stop)
        return vals


@dataclass
class CropTaskConfig:
    input_folder: Optional[str] = None
    input_files: List[str] = field(default_factory=list)
    simulate: bool = True
    output_folder: str = "./out"
    target_size: Tuple[int, int] = (460, 460)
    crop: CropRange = field(default_factory=lambda: CropRange(2, 50, 3))
    debug: bool = False
    debug_max_panels_per_image: Optional[int] = None
    log_level: str = "INFO"
    overwrite: bool = True


# -----------------------------
# Debug Visualizer
# -----------------------------
class DebugVisualizer:
    def __init__(self, enabled: bool, max_panels: Optional[int] = None):
        self.enabled = enabled
        self.max_panels = max_panels
        self._shown = 0
        if self.enabled:
            # Lazy import to avoid overhead if not used
            import matplotlib.pyplot as plt  # noqa: F401

    def maybe_show_triptych(self, original: Image.Image, cropped: Image.Image, resized: Image.Image,
                             title: str, crop_pct: int) -> None:
        if not self.enabled:
            return
        if self.max_panels is not None and self._shown >= self.max_panels:
            return
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(f"{title} | Crop: {crop_pct}%", fontsize=12)

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(original, cmap='gray')
        ax1.set_title("Original (grayscale)")
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(cropped, cmap='gray')
        w_c, h_c = cropped.size
        ax2.set_title(f"Cropped ({w_c}x{h_c})")
        ax2.axis('off')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(resized, cmap='gray')
        w_r, h_r = resized.size
        ax3.set_title(f"Resized ({w_r}x{h_r})")
        ax3.axis('off')

        fig.tight_layout()
        plt.show()
        self._shown += 1


# -----------------------------
# Core ImageCropper
# -----------------------------
class ImageCropper:
    def __init__(self, config: CropTaskConfig):
        self.cfg = config
        self.logger = LoggerFactory.configure(self.cfg.log_level)
        self.visualizer = DebugVisualizer(self.cfg.debug, self.cfg.debug_max_panels_per_image)

        # Ensure output folder exists
        os.makedirs(self.cfg.output_folder, exist_ok=True)
        self.logger.info("Output folder: %s", os.path.abspath(self.cfg.output_folder))

    # ---- Discovery ----
    def collect_images(self) -> List[ImageSource]:
        sources: List[ImageSource] = []
        seen: set[str] = set()

        # From folder
        if self.cfg.input_folder:
            folder = self.cfg.input_folder
            if not os.path.isdir(folder):
                self.logger.warning("Input folder does not exist: %s", folder)
            else:
                for ext in SUPPORTED_EXTS:
                    pattern = os.path.join(folder, f"*{ext}")
                    for path in glob.glob(pattern):
                        norm = os.path.abspath(path)
                        if norm.lower() in seen:
                            continue
                        sources.append(ImageSource(display_name=os.path.basename(path), path=path))
                        seen.add(norm.lower())

        # From explicit files
        for f in self.cfg.input_files:
            if not os.path.isfile(f):
                self.logger.warning("Input file not found (skipped): %s", f)
                continue
            _, ext = os.path.splitext(f)
            if ext.lower() not in SUPPORTED_EXTS:
                self.logger.warning("Unsupported extension (skipped): %s", f)
                continue
            norm = os.path.abspath(f)
            if norm.lower() in seen:
                continue
            sources.append(ImageSource(display_name=os.path.basename(f), path=f))
            seen.add(norm.lower())

        # Simulation
        if self.cfg.simulate:
            sim_img = self._generate_simulated_image()
            sources.insert(0, ImageSource(display_name="SIMULATED_1500x1500_band", path=None, _pil_image=sim_img))
            self.logger.info("Simulation mode is ON. Synthetic image added to the queue first.")

        if not sources:
            self.logger.warning("No input images found. Nothing to do.")
        else:
            self.logger.info("Discovered %d image(s).", len(sources))
        return sources

    # ---- Simulation ----
    def _generate_simulated_image(self) -> Image.Image:
        self.logger.info("Generating 1500x1500 simulated image with a central 100px white band.")
        h = w = 1500
        band_h = 100
        arr = np.zeros((h, w), dtype=np.uint8)
        top = (h - band_h) // 2
        arr[top:top+band_h, :] = 255
        return Image.fromarray(arr, mode='L')

    # ---- Processing primitives ----
    def to_grayscale(self, img: Image.Image) -> Image.Image:
        if img.mode != 'L':
            return img.convert('L')
        return img

    def generate_crop_percents(self) -> List[int]:
        pcs = self.cfg.crop.generate()
        # Ensure standard sequence according to spec (2..50 step 3): 2,5,8,...,50
        return pcs

    def crop_once(self, img: Image.Image, original_size: Tuple[int, int], pct: int) -> Optional[Image.Image]:
        """Crop img by pct computed from original_size (width,height)."""
        orig_w, orig_h = original_size
        dx = int(round(orig_w * (pct / 100.0)))
        dy = int(round(orig_h * (pct / 100.0)))
        left = dx
        top = dy
        right = orig_w - dx
        bottom = orig_h - dy
        new_w = right - left
        new_h = bottom - top
        if new_w <= 0 or new_h <= 0:
            return None
        # Perform crop on the CURRENT image: we must map from original crop box.
        # Since img may be already grayscale-converted but same size as original at this stage,
        # we assume the current img still has original_size.
        return img.crop((left, top, right, bottom))

    def resize_image(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        return img.resize(size, resample=Image.BICUBIC)

    # ---- Filename helpers ----
    @staticmethod
    def _base_name_without_ext(filename: str) -> str:
        base = os.path.basename(filename)
        name, _ = os.path.splitext(base)
        return name

    def _output_path(self, base_name: str, pct: int) -> str:
        fname = f"{base_name}_{pct}_pct.png"
        return os.path.join(self.cfg.output_folder, fname)

    # ---- Per-image pipeline ----
    def process_image(self, src: ImageSource) -> None:
        src_desc = src.display_name
        self.logger.info("Processing image: %s%s",
                         src_desc,
                         " (SIMULATED)" if src.is_simulated() else "")
        try:
            img = src.load()
        except Exception as e:
            self.logger.error("Failed to load image '%s': %s", src_desc, e)
            return

        # Normalize to grayscale
        img = self.to_grayscale(img)
        orig_w, orig_h = img.size
        self.logger.debug("Original size (grayscale): %dx%d", orig_w, orig_h)

        # Determine base name
        if src.is_simulated():
            base_name = "simulated"
        else:
            base_name = self._base_name_without_ext(src.path or src.display_name)

        crop_percents = self.generate_crop_percents()
        iterator: Iterable[int] = crop_percents
        if _HAS_TQDM:
            iterator = tqdm(crop_percents, desc=f"Crops for {src_desc}")

        for pct in iterator:
            cropped = self.crop_once(img, (orig_w, orig_h), pct)
            if cropped is None:
                self.logger.warning("Skipping crop %d%% for '%s' -> non-positive size.", pct, src_desc)
                continue

            resized = self.resize_image(cropped, self.cfg.target_size)

            # Save
            out_path = self._output_path(base_name, pct)
            if os.path.exists(out_path) and not self.cfg.overwrite:
                self.logger.info("File exists, skipping (overwrite=False): %s", out_path)
            else:
                try:
                    resized.save(out_path)
                    self.logger.info("Saved: %s", out_path)
                except Exception as e:
                    self.logger.error("Failed to save '%s': %s", out_path, e)

            # Debug visualization
            self.visualizer.maybe_show_triptych(img, cropped, resized, title=src_desc, crop_pct=pct)

        self.logger.info("Completed: %s", src_desc)

    # ---- Orchestrator ----
    def run(self) -> None:
        sources = self.collect_images()
        if not sources:
            return
        for src in sources:
            self.process_image(src)
        self.logger.info("All done. Processed %d image(s).", len(sources))


# -----------------------------
# Entrypoint with config
# -----------------------------
if __name__ == "__main__":
    # Editable configuration dictionary
    config = {
        "input_folder": None,                     # e.g., "./images" or None
        "input_files": [
            #r'C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\0pct_8.396\0 45 0\1840x1840.BMP',
            r'C:\Users\kvman\Documents\ml_data\accuracy_testing_ML-EBSD-Patterns-Magnetite\0pct_8.396\0 0 0\1840x1840.BMP',
                        ],                        # e.g., ["/path/a.png", "/path/b.jpg"]
        "simulate": False,                         # DEFAULT True per user request
        "output_folder": "./tmp_sim_out",        # default output for simulation
        "target_size": (460, 460),
        "crop": {"start": 5, "stop": 25, "step": 5},
        "debug": True,                           # Set True to see triptych panels
        "debug_max_panels_per_image": None,       # e.g., 5 to limit panels
        "log_level": "INFO",                     # "DEBUG" for more details
        "overwrite": True,
    }

    # Build the typed config objects
    crop_cfg = CropRange(
        start=int(config.get("crop", {}).get("start", 2)),
        stop=int(config.get("crop", {}).get("stop", 50)),
        step=int(config.get("crop", {}).get("step", 3)),
    )

    task_cfg = CropTaskConfig(
        input_folder=config.get("input_folder"),
        input_files=list(config.get("input_files", [])),
        simulate=bool(config.get("simulate", True)),
        output_folder=str(config.get("output_folder", "./out")),
        target_size=tuple(config.get("target_size", (460, 460))),
        crop=crop_cfg,
        debug=bool(config.get("debug", False)),
        debug_max_panels_per_image=config.get("debug_max_panels_per_image"),
        log_level=str(config.get("log_level", "INFO")),
        overwrite=bool(config.get("overwrite", True)),
    )

    cropper = ImageCropper(task_cfg)
    cropper.run()
