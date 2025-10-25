"""
EBSD Kikuchi Pattern Noise Injector — Batch Capable
---------------------------------------------------
Script name: ebsd_gaussian_noise_batch.py

Purpose:
    - Load grayscale EBSD Kikuchi pattern image(s) and add Gaussian noise.
    - Accept `CONFIG["input_path"]` as:
        (a) a single path to an image file → process that one image
        (b) a list of image file paths → process each
        (c) a folder path → discover images in that folder and write results to
            {input_folder}/noisy/ with the SAME basename as the source files
            (only the folder name contains "noisy").
    - For single-file and list modes, business logic (naming/placement) remains
      exactly as before: save next to source using `derive_output_name()`.
    - In folder mode, output filenames are unchanged (no "noisy" in the name),
      but the extension may change to .png to preserve 16-bit unless
      CONFIG["preserve_format"] is True.
    - Debug visualization: matplotlib `plt.show()` is used. In batch modes,
      only the FIRST processed image is visualized by default.

Key features retained:
    - Robust dtype handling for uint8 / uint16 (float fallback)
    - Clipping to valid dynamic range, preserving dtype
    - Deterministic noise when a seed is provided
    - Safe filename creation with overwrite control

Dependencies:
    - Pillow (PIL)
    - numpy
    - matplotlib (only when debug=True)

Usage:
    - Define the `CONFIG` dict inside the `__main__` block and run with Python 3.9+.

Notes on saving format:
    - If the input is 16-bit grayscale and the chosen output format does not support
      16-bit grayscale (e.g., BMP), this script will switch to PNG to preserve bit
      depth unless you set CONFIG["preserve_format"] = True.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt  # Imported lazily in debug mode
except Exception:
    plt = None

# ---------------
# Utility helpers
# ---------------

def _sig_str(x: float, sig: int = 4) -> str:
    """Format a float with up to `sig` significant digits, stripping trailing zeros and dots."""
    if x == 0:
        return "0"
    s = f"{x:.{sig}g}"
    return s.replace(".", ".").rstrip("0").rstrip(".")


def _dynamic_range_for_dtype(dtype: np.dtype) -> Tuple[float, float]:
    """Return (lo, hi) valid range for the dtype. Float dtypes return (0.0, 1.0)."""
    if np.issubdtype(dtype, np.uint8):
        return 0.0, 255.0
    if np.issubdtype(dtype, np.uint16):
        return 0.0, 65535.0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        return 0.0, 1.0
    return 0.0, 255.0


def _ensure_grayscale(img: Image.Image) -> Image.Image:
    """Ensure the PIL image is single-channel grayscale. Converts if not."""
    if img.mode == "L" or img.mode == "I;16":
        return img
    return img.convert("L")


# --------------
# Data structures
# --------------
@dataclass
class NoiseConfig:
    variance: float
    amplitude: float
    mean: float = 0.0


# ----------------------
# Core single-image logic
# ----------------------
class GaussianNoiseAdder:
    """Add Gaussian noise to a grayscale EBSD image with robust dtype handling.

    This class encapsulates SINGLE-IMAGE processing and saving logic. Batch iteration
    is handled by `BatchOrchestrator` so we can reuse the same RNG for determinism.
    """

    def __init__(self, config: dict, rng: Optional[np.random.Generator] = None) -> None:
        # Note: we no longer store a single input_path here; that's provided per call.
        noise = config.get("noise", {})
        ntype = str(noise.get("type", "gaussian")).lower()
        if ntype != "gaussian":
            raise ValueError(f"Unsupported noise type: {ntype}")

        variance = float(noise.get("variance", 0.0))
        amplitude = float(noise.get("amplitude", 1.0))
        mean = float(noise.get("mean", 0.0))
        if variance < 0:
            raise ValueError("Variance must be non-negative.")

        self.noise_cfg = NoiseConfig(variance=variance, amplitude=amplitude, mean=mean)
        self.overwrite: bool = bool(config.get("overwrite", False))
        self.preserve_format: bool = bool(config.get("preserve_format", False))
        self.debug: bool = bool(config.get("debug", False))
        self.debug_batch_first_only: bool = bool(config.get("debug_batch_first_only", True))
        blur_cfg = config.get("blur", {})
        self.blur_enabled: bool = bool(blur_cfg.get("enabled", False))
        self.blur_sigma: float = float(blur_cfg.get("sigma", 1.0))

        # RNG handling: share a single RNG (seeded in orchestrator) across batch for determinism
        self._rng = rng if rng is not None else np.random.default_rng(config.get("seed", None))

    def derive_output_name_to_folder(
        self,
        input_path: Path,
        out_dir: Path,
        relative_root: Optional[Path],
        dtype: np.dtype
    ) -> Path:
        """
        Build a filename for use under a single output folder:
        - Use the path tail relative to `relative_root` if possible.
        - Replace path separators with underscores; keep spaces.
        - Append '_noisy' before the extension.
        - Apply bit-depth extension adjustment via _maybe_adjust_extension.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # Try to compute a relative tail; fall back to basename if not possible
        try:
            if relative_root is not None:
                rel = input_path.relative_to(relative_root)
            else:
                rel = Path(input_path.name)
        except Exception:
            rel = Path(input_path.name)

        # Convert tail parts to a single stem joined by underscores
        parts = list(rel.parts)
        if len(parts) == 0:
            parts = [input_path.name]
        stem_no_ext = Path(parts[-1]).stem
        if len(parts) > 1:
            # prepend directory parts (without separators) before the file stem
            dir_join = "_".join(parts[:-1])  # preserves spaces inside names
            safe_stem = f"{dir_join}_{stem_no_ext}"
        else:
            safe_stem = stem_no_ext

        # Append `_noisy` and keep original extension initially
        ext = input_path.suffix
        candidate = out_dir / f"{safe_stem}_noisy{ext}"

        # Adjust extension if needed for 16-bit preservation policy
        candidate = self._maybe_adjust_extension(candidate, dtype)

        # Avoid collisions unless overwrite=True
        candidate = self._resolve_collision(candidate)
        return candidate

    # ---- I/O ----

    def load_image(self, path: Path) -> Tuple[np.ndarray, np.dtype, str]:
        """Load the input image as grayscale array.
        Returns: (array, original_dtype, original_mode)
        """
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        with Image.open(path) as im:
            im = _ensure_grayscale(im)
            mode = im.mode
            if mode == "I;16":
                arr = np.array(im, dtype=np.uint16)
                return arr, arr.dtype, mode
            else:
                arr = np.array(im.convert("L"), dtype=np.uint8)
                return arr, arr.dtype, "L"

    # ---- Processing ----
    # ---- Optional pre-processing: Gaussian blur on original ----
    def apply_gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply a Gaussian blur to a 2D (grayscale) or 3D array using separable
        convolution with reflect padding. Preserves dtype and value range.
        """
        if sigma <= 0:
            return image

        # Work in float64 for precision
        dtype = image.dtype
        lo, hi = _dynamic_range_for_dtype(dtype)
        img = image.astype(np.float64, copy=False)

        # Build 1D Gaussian kernel
        radius = max(1, int(math.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= k.sum()

        def _convolve1d(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
            pad = len(kernel) // 2
            a_pad = np.pad(a, [(pad, pad) if ax == axis else (0, 0) for ax in range(a.ndim)], mode="reflect")
            # Roll axis to end for easier slicing
            a_roll = np.moveaxis(a_pad, axis, -1)
            out = np.empty_like(np.moveaxis(a, axis, -1), dtype=np.float64)
            # Convolution by sliding window dot-product
            for i in range(out.shape[-1]):
                sl = a_roll[..., i:i + len(kernel)]
                out[..., i] = np.tensordot(sl, kernel, axes=([-1], [0]))
            # Move axis back
            return np.moveaxis(out, -1, axis)

        # Separable convolution: horizontal then vertical
        blurred = _convolve1d(img, k, axis=1)  # X
        blurred = _convolve1d(blurred, k, axis=0)  # Y

        # Clip and cast back
        blurred = np.clip(blurred, lo, hi).astype(dtype)
        return blurred

    def add_noise(self, arr: np.ndarray) -> np.ndarray:
        """Add Gaussian noise using variance and amplitude, preserving dtype & range."""
        dtype = arr.dtype
        lo, hi = _dynamic_range_for_dtype(dtype)

        arr_f = arr.astype(np.float64)
        sigma = math.sqrt(max(self.noise_cfg.variance, 0.0))
        noise = self._rng.normal(loc=self.noise_cfg.mean, scale=sigma, size=arr.shape)
        noise *= self.noise_cfg.amplitude

        noisy = arr_f + noise
        noisy = np.clip(noisy, lo, hi)
        return noisy.astype(dtype)

    # ---- Output naming (single/list modes) ----
    def derive_output_name_single(self, input_path: Path) -> Path:
        """Original single-image naming rule (kept as-is to preserve behavior)."""
        stem = input_path.stem
        ext = input_path.suffix  # includes leading dot
        parts = [stem, 'noisy']
        out_stem = "_".join(parts)
        return input_path.with_name(out_stem + ext)

    # ---- Output naming (folder mode) ----
    def derive_output_name_folder(self, input_path: Path, out_dir: Path, dtype: np.dtype) -> Path:
        """Folder-mode rule: {input_folder}/noisy/{basename}{ext}, possibly switching
        to .png for 16-bit if preserve_format is False.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / input_path.name  # same basename
        return self._maybe_adjust_extension(out_path, dtype)

    def _resolve_collision(self, path: Path) -> Path:
        if self.overwrite or not path.exists():
            return path
        i = 2
        while True:
            new_path = path.with_name(f"{path.stem}_v{i}{path.suffix}")
            if not new_path.exists():
                return new_path
            i += 1

    def _maybe_adjust_extension(self, out_path: Path, dtype: np.dtype) -> Path:
        """Ensure we can save with the desired bit depth. If ext doesn't support it,
        switch to PNG unless preserve_format=True (then we'll downcast if necessary)."""
        ext = out_path.suffix.lower()
        if np.issubdtype(dtype, np.uint16):
            if ext in {".png", ".tif", ".tiff"}:
                return out_path
            if self.preserve_format:
                return out_path
            return out_path.with_suffix(".png")
        return out_path

    # ---- Save ----
    def save_image(self, out_path: Path, arr: np.ndarray, dtype: np.dtype) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if np.issubdtype(dtype, np.uint16):
            pil_mode = "I;16"
            im = Image.fromarray(arr.astype(np.uint16), mode=pil_mode)
        else:
            pil_mode = "L"
            im = Image.fromarray(arr.astype(np.uint8), mode=pil_mode)
        im.save(out_path)
        return out_path

    # ---- Debug viz ----
    def show_debug(self, original: np.ndarray, noisy: np.ndarray, masked: Optional[np.ndarray] = None, title: str = "") -> None:
        if not self.debug:
            return
        if plt is None:
            print("matplotlib not available; skipping debug visualization.")
            return
        # Decide layout: 2 panels if no masked; otherwise 3 panels
        if masked is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            views = [(original, "Original"), (noisy, "Noisy (Gaussian)")]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            views = [(original, "Original"), (noisy, "Noisy"), (masked, "Masked (circular)")]
        for ax, (img, ttl) in zip(axes, views):
            ax.imshow(img, cmap="gray")
            ax.set_title(ttl)
            ax.axis("off")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    # ---- Orchestration for ONE image ----
    @staticmethod
    def apply_circular_mask(image_array):
        """
        Applies a circular mask to a square image array.

        Parameters:
        -----------
        image_array : numpy.ndarray
            The input image array to mask.

        Returns:
        --------
        masked_array : numpy.ndarray
            The masked image array.
        mask : numpy.ndarray
            The mask applied to the image array.
        """
        assert image_array.shape[0] == image_array.shape[1], "Image must be square (nXn shape)"
        size = image_array.shape[0]
        center = size // 2
        radius = center

        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
        mask = dist_from_center <= radius

        if image_array.ndim == 2:  # Grayscale image
            masked_array = np.zeros_like(image_array)
            masked_array[mask] = image_array[mask]
        elif image_array.ndim == 3:  # Color image
            masked_array = np.zeros_like(image_array)
            for i in range(image_array.shape[2]):  # Apply mask to each channel
                masked_array[:, :, i] = np.where(mask, image_array[:, :, i], 0)

        return masked_array

    def process_one(self, input_path: Path, *, mode: str, folder_out_dir: Optional[Path] = None,
                    visualize: bool = True) -> Path:
        """Process a single image path according to the specified mode.

        Args:
            input_path: path to the source image
            mode: one of {"single", "list", "folder"}
            folder_out_dir: required in folder mode to compute the output path
            visualize: whether to show matplotlib debug for this image (if debug=True)
        """
        arr, dtype, _mode = self.load_image(input_path)
        base = self.apply_gaussian_blur(arr, self.blur_sigma) if self.blur_enabled else arr
        noisy = self.add_noise(base)
        #noisy = self.add_noise(arr)

        # Apply circular mask AFTER noise injection
        masked = self.apply_circular_mask(noisy)

        if mode in {"single", "list"}:
            out_path = self.derive_output_name_single(input_path)
            out_path = self._maybe_adjust_extension(out_path, dtype)
        elif mode == "folder":
            if folder_out_dir is None:
                raise ValueError("folder_out_dir must be provided in folder mode")
            out_path = self.derive_output_name_folder(input_path, folder_out_dir, dtype)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out_path = self._resolve_collision(out_path)
        saved_path = self.save_image(out_path, masked, dtype)

        if visualize:
            self.show_debug(arr, noisy, masked=masked, title=f"{input_path.name}")

        print(f"Saved masked noisy image to: {saved_path}")
        return saved_path


# -----------------
# Input determination
# -----------------
class InputResolver:
    """Resolve CONFIG["input_path"] into an execution mode and list of paths.

    Modes:
        - "single": one image file
        - "list": a list of image files
        - "folder": a directory to be scanned for images (non-recursive)
    """

    def __init__(self, raw_input: Union[str, Path, Iterable[Union[str, Path]]], image_exts: Iterable[str]):
        self.image_exts = {str(ext).lower() for ext in image_exts}
        self.mode: str
        self.paths: List[Path] = []
        self.folder: Optional[Path] = None

        self._resolve(raw_input)

    def _is_image(self, p: Path) -> bool:
        return p.suffix.lower() in self.image_exts

    def _resolve(self, raw_input: Union[str, Path, Iterable[Union[str, Path]]]):
        # Case 1: list/tuple/iterable → list mode
        if isinstance(raw_input, (list, tuple)):
            paths = [Path(x).expanduser() for x in raw_input]
            good: List[Path] = []
            for p in paths:
                if p.exists() and p.is_file() and self._is_image(p):
                    good.append(p)
                else:
                    print(f"[WARN] Skipping non-image or missing path: {p}")
            if not good:
                raise FileNotFoundError("No valid image files found in the provided list.")
            self.mode = "list"
            self.paths = good
            self.folder = None
            return

        # Case 2: string/Path → could be file or folder
        p = Path(raw_input).expanduser()
        if p.exists() and p.is_file():
            if not self._is_image(p):
                raise ValueError(f"Provided file is not a supported image: {p}")
            self.mode = "single"
            self.paths = [p]
            self.folder = None
            return
        if p.exists() and p.is_dir():
            # Folder: gather images (non-recursive)
            imgs = [x for x in p.iterdir() if x.is_file() and self._is_image(x)]
            if not imgs:
                raise FileNotFoundError(f"No images found in folder: {p}")
            self.mode = "folder"
            self.paths = sorted(imgs)
            self.folder = p
            return

        raise FileNotFoundError(f"input_path not found: {p}")


# -------------------
# Batch Orchestration
# -------------------
class BatchOrchestrator:
    """Coordinates batch execution across single/list/folder inputs.

    - Shares one RNG across all images for reproducibility when `seed` is set.
    - Applies visualization policy: in batch modes, only the first image is shown
      by default (configurable).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(seed)
        self.adder = GaussianNoiseAdder(config, rng=self.rng)

        raw_input = config.get("input_path")
        if raw_input is None:
            raise ValueError("CONFIG['input_path'] must be provided.")
        image_exts = config.get("folder_image_exts", [".bmp", ".png", ".jpg", ".jpeg"])
        self.resolver = InputResolver(raw_input, image_exts)
    def run(self) -> List[Path]:
        import os

        mode = self.resolver.mode
        paths = self.resolver.paths

        results: List[Path] = []
        show_first_only = self.adder.debug_batch_first_only and (mode in {"list", "folder"})
        first_shown = False

        folder_out_dir: Optional[Path] = None
        if mode == "folder":
            folder_out_dir = self.resolver.folder / "noisy"

        # --- New: single sink folder option ---
        output_folder_cfg = self.config.get("outputFolder", None)
        output_base_dir: Optional[Path] = None
        if output_folder_cfg:
            output_base_dir = Path(output_folder_cfg).expanduser()
            print(f"[WARN] `outputFolder` is set; all outputs will be written to: {output_base_dir}")
            print("       This overrides per-file destinations (single/list/folder modes).")

        # Determine a relative-root for name derivation when outputFolder is set
        relative_root: Optional[Path] = None
        if output_base_dir is not None:
            try:
                if mode == "folder":
                    relative_root = self.resolver.folder
                else:
                    # common ancestor across all input files (list or single)
                    strs = [str(p) for p in paths]
                    common = os.path.commonpath(strs) if len(strs) > 1 else str(paths[0].parent)
                    relative_root = Path(common)
            except Exception:
                relative_root = None  # fall back to basenames only

        for idx, p in enumerate(paths):
            visualize = False
            if self.adder.debug:
                if mode == "single":
                    visualize = True
                elif show_first_only:
                    visualize = (not first_shown)
                else:
                    visualize = True

            if output_base_dir is not None:
                # Route everything to the single sink, with unique filename scheme
                arr, dtype, _ = self.adder.load_image(p)
                base = self.adder.apply_gaussian_blur(arr, self.adder.blur_sigma) if self.adder.blur_enabled else arr
                noisy = self.adder.add_noise(base)
                masked = self.adder.apply_circular_mask(noisy)

                out_path = self.adder.derive_output_name_to_folder(
                    input_path=p,
                    out_dir=output_base_dir,
                    relative_root=relative_root,
                    dtype=dtype
                )
                saved_path = self.adder.save_image(out_path, masked, dtype)

                if visualize:
                    self.adder.show_debug(arr, noisy, masked=masked, title=f"{p.name}")
            else:
                # Original behavior preserved (single/list/folder modes)
                saved_path = self.adder.process_one(
                    p,
                    mode=mode,
                    folder_out_dir=folder_out_dir,
                    visualize=visualize,
                )

            results.append(saved_path)
            if visualize:
                first_shown = True

        print(f"Processed {len(results)} image(s) in mode='{mode}'.")
        return results




# -------------
# Entry point
# -------------
if __name__ == "__main__":
    # Define all configuration HERE (no global CONFIG at top of file)
    CONFIG = {
        # input_path can be:
        # 1) str/Path to a single image
        # 2) list[str|Path] of image files
        # 3) str/Path to a folder containing images
        "input_path": r"testData/groundTruth",  # <-- edit me
        "input_path": [
            r"C:\Users\kvman\PycharmProjects\kikuchiBandAnalyzer\testData\groundTruth\0_0_0_0pctStrain_460x460.bmp",
            r"C:\Users\kvman\PycharmProjects\kikuchiBandAnalyzer\testData\groundTruth\0_45_0_0pctStrain_460x460.bmp"
            #r"testData/groundTruth/0_45_0_0pctStrain_460x460.bmp",
            ], # <-- edit me
       # "input_path": r"E:\Amrutha\accuracy_testing\trainB",  # <-- edit me

        "outputFolder": r"testData/noisyImages_2",  # OPTIONAL; remove or None to keep old behavior

        "noise": {
            "type": "gaussian",     # currently only 'gaussian' supported
            "variance": 400.0,       # variance of base Gaussian (sigma^2). Example: 400 -> sigma=20
            "amplitude": 10.0,       # scalar multiplier applied to the Gaussian sample
            "mean": 0.0,             # mean of Gaussian (typically 0)
        },
        "blur": {
            "enabled": True,  # set to False to skip blurring
            "sigma": 1  # standard deviation in pixels
        },
        "seed": 42,                   # optional: int for reproducible noise; or set to None/omit
        "overwrite": True,            # if False, will append _v2, _v3, ... when file exists
        "preserve_format": False,     # if True, force original extension; may downcast if unsupported
        "debug": True,                # if True, show original & noisy images using matplotlib

        # Batch visualization control: in list/folder mode, show only the first image by default
        "debug_batch_first_only": True,

        # Folder discovery: which extensions to treat as images (case-insensitive)
        "folder_image_exts": [".bmp", ".png", ".jpg", ".jpeg"],
    }

    orchestrator = BatchOrchestrator(CONFIG)
    orchestrator.run()
