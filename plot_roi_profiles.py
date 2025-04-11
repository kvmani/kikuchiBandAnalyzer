import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

def stacked_horizontal_profiles(
        image_path: str,
        guide_colors: tuple = ('r', 'g', 'b'),
        line_width: float = 3.0,
        text_offset: int = 5,        # ← vertical offset (pixels) for labels on the image
):
    """
    Image + stacked intensity profiles with coloured, numbered guide lines.

    * Guide‑line heights: 90 %, 50 %, 10 % of the image (top → bottom).
    * Labels (1,2,3) are printed *above* the guide line by `text_offset` pixels.
    * In the profile stack, numbering is bottom‑to‑top: (1) bottom, (3) top.
    """

    # 1 ── load image ──────────────────────────────────────────────────
    img = Image.open(image_path).convert("L")
    I   = np.asarray(img, dtype=np.float32)
    h, w = I.shape

    rel_pos = [0.9, 0.5, 0.1]                 # top, mid, bottom
    y_positions = [int(p * (h - 1)) for p in rel_pos]
    num_profiles = len(y_positions)

    # 2 ── figure skeleton ─────────────────────────────────────────────
    fig = plt.figure(figsize=(6, 2 + 1.2 * num_profiles))
    gs  = GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.25)

    # (a) image with guide lines
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(I, cmap='gray', origin='upper')

    for idx, (y, c) in enumerate(zip(y_positions, guide_colors), 1):
        ax_img.axhline(y, color=c, lw=line_width)
        ax_img.text(0.03, y - text_offset,            # ← shifted upward
                    f"{4-idx}", color='w', fontsize=14,
                    va='bottom', ha='left', fontweight='bold')

    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title("(a)", loc="left", fontweight="bold")

    # (b) stacked profiles
    gs_profiles = gs[1].subgridspec(num_profiles, 1, hspace=0.05)
    x_pixels = np.arange(w)

    for row, (y, c, gs_row) in enumerate(zip(y_positions,
                                             guide_colors[::-1],
                                             gs_profiles)):
        # label should be bottom‑to‑top → compute reversed index
        label_idx = num_profiles - row            # 3→1, 2→2, 1→3

        ax = fig.add_subplot(gs_row)
        ax.plot(x_pixels, I[y, :], color=c, lw=1.5)

        # cosmetics
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(0, w - 1)
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.set_ylabel("Intensity", fontsize=12)

        if row < num_profiles - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("pixels →", fontsize=12)

        # subplot number
        ax.text(0.02, 0.85, f"({label_idx})", transform=ax.transAxes,
                fontsize=12, fontweight='bold')

    fig.text(0.55, 0.97, "(b) Intensity profiles", ha='center',
             va='top', fontweight="bold")

    plt.tight_layout()
    plt.show()

# Example
stacked_horizontal_profiles(r"roi.png")
