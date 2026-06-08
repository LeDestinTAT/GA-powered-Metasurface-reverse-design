"""Standalone regenerator for §4.1.2's encoding figure (ch4_encoding_a.png + ch4_encoding_b.png).

Replaces the original single-panel ch4_encoding.png with two cleaner subfigures
laid out by the LaTeX subcaption environment:
    (a) the 11x11 binary pattern only
    (b) the row-major flattening into a 121-bit chromosome,
        with the gene-to-pixel mapping shown explicitly

Run from project root:
    python scripts/tools/regen_ch4_encoding.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "font.size": 11,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def make_pattern(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pattern = (rng.random((11, 11)) < 0.40).astype(np.float32)
    # add a few clear cross-shaped clusters for visual readability
    pattern[3:5, 3:8] = 1.0
    pattern[2:8, 5:6] = 1.0
    pattern[7:9, 2:4] = 1.0
    pattern[1:3, 8:10] = 0.0
    return pattern


def panel_a(out: Path) -> None:
    pat = make_pattern()
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(pat, cmap="gray_r", interpolation="nearest")
    for k in range(12):
        ax.axhline(k - 0.5, color="#bdbdbd", lw=0.4)
        ax.axvline(k - 0.5, color="#bdbdbd", lw=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(10.5, -0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def panel_b(out: Path) -> None:
    pat = make_pattern().reshape(-1).astype(int)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    n = 121
    cell_w = 1.0
    cell_h = 0.55
    n_show = 32  # show first 28 bits then ellipsis then last 3
    head = list(range(28))
    tail = list(range(118, 121))
    indices = head + tail

    for slot, gene_idx in enumerate(indices):
        x = slot * cell_w
        bit = pat[gene_idx]
        face = "#222222" if bit == 1 else "#ffffff"
        edge = "#666666"
        ax.add_patch(Rectangle((x, 0), cell_w, cell_h,
                               facecolor=face, edgecolor=edge, lw=0.8))
        ax.text(x + cell_w / 2, cell_h / 2, str(bit),
                ha="center", va="center",
                color="#ffffff" if bit == 1 else "#222222",
                fontsize=8.5)
        if gene_idx in (0, 5, 27, 118, 120):
            ax.text(x + cell_w / 2, -0.35, f"$c_{{{gene_idx + 1}}}$",
                    ha="center", va="top", fontsize=9)

    # ellipsis between head and tail
    ax.text(28 * cell_w + 1.5, cell_h / 2, "$\\cdots$",
            ha="center", va="center", fontsize=18)

    # axis bounds and chromosome label above
    total_w = (len(indices) + 3) * cell_w
    ax.set_xlim(-0.6, total_w)
    ax.set_ylim(-1.4, 1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(total_w / 2, 1.2,
            "染色体  $\\mathbf{c}=[c_1, c_2, \\cdots, c_{121}]$,  $c_i \\in \\{0,1\\}$",
            ha="center", va="center", fontsize=11)
    # arrow + label
    arrow = FancyArrowPatch(
        (total_w / 2 - 4, -0.85), (total_w / 2 + 4, -0.85),
        arrowstyle="-|>", mutation_scale=14, color="#444444", lw=1.2,
    )
    ax.add_patch(arrow)
    ax.text(total_w / 2, -1.20,
            "将 $11\\times 11$ 图案按行优先展平",
            ha="center", va="top", fontsize=10, color="#444444")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_a = FIG_DIR / "ch4_encoding_a.png"
    out_b = FIG_DIR / "ch4_encoding_b.png"
    panel_a(out_a)
    panel_b(out_b)
    print(f"wrote {out_a}")
    print(f"wrote {out_b}")


if __name__ == "__main__":
    main()
