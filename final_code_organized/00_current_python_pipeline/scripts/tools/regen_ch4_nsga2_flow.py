"""Standalone regenerator for §4.3.1's NSGA-II flow diagram (ch4_nsga2_flow.png).

The previous version had a feedback arrow that visually clipped through the
"Init population" box. This version uses an explicit two-row layout so all
arrows route cleanly through whitespace.

Run from project root:
    python scripts/tools/regen_ch4_nsga2_flow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"


def add_box(ax, x, y, w, h, text, fc="#eef2f8", ec="#3b5e93"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, lw=1.4,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=10,
            color="#1f3b65", linespacing=1.25)


def add_arrow(ax, p, q, color="#3b5e93"):
    arr = FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=14,
        color=color, lw=1.4, connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    box_w = 0.135
    box_h = 0.22
    y_top = 0.62

    top_steps = [
        (0.020, "Target\nspectrum"),
        (0.180, "Init\npopulation"),
        (0.340, "Surrogate\ninference"),
        (0.500, "Peak\nextraction\n+ fitness"),
        (0.660, "NSGA-II\nranking"),
        (0.820, "Crossover /\nmutation /\nrepair"),
    ]
    for x, label in top_steps:
        add_box(ax, x, y_top, box_w, box_h, label)

    for i in range(len(top_steps) - 1):
        x_from = top_steps[i][0] + box_w
        x_to = top_steps[i + 1][0]
        add_arrow(ax, (x_from, y_top + box_h / 2), (x_to, y_top + box_h / 2))

    bottom_x = 0.500
    bottom_y = 0.16
    add_box(ax, bottom_x, bottom_y, box_w, box_h,
            "Termination\n+ candidate\nscreening",
            fc="#fdf3e7", ec="#a86b1f")

    # arrow: top-right last step → bottom box (going down through whitespace)
    last_x = top_steps[-1][0] + box_w / 2
    add_arrow(ax, (last_x, y_top), (bottom_x + box_w, bottom_y + box_h / 2))

    # arrow: bottom box → init population (feedback loop, routed cleanly through whitespace)
    init_x = top_steps[1][0] + box_w / 2
    # route: bottom-left of bottom box → down → left → up → init box bottom
    p1 = (bottom_x, bottom_y + box_h / 2)
    p2 = (init_x, bottom_y + box_h / 2)
    p3 = (init_x, y_top)
    arr1 = FancyArrowPatch(p1, p2, arrowstyle="-",
                           mutation_scale=12, color="#a86b1f", lw=1.4)
    arr2 = FancyArrowPatch(p2, p3, arrowstyle="-|>",
                           mutation_scale=14, color="#a86b1f", lw=1.4)
    ax.add_patch(arr1)
    ax.add_patch(arr2)
    ax.text(init_x - 0.01, (bottom_y + box_h / 2 + y_top) / 2,
            "next generation",
            ha="right", va="center", fontsize=9, color="#a86b1f")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "ch4_nsga2_flow.png"
    fig.savefig(out, bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
