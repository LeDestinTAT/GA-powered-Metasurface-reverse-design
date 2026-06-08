"""Standalone regenerator for the §4.4.2 Pareto figure (ch4_pareto.png).

Run this once to refresh figures/generated/ch4_pareto.png with the redesigned
Pareto + representative-candidates plot. Only depends on numpy + matplotlib;
does NOT need torch, tensorboard, or the trained surrogate, unlike the full
generate_paper_figures.py.

Usage (from the project root):
    python scripts/tools/regen_ch4_pareto.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"

TARGET_POS = 8.5
TARGET_AMP = 0.95


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)


def main() -> None:
    pareto = json.loads((RUN_DIR / "pareto_summary.json").read_text(encoding="utf-8"))

    pos = np.array([float(r["main_pos"]) for r in pareto])
    amp = np.array([float(r["main_amp"]) for r in pareto])
    spur = np.array([float(r["spur_excess"]) for r in pareto])
    npeaks = np.array([int(r["n_peaks"]) for r in pareto])

    height_err = np.abs(amp - TARGET_AMP)
    pos_dev = np.abs(pos - TARGET_POS)

    chosen = [
        (0,  "A: on-target, higher peak",  "#c92a2a"),
        (3,  "B: on-target, clean spectrum", "#0a7c5a"),
        (4,  "C: higher peak, peak off",   "#d97f1c"),
        (40, "D: clean spectrum, peak shifted", "#3a5fa3"),
    ]

    fig = plt.figure(figsize=(12.6, 5.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        height_err,
        spur,
        c=pos_dev,
        cmap="viridis",
        s=44 + 14 * npeaks,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )

    for rank, label, color in chosen:
        r = pareto[rank]
        x = abs(float(r["main_amp"]) - TARGET_AMP)
        y = float(r["spur_excess"])
        ax.scatter([x], [y], s=170, facecolor="none",
                   edgecolor=color, lw=2.2, zorder=4)
        tag = label.split(":")[0]
        ax.annotate(tag, xy=(x, y), xytext=(8, 6),
                    textcoords="offset points",
                    color=color, fontweight="bold", fontsize=11)

    ax.set_xlabel("Main-peak height error  |A - 0.95|")
    ax.set_ylabel("Side-peak excess  (spur_excess)")
    ax.set_title("Pareto front of the 8.5 um single-peak target  (42 non-dominated candidates)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Main-peak position deviation  |pos - 8.5|  (um)")

    sub = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.32)
    for i, (rank, label, color) in enumerate(chosen):
        axp = fig.add_subplot(sub[i, 0])
        pat = np.load(RUN_DIR / f"pattern_{rank:03d}.npy").astype(np.float32)
        axp.imshow(pat, cmap="gray_r", interpolation="nearest")
        axp.set_xticks([])
        axp.set_yticks([])
        for spine in axp.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8)
        axp.set_title(label, fontsize=9.5, color=color, pad=3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "ch4_pareto.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
