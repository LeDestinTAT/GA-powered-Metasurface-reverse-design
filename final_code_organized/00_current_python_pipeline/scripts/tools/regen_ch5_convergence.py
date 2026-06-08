"""Standalone regenerator for the §5.5.1 convergence figure (ch5_optimization_convergence.png).

Replaces the previous version, which plotted only two of three objectives on a
shared y-axis (so one looked stuck at ~25 and the other was invisible near 0)
and showed front size dwarfed by cache size on the right panel.

The new version uses three separate objective panels (so each has its own scale,
and the obj2 panel uses semilog because peak-height error spans several orders
of magnitude), and a twin-axis population panel so the Pareto front size and the
unique-structures count are both visible. Data comes from the same converged run
as the closed-loop, multi-solution and Pareto figures.

Run from project root:
    python scripts/tools/regen_ch5_convergence.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"


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
    progress = json.loads((RUN_DIR / "progress.json").read_text(encoding="utf-8"))
    gen = np.array([row["gen"] for row in progress], dtype=float)
    obj1 = np.array([row["obj1_best"] for row in progress], dtype=float)
    obj2 = np.array([row["obj2_best"] for row in progress], dtype=float)
    obj3 = np.array([row["obj3_best"] for row in progress], dtype=float)
    front = np.array([row["front_size"] for row in progress], dtype=float)
    cache_size = np.array([row["cache_size"] for row in progress], dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.6))

    axes[0].plot(gen, obj1, lw=2.0, color="#4e79a7")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Peak-position penalty")
    axes[0].set_title("Peak position")

    axes[1].semilogy(gen, np.maximum(obj2, 1e-5), lw=2.0, color="#f28e2b")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Peak-height error")
    axes[1].set_title("Peak height (log)")

    axes[2].plot(gen, obj3, lw=2.0, color="#59a14f")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Side-peak excess")
    axes[2].set_title("Side-peak suppression")

    ax_left = axes[3]
    ax_right = ax_left.twinx()
    l1, = ax_left.plot(gen, front, lw=2.0, color="#4e79a7", label="Pareto front size")
    l2, = ax_right.plot(gen, cache_size, lw=2.0, color="#e15759",
                        label="Unique structures evaluated")
    ax_left.set_xlabel("Generation")
    ax_left.set_ylabel("Pareto front size", color="#4e79a7")
    ax_right.set_ylabel("Unique structures", color="#e15759")
    ax_left.tick_params(axis="y", labelcolor="#4e79a7")
    ax_right.tick_params(axis="y", labelcolor="#e15759")
    ax_left.set_title("Population diversity")
    ax_left.legend(handles=[l1, l2], frameon=False, loc="lower right", fontsize=8.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = FIG_DIR / "ch5_optimization_convergence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
