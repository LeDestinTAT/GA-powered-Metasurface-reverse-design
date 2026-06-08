"""Standalone regenerator for §5.3.1's representative design figure (ch5_closed_loop.png).

Single-panel layout showing the 8.5 um single-peak target candidate from the
converged optimization run (FNO/runs_peak_nsga2_v2). Matches the multi-solution
and Pareto figures so all of section 5.3 references the same run.

Run from project root:
    python scripts/tools/regen_ch5_closed_loop.py
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
    report = json.loads((RUN_DIR / "best_report.json").read_text(encoding="utf-8"))
    pattern = np.load(RUN_DIR / "best_pattern.npy").astype(np.float32)
    spectrum = np.load(RUN_DIR / "best_spectrum.npy").astype(np.float32)
    lambda_um = np.linspace(3.0, 12.0, int(spectrum.shape[0]), dtype=np.float32)

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.88, 1.55], wspace=0.28)

    axp = fig.add_subplot(gs[0])
    axp.imshow(pattern, cmap="gray_r", interpolation="nearest")
    axp.set_xticks([])
    axp.set_yticks([])
    axp.set_title("Selected 11x11 pattern")

    ax = fig.add_subplot(gs[1])
    ax.plot(lambda_um, spectrum, color="#2f6f9f", lw=2.0, label="Predicted A")
    ax.axvline(TARGET_POS, color="#c92a2a", ls="--", lw=1.4, label="Target pos.")
    ax.axhline(TARGET_AMP, color="#3b8b3b", ls=":", lw=1.4, label="Target amp.")
    ax.scatter(
        [float(report["main_pos"])],
        [float(report["main_amp"])],
        s=44, color="#f28e2b", zorder=4, label="Main peak",
    )
    ax.set_xlim(3.0, 12.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Absorption")
    ax.set_title("8.5 um single-peak representative design")
    ax.legend(frameon=False, loc="lower right")

    summary = (
        f"main peak: {report['main_pos']:.2f} um\n"
        f"peak height: {report['main_amp']:.3f}\n"
        f"height error: {abs(report['main_amp'] - TARGET_AMP):.3f}\n"
        f"side-peak excess: {report['spur_excess']:.3f}"
    )
    ax.text(
        0.03, 0.96, summary,
        transform=ax.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white",
              "ec": "#d0d0d0", "alpha": 0.92},
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "ch5_closed_loop.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
