"""Standalone regenerator for §5.3.2's multi-solution figure (ch5_multi_solution.png).

Replaces the previous version, which pulled candidates from a different optimization
run (target 8 um, ranks 1-4) and showed flat spectra that did not match §5.3.1's
headline 8.5 um result. The new version pulls candidates from the same converged
run as the closed-loop figure (FNO/runs_peak_nsga2_v2) and picks four candidates
that span the Pareto trade-off corners.

Run from project root:
    python scripts/tools/regen_ch5_multi_solution.py
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
        "axes.titlesize": 10,
        "axes.labelsize": 10,
    }
)


def main() -> None:
    pareto = json.loads((RUN_DIR / "pareto_summary.json").read_text(encoding="utf-8"))

    chosen = [
        (3,  "B: on-target clean peak"),
        (0,  "A: on-target high peak"),
        (4,  "C: highest peak"),
        (40, "D: clean spectrum, peak at 8.05 um"),
    ]

    n_lambda = int(np.load(RUN_DIR / "spectrum_000.npy").shape[0])
    lambda_um = np.linspace(3.0, 12.0, n_lambda, dtype=np.float32)

    fig = plt.figure(figsize=(13.0, 6.6))
    outer = gridspec.GridSpec(2, 4, hspace=0.20, wspace=0.18)

    for idx, (rank, title) in enumerate(chosen):
        report = pareto[rank]
        pattern = np.load(RUN_DIR / f"pattern_{rank:03d}.npy").astype(np.float32)
        spectrum = np.load(RUN_DIR / f"spectrum_{rank:03d}.npy").astype(np.float32)

        axp = fig.add_subplot(outer[0, idx])
        axp.imshow(pattern, cmap="gray_r", interpolation="nearest")
        axp.set_xticks([])
        axp.set_yticks([])
        axp.set_title(title, fontsize=10)

        axs = fig.add_subplot(outer[1, idx])
        axs.plot(lambda_um, spectrum, color="#4e79a7", lw=1.8)
        axs.axvline(TARGET_POS, color="#d62728", ls="--", lw=1.2)
        axs.axhline(TARGET_AMP, color="#3b8b3b", ls=":", lw=1.0)
        axs.scatter(
            [float(report["main_pos"])],
            [float(report["main_amp"])],
            s=36, color="#f28e2b", zorder=4,
        )
        axs.set_xlim(3.0, 12.0)
        axs.set_ylim(0.0, 1.02)
        axs.set_xlabel("Wavelength (um)")
        axs.set_title(
            f"$\\lambda_p$={report['main_pos']:.2f}$\\,\\mu$m, "
            f"$A_p$={report['main_amp']:.3f}",
            fontsize=9.5,
        )
        if idx == 0:
            axs.set_ylabel("Predicted absorption")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "ch5_multi_solution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
