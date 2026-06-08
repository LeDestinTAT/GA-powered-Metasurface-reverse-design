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
OUT = FIG_DIR / "ch5_multi_solution_slide.png"

TARGET_POS = 8.5
TARGET_AMP = 0.95

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    }
)


def main() -> None:
    pareto = json.loads((RUN_DIR / "pareto_summary.json").read_text(encoding="utf-8"))
    chosen = [
        (3, "B 目标峰干净"),
        (0, "A 目标峰较高"),
        (4, "C 最高峰值"),
        (40, "D 谱形折中"),
    ]

    n_lambda = int(np.load(RUN_DIR / "spectrum_000.npy").shape[0])
    lambda_um = np.linspace(3.0, 12.0, n_lambda, dtype=np.float32)

    fig = plt.figure(figsize=(14.6, 4.0))
    outer = gridspec.GridSpec(2, 4, height_ratios=[0.95, 1.25], hspace=0.16, wspace=0.16)

    for idx, (rank, title) in enumerate(chosen):
        report = pareto[rank]
        pattern = np.load(RUN_DIR / f"pattern_{rank:03d}.npy").astype(np.float32)
        spectrum = np.load(RUN_DIR / f"spectrum_{rank:03d}.npy").astype(np.float32)

        axp = fig.add_subplot(outer[0, idx])
        axp.imshow(pattern, cmap="gray_r", interpolation="nearest")
        axp.set_xticks([])
        axp.set_yticks([])
        axp.set_title(title, fontsize=12, color="#18324a", pad=5)
        for spine in axp.spines.values():
            spine.set_visible(False)

        axs = fig.add_subplot(outer[1, idx])
        axs.plot(lambda_um, spectrum, color="#1f66a8", lw=1.8)
        axs.axvline(TARGET_POS, color="#d62728", ls="--", lw=1.1)
        axs.axhline(TARGET_AMP, color="#3b8b3b", ls=":", lw=1.0)
        axs.scatter([float(report["main_pos"])], [float(report["main_amp"])], s=30, color="#f28c28", zorder=4)
        axs.set_xlim(3.0, 12.0)
        axs.set_ylim(0.0, 1.02)
        axs.tick_params(labelsize=8)
        axs.set_xlabel("波长 (μm)", fontsize=9)
        if idx == 0:
            axs.set_ylabel("吸收率", fontsize=9)
        axs.set_title(f"峰位 {report['main_pos']:.2f} μm / 峰高 {report['main_amp']:.3f}", fontsize=9, pad=4)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
