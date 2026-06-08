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
OUT = FIG_DIR / "ch5_optimization_convergence_slide.png"

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
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def normalize_for_trend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def main() -> None:
    progress = json.loads((RUN_DIR / "progress.json").read_text(encoding="utf-8"))
    gen = np.array([row["gen"] for row in progress], dtype=float)
    obj1 = np.array([row["obj1_best"] for row in progress], dtype=float)
    obj2 = np.array([row["obj2_best"] for row in progress], dtype=float)
    obj3 = np.array([row["obj3_best"] for row in progress], dtype=float)
    front = np.array([row["front_size"] for row in progress], dtype=float)
    cache_size = np.array([row["cache_size"] for row in progress], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.18, 1.0]})

    ax = axes[0]
    ax.plot(gen, normalize_for_trend(obj1), lw=2.4, color="#1f66a8", label="峰位误差")
    ax.plot(gen, normalize_for_trend(obj2), lw=2.4, color="#f28c28", label="峰高误差")
    ax.plot(gen, normalize_for_trend(obj3), lw=2.4, color="#59a14f", label="杂峰惩罚")
    ax.set_title("目标误差收敛趋势")
    ax.set_xlabel("迭代代数")
    ax.set_ylabel("归一化误差")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False, loc="upper right")

    ax_left = axes[1]
    ax_right = ax_left.twinx()
    l1, = ax_left.plot(gen, front, lw=2.5, color="#1f66a8", label="Pareto 候选数")
    l2, = ax_right.plot(gen, cache_size, lw=2.3, color="#e15759", label="已评估结构数")
    ax_left.set_title("候选集与搜索规模")
    ax_left.set_xlabel("迭代代数")
    ax_left.set_ylabel("Pareto 候选数", color="#1f66a8")
    ax_right.set_ylabel("已评估结构数", color="#e15759")
    ax_left.tick_params(axis="y", labelcolor="#1f66a8")
    ax_right.tick_params(axis="y", labelcolor="#e15759")
    ax_left.legend(handles=[l1, l2], frameon=False, loc="lower right")

    fig.tight_layout(pad=0.8, w_pad=2.0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
