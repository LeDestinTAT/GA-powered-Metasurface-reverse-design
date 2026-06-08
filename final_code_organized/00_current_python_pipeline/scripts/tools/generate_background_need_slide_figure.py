from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "paper" / "figures" / "generated" / "ch1_background_need_slide.png"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def add_box(ax, xy, wh, title, body, color, edge):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=2.2,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=18, color=edge, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=13.5, color="#18324a", linespacing=1.35)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.88, "红外 MIM 超表面吸收器设计需求", ha="center", va="center", fontsize=20, color="#18324a", fontweight="bold")

    add_box(ax, (0.08, 0.55), (0.24, 0.20), "应用需求", "红外探测\n热辐射调控\n选择性吸收", "#edf5fb", "#1f66a8")
    add_box(ax, (0.38, 0.55), (0.24, 0.20), "MIM 结构", "金属-介质-金属\n局域共振\n光谱可调", "#fff4e8", "#f28c28")
    add_box(ax, (0.68, 0.55), (0.24, 0.20), "设计瓶颈", "离散图案空间大\n仿真成本高\n参数扫描困难", "#eef8eb", "#3b7a2a")

    for x0, x1 in [(0.32, 0.38), (0.62, 0.68)]:
        ax.annotate("", xy=(x1 - 0.01, 0.65), xytext=(x0 + 0.01, 0.65), arrowprops=dict(arrowstyle="-|>", lw=2, color="#5c6f7f"))

    ax.text(0.5, 0.30, "核心问题", ha="center", va="center", fontsize=16, color="#f28c28", fontweight="bold")
    ax.text(
        0.5,
        0.21,
        "如何把高成本仿真结果转化为可复用、可快速筛选的设计依据？",
        ha="center",
        va="center",
        fontsize=17,
        color="#0e4b52",
        fontweight="bold",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
