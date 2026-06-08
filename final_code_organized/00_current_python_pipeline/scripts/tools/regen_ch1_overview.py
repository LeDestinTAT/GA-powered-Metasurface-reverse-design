"""Standalone regenerator for the §1.4 research overview figure.

The figure is used in chapter 1 to give readers a visual map of the thesis:
structure encoding, full-wave simulation data, physics-constrained surrogate
modeling, and surrogate-driven inverse design.

Run from project root:
    python scripts/tools/regen_ch1_overview.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"
FONT_DIR = PROJECT_ROOT / "paper" / "fonts"


def load_font(name: str, size: float, weight: str = "normal") -> font_manager.FontProperties:
    return font_manager.FontProperties(
        fname=str(FONT_DIR / name),
        size=size,
        weight=weight,
    )


def add_box(ax, x, y, w, h, title, body, fc, ec, title_color, title_font, body_font):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.014,rounding_size=0.020",
        facecolor=fc, edgecolor=ec, lw=1.6,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h - 0.075, title,
        ha="center", va="top", fontproperties=title_font,
        color=title_color, linespacing=1.18,
    )
    ax.text(
        x + w / 2, y + h - 0.215, body,
        ha="center", va="top", fontproperties=body_font,
        color="#333333", linespacing=1.42,
    )


def add_arrow(ax, p, q, color="#5a5a5a"):
    arr = FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=18,
        color=color, lw=1.6,
    )
    ax.add_patch(arr)


def main() -> None:
    title_font = load_font("simhei.ttf", 16, "bold")
    box_title_font = load_font("simhei.ttf", 12.5, "bold")
    body_font = load_font("simsun.ttc", 10.5)
    note_font = load_font("simsun.ttc", 10.2)

    fig, ax = plt.subplots(figsize=(10.2, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    box_w = 0.205
    box_h = 0.58
    y = 0.22
    spacing = 0.035
    x0 = 0.03
    xs = [x0 + i * (box_w + spacing) for i in range(4)]

    stages = [
        {
            "title": "结构编码",
            "body": "11×11 二值图案\nMIM 层状结构\n批量建模脚本",
            "fc": "#eef2f8", "ec": "#3b5e93", "tc": "#1f3b65",
        },
        {
            "title": "全波仿真\n数据集",
            "body": "吸收谱与 S 参数\n复电磁场分布\nAu/SiO2 色散参数",
            "fc": "#fdf3e7", "ec": "#a86b1f", "tc": "#7c4f17",
        },
        {
            "title": "物理约束\n代理模型",
            "body": "FNO 全场预测\n场监督损失\n频域 Maxwell 残差",
            "fc": "#eaf3de", "ec": "#3b6d11", "tc": "#244509",
        },
        {
            "title": "目标谱\n逆向设计",
            "body": "NSGA-II 搜索\n多目标适应度\n候选结构复核",
            "fc": "#fcebeb", "ec": "#a32d2d", "tc": "#791f1f",
        },
    ]

    for x, st in zip(xs, stages):
        add_box(ax, x, y, box_w, box_h, st["title"], st["body"],
                st["fc"], st["ec"], st["tc"], box_title_font, body_font)

    for i in range(3):
        add_arrow(ax,
                  (xs[i] + box_w, y + box_h / 2),
                  (xs[i + 1], y + box_h / 2))

    ax.text(
        0.5, 0.93,
        "红外 MIM 超表面代理建模与逆向设计流程",
        ha="center", va="center", fontproperties=title_font, color="#222222",
    )

    ax.text(
        0.5, 0.075,
        "数据流：结构表示 → 全波仿真 → 物理增强代理模型 → 多目标结构优化",
        ha="center", va="center", fontproperties=note_font, color="#555555",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "ch1_overview.png"
    fig.savefig(out, bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
