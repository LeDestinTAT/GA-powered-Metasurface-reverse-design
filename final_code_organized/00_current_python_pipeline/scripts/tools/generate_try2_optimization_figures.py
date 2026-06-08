from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRY2_ROOT = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def lambda_um(n: int) -> np.ndarray:
    return np.linspace(3.0, 12.0, n, dtype=np.float32)


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def draw_pattern(ax: plt.Axes, pattern: np.ndarray, title: str) -> None:
    ax.imshow(pattern, cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def draw_spectrum(
    ax: plt.Axes,
    spectrum: np.ndarray,
    report: dict,
    title: str,
    show_ylabel: bool = True,
) -> None:
    lam = lambda_um(len(spectrum))
    ax.plot(lam, spectrum, color="#2f6f9f", lw=2.0, label="Predicted A")
    ax.axvline(TARGET_POS, color="#c92a2a", ls="--", lw=1.4, label="Target pos.")
    ax.axhline(TARGET_AMP, color="#3b8b3b", ls=":", lw=1.4, label="Target amp.")
    ax.scatter(
        [float(report["main_pos"])],
        [float(report["main_amp"])],
        s=38,
        color="#f28e2b",
        zorder=4,
        label="Main peak",
    )
    ax.set_xlim(3.0, 12.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Wavelength (um)")
    if show_ylabel:
        ax.set_ylabel("Absorption")
    ax.set_title(title)


def generate_closed_loop() -> None:
    report = load_json(TRY2_ROOT / "best_report.json")
    pattern = np.load(TRY2_ROOT / "best_pattern.npy").astype(np.float32)
    spectrum = np.load(TRY2_ROOT / "best_spectrum.npy").astype(np.float32)

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.88, 1.55], wspace=0.28)
    draw_pattern(fig.add_subplot(gs[0]), pattern, "Selected 11x11 pattern")

    ax = fig.add_subplot(gs[1])
    draw_spectrum(ax, spectrum, report, "Try2 representative design")
    ax.legend(frameon=False, loc="lower right")
    summary = (
        f"main peak: {report['main_pos']:.2f} um\n"
        f"peak height: {report['main_amp']:.3f}\n"
        f"height error: {abs(report['main_amp'] - TARGET_AMP):.3f}\n"
        f"side-peak excess: {report['spur_excess']:.3f}"
    )
    ax.text(
        0.03,
        0.96,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d0d0d0", "alpha": 0.92},
    )
    savefig(fig, "ch5_closed_loop.png")


def report_by_rank() -> dict[int, dict]:
    return {int(row["rank"]): row for row in load_json(TRY2_ROOT / "pareto_summary.json")}


def generate_multi_solution() -> None:
    reports = report_by_rank()
    candidates = [
        (3, "Selected"),
        (0, "Exact position"),
        (4, "Higher peak"),
        (40, "No side peak"),
    ]
    fig = plt.figure(figsize=(13.0, 6.4))
    outer = gridspec.GridSpec(2, 4, hspace=0.20, wspace=0.18)
    for idx, (rank, title) in enumerate(candidates):
        report = reports[rank]
        pattern_path = TRY2_ROOT / f"pattern_{rank:03d}.npy"
        spectrum_path = TRY2_ROOT / f"spectrum_{rank:03d}.npy"
        pattern = np.load(pattern_path).astype(np.float32)
        spectrum = np.load(spectrum_path).astype(np.float32)

        draw_pattern(fig.add_subplot(outer[0, idx]), pattern, f"{title} (rank {rank})")
        ax = fig.add_subplot(outer[1, idx])
        draw_spectrum(
            ax,
            spectrum,
            report,
            f"{report['main_pos']:.2f} um, A={report['main_amp']:.3f}",
            show_ylabel=idx == 0,
        )
        ax.text(
            0.03,
            0.94,
            f"spur ratio {report['spur_ratio']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d8d8d8", "alpha": 0.9},
        )
    savefig(fig, "ch5_multi_solution.png")


def generate_convergence() -> None:
    progress = load_json(TRY2_ROOT / "progress.json")
    gen = np.array([row["gen"] for row in progress], dtype=float)
    obj_pos = np.array([row["obj1_best"] for row in progress], dtype=float)
    obj_amp = np.array([row["obj2_best"] for row in progress], dtype=float)
    obj_spur = np.array([row["obj3_best"] for row in progress], dtype=float)
    front = np.array([row["front_size"] for row in progress], dtype=float)
    cache = np.array([row["cache_size"] for row in progress], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0))
    axes[0].plot(gen, obj_pos, lw=2.0, color="#4e79a7", label="Peak-position objective")
    axes[0].plot(gen, obj_amp, lw=2.0, color="#f28e2b", label="Peak-height objective")
    axes[0].plot(gen, obj_spur, lw=2.0, color="#59a14f", label="Side-peak objective")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best objective value")
    axes[0].set_title("Objective convergence")
    axes[0].legend(frameon=False)

    ax_count = axes[1].twinx()
    axes[1].plot(gen, front, lw=2.0, color="#4e79a7", label="Pareto front size")
    ax_count.plot(gen, cache, lw=2.0, color="#e15759", label="Evaluated unique patterns")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Pareto front size", color="#4e79a7")
    ax_count.set_ylabel("Evaluated unique patterns", color="#e15759")
    axes[1].tick_params(axis="y", labelcolor="#4e79a7")
    ax_count.tick_params(axis="y", labelcolor="#e15759")
    axes[1].set_title("Population statistics")
    lines = axes[1].get_lines() + ax_count.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    savefig(fig, "ch5_optimization_convergence.png")


def main() -> None:
    generate_closed_loop()
    generate_multi_solution()
    generate_convergence()


if __name__ == "__main__":
    main()
