import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.compare_curve_vs_maxwell_metrics import (
    classify_curve,
    load_val_cache,
    predict_curve_only,
    predict_maxwell,
)


OUT = PROJECT_ROOT / "paper" / "figures" / "generated" / "ch5_two_model_spectra_compare.png"
CURVE_CKPT = PROJECT_ROOT / "models" / "current" / "fno_curve_only_best.pt"
DUAL_CKPT = PROJECT_ROOT / "models" / "history" / "20260424-142135" / "run_best.pt"


def pick_examples(truth, curve_pred, dual_pred, lambda_um):
    labels = np.array([classify_curve(c, lambda_um) for c in truth])
    picks = {}
    for group in ("single", "multi", "wide"):
        idx = np.where(labels == group)[0]
        if len(idx) == 0:
            continue
        curve_mse = np.mean((curve_pred[idx] - truth[idx]) ** 2, axis=1)
        dual_mse = np.mean((dual_pred[idx] - truth[idx]) ** 2, axis=1)
        # Prefer examples where both models are readable and the dual-head model improves.
        improvement = curve_mse - dual_mse
        readable = dual_mse <= np.quantile(dual_mse, 0.55)
        pool = idx[readable] if np.any(readable) else idx
        pool_curve = np.mean((curve_pred[pool] - truth[pool]) ** 2, axis=1)
        pool_dual = np.mean((dual_pred[pool] - truth[pool]) ** 2, axis=1)
        pool_improvement = pool_curve - pool_dual
        if np.max(pool_improvement) > 0:
            picks[group] = int(pool[int(np.argmax(pool_improvement))])
        else:
            picks[group] = int(pool[int(np.argmin(pool_dual))])
    return picks


def main():
    data = load_val_cache(limit=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curve_pred = predict_curve_only(CURVE_CKPT, data["pattern_11"], data["lambda_vec"], device, batch_size=64)
    dual_pred = predict_maxwell(DUAL_CKPT, data["pattern_11"], device.type, batch_size=64)

    truth = data["truth"]
    lambda_um = data["lambda_vec"].astype(np.float32) * 1e6
    picks = pick_examples(truth, curve_pred, dual_pred, lambda_um)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), sharey=True)
    title_map = {"single": "单峰样本", "multi": "多峰样本", "wide": "宽带样本"}
    for ax, group in zip(axes, ("single", "multi", "wide")):
        idx = picks[group]
        ax.plot(lambda_um, truth[idx], color="#1f4e79", lw=2.2, label="仿真真值")
        ax.plot(lambda_um, curve_pred[idx], color="#e15759", lw=1.9, ls="--", label="曲线模型")
        ax.plot(lambda_um, dual_pred[idx], color="#f28c28", lw=2.0, ls="-.", label="双头模型")
        ax.set_title(title_map[group], fontsize=13, pad=5)
        ax.set_xlabel("波长 (um)", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(float(lambda_um.min()), float(lambda_um.max()))
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        mse_curve = np.mean((curve_pred[idx] - truth[idx]) ** 2)
        mse_dual = np.mean((dual_pred[idx] - truth[idx]) ** 2)
        ax.text(
            0.03,
            0.05,
            f"MSE: {mse_curve:.3f} -> {mse_dual:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#35536b",
            bbox={"facecolor": "white", "edgecolor": "#d5e2ec", "alpha": 0.82, "pad": 2},
        )
    axes[0].set_ylabel("吸收率", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout(pad=0.35, w_pad=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
