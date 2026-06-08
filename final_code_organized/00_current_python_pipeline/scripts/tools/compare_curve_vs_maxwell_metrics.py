from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fullfield_dual_surrogate import FullFieldDualSurrogatePredictor
from src.project_paths import FIELD_DATA_DIR, SAMPLING_META_PATH

with (Path(__file__).resolve().parents[2] / "outputs" / "_curve_only_import.log").open("w", encoding="utf-8") as log:
    with redirect_stdout(log):
        from scripts.train.train_fno_curve_only_pycharm import (
            CurveOnlySpectrumModel,
            normalize_interval,
            project_s11_to_passive,
            s11_to_absorption_torch,
        )


CURVE_CACHE_PATH = PROJECT_ROOT / "data" / "curve_cache" / "curve_dataset_11x11_s11_a.npz"
OUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"


def build_val_sample_ids(limit: int = 256) -> list[int]:
    sample_files = sorted(FIELD_DATA_DIR.glob("sample_*.mat"))
    perm = np.random.default_rng(42).permutation(len(sample_files))
    n_train = int(0.85 * len(sample_files))
    val_files = sorted([sample_files[i] for i in perm[n_train:]], key=lambda p: p.name)[:limit]
    return [int(p.stem.split("_")[-1]) for p in val_files]


def load_val_cache(limit: int = 256) -> dict[str, np.ndarray]:
    cache = dict(np.load(CURVE_CACHE_PATH, allow_pickle=False))
    sample_ids = cache["sample_id"].astype(int).reshape(-1)
    val_ids = set(build_val_sample_ids(limit))
    mask = np.array([int(sid) in val_ids for sid in sample_ids], dtype=bool)
    return {
        "lambda_vec": cache["lambda_vec"].astype(np.float32).reshape(-1),
        "sample_id": sample_ids[mask],
        "pattern_11": cache["pattern_11"][mask].astype(np.float32),
        "truth": cache["absorption"][mask].astype(np.float32),
    }


def predict_curve_only(checkpoint_path: Path, patterns: np.ndarray, lambda_vec: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model = CurveOnlySpectrumModel(
        modes_x=int(cfg.get("MODES_X", 8)),
        modes_y=int(cfg.get("MODES_Y", 8)),
        width=int(cfg.get("WIDTH", 48)),
        depth=int(cfg.get("DEPTH", 4)),
        lam_ff=int(cfg.get("LAM_FF", 6)),
        head_hidden=int(cfg.get("HEAD_HIDDEN", 192)),
        curve_blocks=int(cfg.get("CURVE_BLOCKS", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    lam_norm = torch.from_numpy(normalize_interval(lambda_vec).astype(np.float32)).view(1, -1, 1).to(device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(patterns), batch_size):
            batch = torch.from_numpy(patterns[start : start + batch_size, None]).to(device)
            lam = lam_norm.expand(batch.shape[0], -1, -1)
            pred_s11_raw, _ = model(batch, lam)
            pred_s11 = project_s11_to_passive(pred_s11_raw)
            pred_a = s11_to_absorption_torch(pred_s11)
            preds.append(pred_a.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def predict_maxwell(checkpoint_path: Path, patterns: np.ndarray, device_name: str, batch_size: int) -> np.ndarray:
    predictor = FullFieldDualSurrogatePredictor(
        checkpoint_path=checkpoint_path,
        meta_path=SAMPLING_META_PATH,
        device=device_name,
        forward_batch_size=batch_size,
        lambda_chunk_size=24,
    )
    pred, _ = predictor.predict_spectrum(patterns)
    return pred.astype(np.float32)


def classify_curve(curve: np.ndarray, lambda_um: np.ndarray) -> str:
    from scipy.signal import find_peaks, peak_widths

    peaks, _ = find_peaks(curve, height=0.35, distance=4, prominence=0.03)
    main_idx = int(np.argmax(curve))
    if len(peaks) > 0:
        amps = np.sort(curve[peaks])[::-1]
    else:
        amps = np.array([float(curve[main_idx])], dtype=np.float32)
    main_amp = float(amps[0])
    second_amp = float(amps[1]) if len(amps) > 1 else 0.0
    dominance = main_amp / (second_amp + 1e-6)
    width_um = float(peak_widths(curve, [main_idx], rel_height=0.5)[0][0]) * float(lambda_um[1] - lambda_um[0])
    if width_um > 1.6 or float(np.mean(curve)) > 0.48:
        return "wide"
    if dominance > 1.35 and width_um < 0.9:
        return "single"
    return "multi"


def summarize_prediction(pred: np.ndarray, truth: np.ndarray, lambda_vec: np.ndarray) -> dict:
    lambda_um = lambda_vec.astype(np.float32) * 1e6
    true_main_idx = np.argmax(truth, axis=1)
    pred_main_idx = np.argmax(pred, axis=1)
    mse = np.mean((pred - truth) ** 2, axis=1)
    mae = np.mean(np.abs(pred - truth), axis=1)
    pos_err = np.abs(lambda_um[pred_main_idx] - lambda_um[true_main_idx])
    height_err = np.abs(pred[np.arange(len(pred)), pred_main_idx] - truth[np.arange(len(truth)), true_main_idx])
    ss_res = np.sum((pred - truth) ** 2, axis=1)
    ss_tot = np.sum((truth - truth.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    labels = np.array([classify_curve(c, lambda_um) for c in truth])

    def pack(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "n": int(mask.sum()),
            "mse": float(np.mean(mse[mask])),
            "mae": float(np.mean(mae[mask])),
            "r2": float(np.mean(r2[mask])),
            "main_pos_err_um": float(np.mean(pos_err[mask])),
            "main_height_err": float(np.mean(height_err[mask])),
        }

    out = {"overall": pack(np.ones(len(truth), dtype=bool)), "groups": {}}
    for name in ("single", "multi", "wide"):
        mask = labels == name
        if np.any(mask):
            out["groups"][name] = pack(mask)
    return out


def add_improvement(result: dict) -> None:
    curve = result["curve_only"]["overall"]
    maxwell = result["maxwell"]["overall"]
    result["relative_improvement"] = {}
    for key in ("mse", "mae", "main_pos_err_um", "main_height_err"):
        base = float(curve[key])
        new = float(maxwell[key])
        result["relative_improvement"][key] = float((base - new) / base) if abs(base) > 1e-12 else None


def save_metric_figure(result: dict) -> None:
    labels = ["Curve MSE", "Peak position error (um)", "Peak height error"]
    keys = ["mse", "main_pos_err_um", "main_height_err"]
    curve = [result["curve_only"]["overall"][k] for k in keys]
    maxwell = [result["maxwell"]["overall"][k] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    colors = ["#4e79a7", "#e15759"]
    for ax, label, cval, mval in zip(axes, labels, curve, maxwell):
        ax.bar(["Curve only", "Field+Maxwell"], [cval, mval], color=colors, width=0.62)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymax = max(cval, mval) * 1.22
        ax.set_ylim(0.0, ymax)
        ax.text(0, cval + ymax * 0.025, f"{cval:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(1, mval + ymax * 0.025, f"{mval:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Unified validation metrics on the same 256-sample split", y=1.05)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "ch5_training_ablation.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-ckpt", type=Path, default=PROJECT_ROOT / "models" / "current" / "fno_curve_only_best.pt")
    parser.add_argument("--maxwell-ckpt", type=Path, default=PROJECT_ROOT / "models" / "history" / "20260419-205756" / "run_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data = load_val_cache(args.limit)
    device = torch.device(args.device)

    curve_pred = predict_curve_only(args.curve_ckpt, data["pattern_11"], data["lambda_vec"], device, args.batch_size)
    maxwell_pred = predict_maxwell(args.maxwell_ckpt, data["pattern_11"], args.device, args.batch_size)

    result = {
        "val_samples": int(len(data["truth"])),
        "lambda_points": int(len(data["lambda_vec"])),
        "curve_checkpoint": str(args.curve_ckpt),
        "maxwell_checkpoint": str(args.maxwell_ckpt),
        "curve_only": summarize_prediction(curve_pred, data["truth"], data["lambda_vec"]),
        "maxwell": summarize_prediction(maxwell_pred, data["truth"], data["lambda_vec"]),
    }
    add_improvement(result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "curve_vs_maxwell_metrics.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_metric_figure(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
