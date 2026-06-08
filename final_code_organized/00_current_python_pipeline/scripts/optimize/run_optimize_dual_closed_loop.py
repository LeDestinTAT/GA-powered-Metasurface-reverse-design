import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from run_optimize_dual import build_optimizer_config, plot_best, run_nsga2
from src.fullfield_dual_surrogate import FullFieldDualSurrogatePredictor, field_to_view


def choose_lambda_index(lambda_um, spectrum, cfg):
    post_cfg = cfg.get("postprocess", {})
    mode = str(post_cfg.get("lambda_mode", "max_absorption")).lower()
    lambda_um = np.asarray(lambda_um, dtype=np.float32).reshape(-1)
    spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)

    if mode == "target_peak":
        target_um = float(cfg["target_peak"]["pos"])
        return int(np.argmin(np.abs(lambda_um - target_um)))
    if mode == "custom":
        target_um = float(post_cfg["lambda_um"])
        return int(np.argmin(np.abs(lambda_um - target_um)))
    return int(np.argmax(spectrum))


def plot_closed_loop_field(run_dir: Path, predictor, fields, field_component: str, field_view: str, lambda_um: float):
    xv = predictor.xv
    yv = predictor.yv
    zv = predictor.zv
    z_idx = len(zv) // 2
    y_idx = len(yv) // 2

    pred_xy = field_to_view(fields[field_component][:, :, z_idx], field_view)
    pred_xz = field_to_view(fields[field_component][:, y_idx, :], field_view)

    vmax_xy = max(np.max(np.abs(pred_xy)), 1e-12)
    vmax_xz = max(np.max(np.abs(pred_xz)), 1e-12)
    vmin_xy = -vmax_xy if field_view != "magnitude" else 0.0
    vmin_xz = -vmax_xz if field_view != "magnitude" else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent_xy = [xv[0] * 1e6, xv[-1] * 1e6, yv[0] * 1e6, yv[-1] * 1e6]
    extent_xz = [xv[0] * 1e6, xv[-1] * 1e6, zv[0] * 1e9, zv[-1] * 1e9]

    im = axes[0].imshow(pred_xy.T, origin="lower", extent=extent_xy, aspect="auto", cmap="jet", vmin=vmin_xy, vmax=vmax_xy)
    axes[0].set_title(f"Pred {field_component} XY @ z={zv[z_idx] * 1e9:.1f} nm")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(pred_xz.T, origin="lower", extent=extent_xz, aspect="auto", cmap="jet", vmin=vmin_xz, vmax=vmax_xz)
    axes[1].set_title(f"Pred {field_component} XZ @ y={yv[y_idx] * 1e6:.2f} um")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (nm)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle(f"Closed-loop {field_component} | view={field_view} | lambda={lambda_um:.3f} um", fontsize=13)
    plt.tight_layout()
    fig.savefig(run_dir / f"closed_loop_{field_component}_lambda_{lambda_um:.3f}um.png", dpi=200)
    plt.close(fig)


def plot_closed_loop_absorption(run_dir: Path, lambda_um, absorption, selected_idx: int):
    fig = plt.figure(figsize=(6.8, 4.4))
    plt.plot(lambda_um, absorption, label="Pred A")
    plt.scatter([lambda_um[selected_idx]], [absorption[selected_idx]], color="k", zorder=3)
    plt.xlabel("lambda (um)")
    plt.ylabel("A")
    plt.title("Closed-loop Predicted Absorption")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "closed_loop_absorption.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg = build_optimizer_config(cfg)
    print("[Config]")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    run_dir_str, lambda_um = run_nsga2(cfg)
    run_dir = Path(run_dir_str)
    plot_best(run_dir, cfg, lambda_um)

    best_pattern = np.load(run_dir / "best_pattern.npy").astype(np.float32)
    best_spectrum = np.load(run_dir / "best_spectrum.npy").astype(np.float32)

    predictor_cfg = dict(cfg.get("predictor", {}))
    geometry_cfg = dict(cfg.get("geometry", {}))
    predictor = FullFieldDualSurrogatePredictor(
        checkpoint_path=Path(cfg["resolved_ckpt_path"]),
        meta_path=Path(cfg["meta_path"]),
        device=cfg["device"],
        bottom_metal_zmax=float(geometry_cfg.get("bottom_metal_zmax", 100e-9)),
        dielectric_zmax=float(geometry_cfg.get("dielectric_zmax", 400e-9)),
        top_pattern_zmax=float(geometry_cfg.get("top_pattern_zmax", 430e-9)),
        forward_batch_size=int(predictor_cfg.get("forward_batch_size", 64)),
        lambda_chunk_size=int(predictor_cfg.get("lambda_chunk_size", 16)),
    )

    selected_idx = choose_lambda_index(lambda_um, best_spectrum, cfg)
    post_cfg = cfg.get("postprocess", {})
    field_component = str(post_cfg.get("field_component", "Ez"))
    field_view = str(post_cfg.get("field_view", "magnitude"))

    field_result = predictor.predict_field_at_lambda(best_pattern[None, ...], lambda_index=selected_idx)
    selected_lambda_um = field_result["lambda_m"] * 1e6
    fields = field_result["fields"][0]

    plot_closed_loop_absorption(run_dir, lambda_um, best_spectrum, selected_idx)
    plot_closed_loop_field(run_dir, predictor, fields, field_component, field_view, selected_lambda_um)

    np.savez_compressed(
        run_dir / "closed_loop_prediction.npz",
        best_pattern=best_pattern.astype(np.float32),
        lambda_um=np.asarray(lambda_um, dtype=np.float32),
        best_spectrum=best_spectrum.astype(np.float32),
        selected_lambda_um=np.float32(selected_lambda_um),
        selected_lambda_index=np.int32(selected_idx),
        ex_norm=fields["Ex"].astype(np.complex64),
        ey_norm=fields["Ey"].astype(np.complex64),
        ez_norm=fields["Ez"].astype(np.complex64),
        hx_norm=fields["Hx"].astype(np.complex64),
        hy_norm=fields["Hy"].astype(np.complex64),
        hz_norm=fields["Hz"].astype(np.complex64),
    )

    print(f"\nClosed-loop finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
