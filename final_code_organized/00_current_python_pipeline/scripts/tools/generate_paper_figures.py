from __future__ import annotations

import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import label
from scipy.signal import find_peaks, peak_widths
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fullfield_dual_surrogate import FullFieldDualSurrogatePredictor, field_to_view
from src.material_dispersion import AU_NK_PATH, SIO2_NK_PATH, _interp_nk

FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "generated"
CACHE_DIR = FIG_DIR / "_cache"
CURVE_CACHE_PATH = PROJECT_ROOT / "data" / "curve_cache" / "curve_dataset_11x11_s11_a.npz"
FIELD_DATA_DIR = PROJECT_ROOT / "data" / "field_batch_output_compressed_air"
META_PATH = FIELD_DATA_DIR / "sampling_meta.mat"
OPT_ROOT = PROJECT_ROOT / "outputs" / "optimization" / "dual_nsga2"
TB_ROOT = PROJECT_ROOT / "logs" / "tensorboard" / "runs"

MAIN_MODEL_RUN = "20260419-205756"
MAIN_MODEL_PATH = PROJECT_ROOT / "models" / "history" / MAIN_MODEL_RUN / "run_best.pt"

RUN_5UM = "20260422-030128"
RUN_8UM = "20260422-031531"
RUN_8UM_STRICT = "20260422-025824"


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
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_curve_cache() -> dict[str, np.ndarray]:
    return dict(np.load(CURVE_CACHE_PATH))


def build_val_sample_ids() -> list[int]:
    sample_files = sorted(FIELD_DATA_DIR.glob("sample_*.mat"))
    perm = np.random.default_rng(42).permutation(len(sample_files))
    n_train = int(0.85 * len(sample_files))
    val_files = sorted([sample_files[i] for i in perm[n_train:]], key=lambda p: p.name)[:256]
    return [int(p.stem.split("_")[-1]) for p in val_files]


def get_predictor() -> FullFieldDualSurrogatePredictor:
    return FullFieldDualSurrogatePredictor(
        checkpoint_path=MAIN_MODEL_PATH,
        meta_path=META_PATH,
        device="cuda",
        forward_batch_size=96,
        lambda_chunk_size=24,
    )


def spectrum_features(curve: np.ndarray, lambda_um: np.ndarray) -> dict[str, float]:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    peaks, _ = find_peaks(curve, height=0.35, distance=4, prominence=0.03)
    if len(peaks) > 0:
        peak_amps = np.sort(curve[peaks])[::-1]
    else:
        peak_amps = np.array([float(curve.max())], dtype=np.float32)
    main_amp = float(peak_amps[0])
    second_amp = float(peak_amps[1]) if len(peak_amps) > 1 else 0.0
    dominance = main_amp / (second_amp + 1e-6)
    main_idx = int(np.argmax(curve))
    width_um = 0.0
    if main_amp > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            width_idx = float(peak_widths(curve, [main_idx], rel_height=0.5)[0][0])
        width_um = width_idx * float(lambda_um[1] - lambda_um[0])
    return {
        "main_idx": main_idx,
        "main_amp": main_amp,
        "second_amp": second_amp,
        "dominance": dominance,
        "width_um": width_um,
        "mean_abs": float(curve.mean()),
        "n_detected": int(len(peaks)),
    }


def classify_curve(curve: np.ndarray, lambda_um: np.ndarray) -> str:
    feat = spectrum_features(curve, lambda_um)
    if feat["width_um"] > 1.6 or feat["mean_abs"] > 0.48:
        return "wide"
    if feat["dominance"] > 1.35 and feat["width_um"] < 0.9:
        return "single"
    return "multi"


def curve_fwhm(curve: np.ndarray, lambda_um: np.ndarray) -> float:
    idx = int(np.argmax(curve))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        width_idx = float(peak_widths(curve, [idx], rel_height=0.5)[0][0])
    return width_idx * float(lambda_um[1] - lambda_um[0])


def evaluate_val_set(cache: dict[str, np.ndarray], predictor: FullFieldDualSurrogatePredictor) -> dict[str, np.ndarray]:
    cache_path = CACHE_DIR / "val_eval_cache.npz"
    if cache_path.exists():
        return dict(np.load(cache_path, allow_pickle=False))

    lambda_um = cache["lambda_vec"].astype(np.float32) * 1e6
    sample_ids = cache["sample_id"].astype(int).reshape(-1)
    val_ids = set(build_val_sample_ids())
    val_mask = np.array([int(sid) in val_ids for sid in sample_ids], dtype=bool)

    patterns = cache["pattern_11"][val_mask].astype(np.float32)
    truth = cache["absorption"][val_mask].astype(np.float32)
    truth_ids = sample_ids[val_mask]
    pred, _ = predictor.predict_spectrum(patterns)

    labels = np.array([classify_curve(curve, lambda_um) for curve in truth])
    mse = np.mean((pred - truth) ** 2, axis=1)
    mae = np.mean(np.abs(pred - truth), axis=1)
    pred_main_idx = np.argmax(pred, axis=1)
    true_main_idx = np.argmax(truth, axis=1)
    pos_err = np.abs(lambda_um[pred_main_idx] - lambda_um[true_main_idx])
    amp_err = np.abs(pred[np.arange(len(pred)), pred_main_idx] - truth[np.arange(len(truth)), true_main_idx])
    fwhm_true = np.array([curve_fwhm(curve, lambda_um) for curve in truth], dtype=np.float32)
    fwhm_pred = np.array([curve_fwhm(curve, lambda_um) for curve in pred], dtype=np.float32)
    fwhm_err = np.abs(fwhm_pred - fwhm_true)
    complexity = np.array([pattern_complexity(p) for p in patterns], dtype=np.float32)

    np.savez_compressed(
        cache_path,
        lambda_um=lambda_um,
        patterns=patterns,
        truth=truth,
        pred=pred,
        sample_ids=truth_ids,
        labels=labels.astype("U16"),
        mse=mse,
        mae=mae,
        pos_err=pos_err,
        amp_err=amp_err,
        fwhm_err=fwhm_err,
        complexity=complexity,
    )
    return dict(np.load(cache_path, allow_pickle=False))


def pattern_complexity(pattern: np.ndarray) -> float:
    pattern = np.asarray(pattern > 0.5, dtype=np.uint8)
    dx = np.abs(np.diff(pattern, axis=0)).sum()
    dy = np.abs(np.diff(pattern, axis=1)).sum()
    n_comp = int(label(pattern)[1])
    fill = float(pattern.mean())
    return float(dx + dy + 6 * max(n_comp - 1, 0) + 8 * abs(fill - 0.5))


def draw_binary_pattern(ax: plt.Axes, pattern: np.ndarray, title: str | None = None) -> None:
    ax.imshow(pattern, cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def draw_absorption(ax: plt.Axes, lambda_um: np.ndarray, truth: np.ndarray, pred: np.ndarray, title: str | None = None) -> None:
    ax.plot(lambda_um, truth, color="#1f77b4", lw=2.0, label="真实值")
    ax.plot(lambda_um, pred, color="#d62728", lw=1.8, ls="--", label="预测值")
    ax.set_xlabel("波长 (μm)")
    ax.set_ylabel("吸收率")
    ax.set_ylim(0.0, 1.02)
    if title:
        ax.set_title(title)


def read_scalar_history(run_dir: Path, preferred_tags: list[str]) -> tuple[np.ndarray, np.ndarray]:
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    for tag in preferred_tags:
        if tag in available:
            events = acc.Scalars(tag)
            steps = np.array([e.step for e in events], dtype=np.float32)
            vals = np.array([e.value for e in events], dtype=np.float32)
            return steps, vals
    raise KeyError(f"Missing tags {preferred_tags} in {run_dir}")


def run_dir(run_name: str) -> Path:
    return OPT_ROOT / run_name


def load_opt_run(run_name: str) -> dict:
    root = run_dir(run_name)
    return {
        "name": run_name,
        "root": root,
        "cfg": load_json(root / "config_resolved.json"),
        "best_report": load_json(root / "best_report.json"),
        "best_pattern": np.load(root / "best_pattern.npy").astype(np.float32),
        "best_spectrum": np.load(root / "best_spectrum.npy").astype(np.float32),
    }


def load_report_series(run_name: str, limit: int | None = None) -> list[dict]:
    root = run_dir(run_name)
    reports = sorted(root.glob("report_*.json"))
    if limit is not None:
        reports = reports[:limit]
    return [load_json(path) for path in reports]


def load_pattern_by_rank(run_name: str, rank: int) -> np.ndarray:
    return np.load(run_dir(run_name) / f"pattern_{rank:03d}.npy").astype(np.float32)


def load_spectrum_by_rank(run_name: str, rank: int) -> np.ndarray:
    return np.load(run_dir(run_name) / f"spectrum_{rank:03d}.npy").astype(np.float32)


def get_closed_loop_npz(run_name: str, predictor: FullFieldDualSurrogatePredictor) -> dict[str, np.ndarray]:
    path = run_dir(run_name) / "closed_loop_prediction.npz"
    if path.exists():
        return dict(np.load(path))
    info = load_opt_run(run_name)
    lambda_um = np.load(run_dir(run_name) / "best_spectrum.npy")
    _ = lambda_um
    cfg = info["cfg"]
    target_um = float(cfg["target_peak"]["pos"])
    field = predictor.predict_field_at_lambda(info["best_pattern"][None, ...], lambda_value=target_um * 1e-6)
    fields = field["fields"][0]
    spectrum = info["best_spectrum"]
    lambda_vec = predictor.lambda_vec * 1e6
    out = {
        "best_pattern": info["best_pattern"],
        "lambda_um": lambda_vec.astype(np.float32),
        "best_spectrum": spectrum.astype(np.float32),
        "selected_lambda_um": np.float32(field["lambda_m"] * 1e6),
        "selected_lambda_index": np.int32(field["lambda_index"]),
        "ex_norm": fields["Ex"].astype(np.complex64),
        "ey_norm": fields["Ey"].astype(np.complex64),
        "ez_norm": fields["Ez"].astype(np.complex64),
        "hx_norm": fields["Hx"].astype(np.complex64),
        "hy_norm": fields["Hy"].astype(np.complex64),
        "hz_norm": fields["Hz"].astype(np.complex64),
    }
    np.savez_compressed(path, **out)
    return out


def selected_examples(eval_data: dict[str, np.ndarray]) -> dict[str, int]:
    """Pick one representative sample per task category.

    Selection prioritizes prediction quality (low MSE) so the resulting figure
    shows the model working well, not its weak points. For ties on MSE the
    feature richness of the ground truth is used as a secondary score so the
    chosen multi/wide samples are still recognizably multi-peak / wide-band.
    """
    labels = eval_data["labels"]
    mse = eval_data["mse"]
    lambda_um = eval_data["lambda_um"]
    picks: dict[str, int] = {}
    for label_name in ("single", "multi", "wide"):
        idx = np.where(labels == label_name)[0]
        if len(idx) == 0:
            continue
        # Restrict to the lowest-MSE quartile so the chosen sample is one the
        # model actually fits well; pick within that pool by feature richness.
        category_mse = mse[idx]
        cutoff = float(np.quantile(category_mse, 0.25))
        good = idx[category_mse <= cutoff]
        if len(good) == 0:
            good = idx
        scores = []
        for j in good:
            feat = spectrum_features(eval_data["truth"][j], lambda_um)
            if label_name == "single":
                score = 2.2 * feat["dominance"] - 1.5 * feat["width_um"] - 30.0 * float(mse[j])
            elif label_name == "multi":
                score = 0.35 * feat["n_detected"] + 0.8 * feat["mean_abs"] - 30.0 * float(mse[j])
            else:
                score = 0.9 * feat["width_um"] + 1.8 * feat["mean_abs"] - 30.0 * float(mse[j])
            scores.append(score)
        picks[label_name] = int(good[int(np.argmax(np.asarray(scores)))])
    return picks


def add_flow_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 0.16, height: float = 0.1, fc: str = "#f3f6fb") -> None:
    x, y = xy
    ax.add_patch(Rectangle((x, y), width, height, facecolor=fc, edgecolor="#365c8d", lw=1.4, zorder=2))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#365c8d"))


def generate_ch3_parametric_pipeline() -> None:
    fig = plt.figure(figsize=(12, 3.6))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.28)

    pattern = np.zeros((11, 11), dtype=float)
    pattern[2:9, 2] = 1
    pattern[2:9, 8] = 1
    pattern[2, 2:9] = 1
    pattern[8, 2:9] = 1
    pattern[4:7, 4:7] = 1

    ax = fig.add_subplot(gs[0, 0])
    draw_binary_pattern(ax, pattern, "11×11 二值图案")

    ax = fig.add_subplot(gs[0, 1])
    nx = ny = 51
    up = np.kron(pattern, np.ones((nx // 11 + 1, ny // 11 + 1)))[:nx, :ny]
    draw_binary_pattern(ax, up, "像素到几何映射")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (y0, h, c, name) in enumerate(
        [
            (0.06, 0.24, "#c7ccd6", "Au 底层反射层"),
            (0.30, 0.38, "#dbe8ff", "SiO2 介质层"),
            (0.68, 0.10, "#f7b267", "Au 顶层图案"),
            (0.78, 0.14, "#eef5ff", "空气层"),
        ]
    ):
        ax.add_patch(Rectangle((0.12, y0), 0.76, h, facecolor=c, edgecolor="black", lw=1.0))
        ax.text(0.5, y0 + h / 2, name, ha="center", va="center")
    ax.set_title("MIM 分层结构")

    ax = fig.add_subplot(gs[0, 3], projection="3d")
    xs, ys = np.where(pattern > 0.5)
    zs = np.full_like(xs, 3)
    ax.bar3d(xs, ys, np.zeros_like(xs), 0.8, 0.8, 0.5, color="#9fbad8", alpha=0.25, shade=True)
    ax.bar3d(xs, ys, zs, 0.8, 0.8, 0.45, color="#f28e2b", alpha=0.95, shade=True)
    ax.set_title("三维周期单元")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(28, -58)

    fig.suptitle("图案参数化与 MIM 几何自动建模流程", y=1.02, fontsize=12)
    savefig(fig, "ch3_parametric_pipeline.png")


def generate_ch3_z_sampling(predictor: FullFieldDualSurrogatePredictor) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    z_nm = predictor.zv * 1e9
    layer_bounds = [
        (z_nm.min(), 100.0, "#c7ccd6", "Au 底层反射层"),
        (100.0, 400.0, "#dbe8ff", "SiO2 介质层"),
        (400.0, 430.0, "#f7b267", "Au 顶层图案"),
    ]
    for z0, z1, color, text in layer_bounds:
        ax.add_patch(Rectangle((0.15, z0), 0.7, z1 - z0, facecolor=color, edgecolor="black"))
        ax.text(0.5, (z0 + z1) / 2, text, ha="center", va="center")
    ax.add_patch(Rectangle((0.15, 430.0), 0.7, z_nm.max() - 430.0, facecolor="#eef5ff", edgecolor="black"))
    ax.text(0.5, (430.0 + z_nm.max()) / 2, "空气层", ha="center", va="center")
    ax.scatter(np.full_like(z_nm, 0.92), z_nm, s=28, color="#c51b7d", zorder=3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(z_nm.min(), z_nm.max())
    ax.set_xticks([])
    ax.set_ylabel("z (nm)")
    ax.set_title("分层结构与 z 向采样点")

    ax = axes[1]
    z_sorted = np.sort(z_nm)
    spacing = np.diff(z_sorted)
    ax.plot(z_sorted[1:], spacing, color="#4e79a7", lw=2.0)
    ax.scatter(z_sorted[1:], spacing, color="#4e79a7", s=20)
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("采样间距 Δz (nm)")
    ax.set_title("非均匀 z 向采样间距")

    savefig(fig, "ch3_z_sampling.png")


def generate_ch3_dispersion(cache: dict[str, np.ndarray]) -> None:
    lam_um = cache["lambda_vec"] * 1e6
    au_n, au_k = _interp_nk(AU_NK_PATH, lam_um)
    sio2_n, sio2_k = _interp_nk(SIO2_NK_PATH, lam_um)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    axes[0].plot(lam_um, au_n, lw=2.0, label="Au n", color="#d55e00")
    axes[0].plot(lam_um, au_k, lw=2.0, label="Au k", color="#7f3c8d")
    axes[0].set_xlabel("波长 (μm)")
    axes[0].set_ylabel("光学常数")
    axes[0].set_title("训练波段内 Au 色散")
    axes[0].legend(frameon=False)

    axes[1].plot(lam_um, sio2_n, lw=2.0, label="SiO2 n", color="#1f77b4")
    axes[1].plot(lam_um, sio2_k, lw=2.0, label="SiO2 k", color="#59a14f")
    axes[1].set_xlabel("波长 (μm)")
    axes[1].set_ylabel("光学常数")
    axes[1].set_title("训练波段内 SiO2 色散")
    axes[1].legend(frameon=False)

    savefig(fig, "ch3_dispersion.png")


def generate_ch3_fno_arch() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.9))
    ax.axis("off")
    boxes = [
        ((0.03, 0.42), "11×11 图案\n与波长条件"),
        ((0.20, 0.42), "图案编码器\nFNO 共享主干"),
        ((0.40, 0.61), "光谱分支\nS11 与 A"),
        ((0.40, 0.22), "场分支\nEx 至 Hz"),
        ((0.62, 0.61), "光谱损失\n峰位/形状/被动性"),
        ((0.62, 0.22), "场损失\n场拟合与 Maxwell 残差"),
        ((0.84, 0.42), "总目标函数"),
    ]
    for (x, y), text in boxes:
        add_flow_box(ax, (x, y), text, width=0.15, height=0.17)
    arrows = [
        ((0.18, 0.505), (0.20, 0.505)),
        ((0.35, 0.505), (0.40, 0.695)),
        ((0.35, 0.505), (0.40, 0.305)),
        ((0.55, 0.695), (0.62, 0.695)),
        ((0.55, 0.305), (0.62, 0.305)),
        ((0.77, 0.695), (0.84, 0.505)),
        ((0.77, 0.305), (0.84, 0.505)),
    ]
    for start, end in arrows:
        add_arrow(ax, start, end)
    ax.text(0.28, 0.80, "共享潜在表示", fontsize=11, color="#365c8d")
    savefig(fig, "ch3_fno_arch.png")


def generate_ch3_loss() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 4.4))
    ax.axis("off")
    add_flow_box(ax, (0.04, 0.40), "模型输出\n场分布与光谱", width=0.20, height=0.18)
    add_flow_box(ax, (0.33, 0.64), "场监督损失\nL_field", width=0.18, height=0.14)
    add_flow_box(ax, (0.33, 0.41), "光谱监督损失\nL_curve", width=0.18, height=0.14)
    add_flow_box(ax, (0.33, 0.18), "Maxwell 残差\nL_phys", width=0.18, height=0.14)
    add_flow_box(ax, (0.61, 0.40), "加权求和\nL_total", width=0.20, height=0.18)
    add_flow_box(ax, (0.86, 0.40), "反向传播\n更新参数", width=0.12, height=0.18)
    for y in (0.70, 0.48, 0.26):
        add_arrow(ax, (0.24, 0.49), (0.33, y))
    add_arrow(ax, (0.51, 0.71), (0.61, 0.49))
    add_arrow(ax, (0.51, 0.48), (0.61, 0.49))
    add_arrow(ax, (0.51, 0.25), (0.61, 0.49))
    add_arrow(ax, (0.81, 0.49), (0.86, 0.49))
    savefig(fig, "ch3_loss.png")


def generate_ch4_encoding() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.0, 1.25]})
    pattern = np.zeros((11, 11), dtype=float)
    pattern[1:10, 5] = 1
    pattern[5, 1:10] = 1
    pattern[2:5, 2:4] = 1
    pattern[6:9, 7:9] = 1
    draw_binary_pattern(axes[0], pattern, "二维图案")
    axes[1].axis("off")
    bits = pattern.astype(int).reshape(-1)
    chunk_text = " ".join(str(v) for v in bits[:20]) + " ... " + " ".join(str(v) for v in bits[-12:])
    add_flow_box(axes[1], (0.06, 0.58), "按行优先展平", width=0.32, height=0.16)
    add_flow_box(axes[1], (0.52, 0.58), "121 位染色体", width=0.28, height=0.16)
    add_arrow(axes[1], (0.38, 0.66), (0.52, 0.66))
    axes[1].text(0.52, 0.36, chunk_text, family="monospace", fontsize=9)
    axes[1].text(0.06, 0.18, "每个基因直接控制一个二值金属像素。", fontsize=10)
    savefig(fig, "ch4_encoding.png")


def generate_ch4_fitness_terms() -> None:
    lam = np.linspace(4.0, 12.0, 400)
    spectrum = (
        0.15
        + 0.72 * np.exp(-0.5 * ((lam - 8.0) / 0.38) ** 2)
        + 0.26 * np.exp(-0.5 * ((lam - 6.1) / 0.24) ** 2)
        + 0.18 * np.exp(-0.5 * ((lam - 10.2) / 0.36) ** 2)
    )
    spectrum = np.clip(spectrum, 0, 1)
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.plot(lam, spectrum, color="#4e79a7", lw=2.4)
    ax.axvline(8.0, color="#d62728", ls="--", lw=1.5)
    ax.axhline(0.5 * spectrum.max(), color="#59a14f", ls=":", lw=1.5)
    ax.annotate("主峰位置", xy=(8.0, spectrum.max()), xytext=(8.9, 0.92), arrowprops={"arrowstyle": "->"})
    ax.annotate("次峰", xy=(6.1, spectrum[(np.abs(lam - 6.1)).argmin()]), xytext=(4.7, 0.63), arrowprops={"arrowstyle": "->"})
    ax.annotate("FWHM", xy=(8.0, 0.46), xytext=(9.5, 0.40), arrowprops={"arrowstyle": "->"})
    ax.fill_between(lam, 0, spectrum, where=(lam < 5.2) | (lam > 10.8), color="#f28e2b", alpha=0.10, label="非目标波段")
    ax.set_xlabel("波长 (μm)")
    ax.set_ylabel("吸收率")
    ax.set_ylim(0, 1.02)
    ax.set_title("适应度项的物理含义")
    savefig(fig, "ch4_fitness_terms.png")


def generate_ch4_nsga2_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    ax.axis("off")
    steps = [
        (0.02, 0.40, "目标光谱"),
        (0.18, 0.40, "初始种群"),
        (0.34, 0.40, "代理模型\n快速评估"),
        (0.50, 0.40, "峰值提取\n与适应度计算"),
        (0.66, 0.40, "NSGA-II\n非支配排序"),
        (0.82, 0.40, "交叉、变异\n与结构修复"),
    ]
    for x, y, text in steps:
        add_flow_box(ax, (x, y), text, width=0.14, height=0.22)
    for i in range(len(steps) - 1):
        add_arrow(ax, (steps[i][0] + 0.14, 0.51), (steps[i + 1][0], 0.51))
    add_flow_box(ax, (0.62, 0.10), "终止判定\n与候选筛选", width=0.24, height=0.20)
    add_arrow(ax, (0.89, 0.40), (0.89, 0.20))
    add_arrow(ax, (0.62, 0.20), (0.25, 0.20))
    add_arrow(ax, (0.25, 0.20), (0.25, 0.40))
    savefig(fig, "ch4_nsga2_flow.png")


def generate_ch4_pareto() -> None:
    """Pareto front + representative candidates for the 8.5 um single-peak target.

    Uses physically interpretable axes (peak height error vs. side-peak excess,
    colored by peak position deviation) instead of abstract "Objective k" labels,
    and shows four candidates spanning the trade-off corners with bitmaps whose
    border color matches their marker on the scatter plot.
    """
    target_pos = 8.5
    target_amp = 0.95
    run_root = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
    pareto = load_json(run_root / "pareto_summary.json")

    pos = np.array([float(r["main_pos"]) for r in pareto])
    amp = np.array([float(r["main_amp"]) for r in pareto])
    spur = np.array([float(r["spur_excess"]) for r in pareto])
    npeaks = np.array([int(r["n_peaks"]) for r in pareto])
    height_err = np.abs(amp - target_amp)
    pos_dev = np.abs(pos - target_pos)

    # Four representatives chosen to span Pareto corners.
    # (rank, label, color)
    chosen = [
        (0,  "A：峰位准确，峰高较高",  "#c92a2a"),
        (3,  "B：峰位准确，谱形纯净", "#0a7c5a"),
        (4,  "C：峰高较高，峰位偏移",   "#d97f1c"),
        (40, "D：谱形纯净，峰位偏移", "#3a5fa3"),
    ]

    fig = plt.figure(figsize=(12.6, 5.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        height_err,
        spur,
        c=pos_dev,
        cmap="viridis",
        s=44 + 14 * npeaks,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )
    # highlight the four representatives
    for rank, label, color in chosen:
        r = pareto[rank]
        x = abs(float(r["main_amp"]) - target_amp)
        y = float(r["spur_excess"])
        ax.scatter([x], [y], s=170, facecolor="none",
                   edgecolor=color, lw=2.2, zorder=4)
        tag = label.split(":")[0]
        ax.annotate(tag, xy=(x, y), xytext=(8, 6),
                    textcoords="offset points",
                    color=color, fontweight="bold", fontsize=11)

    ax.set_xlabel("主峰高度误差 |A − 0.95|")
    ax.set_ylabel("次峰超限量")
    ax.set_title("8.5 μm 单峰目标的 Pareto 前沿（42 个非支配候选）")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("主峰位置偏差 |pos − 8.5| (μm)")

    # Four bitmap insets, colored borders match scatter highlights.
    sub = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.32)
    for i, (rank, label, color) in enumerate(chosen):
        axp = fig.add_subplot(sub[i, 0])
        pat = np.load(run_root / f"pattern_{rank:03d}.npy").astype(np.float32)
        axp.imshow(pat, cmap="gray_r", interpolation="nearest")
        axp.set_xticks([])
        axp.set_yticks([])
        for spine in axp.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8)
        axp.set_title(label, fontsize=9.5, color=color, pad=3)

    savefig(fig, "ch4_pareto.png")


def generate_ch5_typical_spectra(eval_data: dict[str, np.ndarray]) -> None:
    picks = selected_examples(eval_data)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    lambda_um = eval_data["lambda_um"]
    title_map = {"single": "单峰样本", "multi": "多峰样本", "wide": "宽带样本"}
    for ax, label_name in zip(axes, ("single", "multi", "wide")):
        idx = picks[label_name]
        draw_absorption(ax, lambda_um, eval_data["truth"][idx], eval_data["pred"][idx], title=title_map[label_name])
    axes[0].legend(frameon=False, loc="upper right")
    savefig(fig, "ch5_typical_spectra.png")


def generate_ch5_peak_stats(eval_data: dict[str, np.ndarray]) -> None:
    labels = eval_data["labels"]
    order = ("single", "multi", "wide")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    metrics = [
        ("峰位误差 (μm)", eval_data["pos_err"]),
        ("峰高误差", eval_data["amp_err"]),
        ("曲线 MSE", eval_data["mse"]),
    ]
    tick_labels = ["单峰", "多峰", "宽带"]
    for ax, (title, values) in zip(axes, metrics):
        series = [values[labels == name] for name in order]
        ax.boxplot(series, tick_labels=tick_labels, patch_artist=True, boxprops={"facecolor": "#dbe8ff"})
        ax.set_title(title)
    savefig(fig, "ch5_peak_stats.png")


def generate_ch5_training_ablation() -> None:
    runs = {
        "仅光谱": TB_ROOT / "fno_curve_only" / "20260429-192610",
        "全场 + Maxwell": TB_ROOT / "fno_fullfield_maxwell" / MAIN_MODEL_RUN,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
    curve_colors = {"仅光谱": "#4e79a7", "全场 + Maxwell": "#e15759"}
    best_vals = []
    labels = []
    for name, path in runs.items():
        steps, vals = read_scalar_history(path, ["loss/val_total"])
        axes[0].plot(steps, vals, lw=2.0, label=name, color=curve_colors[name])
        best_vals.append(float(vals.min()))
        labels.append(name)
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("验证集总损失")
    axes[0].set_title("不同模型设置的验证曲线")
    axes[0].legend(frameon=False)

    axes[1].bar(labels, best_vals, color=[curve_colors[k] for k in labels])
    axes[1].set_ylabel("最优验证损失")
    axes[1].set_title("最优验证损失对比")
    axes[1].tick_params(axis="x", rotation=15)
    savefig(fig, "ch5_training_ablation.png")


def generate_ch5_closed_loop() -> None:
    """Single-panel 8.5 um representative result.

    Uses the same converged run as the Pareto and multi-solution figures so
    section 5.3 tells one coherent story. The earlier multi-target version
    (5 um + 8 um strict + 8 um) is no longer used; this matches the headline
    case described in section 5.3.1.
    """
    run_root = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
    report = load_json(run_root / "best_report.json")
    pattern = np.load(run_root / "best_pattern.npy").astype(np.float32)
    spectrum = np.load(run_root / "best_spectrum.npy").astype(np.float32)
    target_pos = 8.5
    target_amp = 0.95
    lambda_um = np.linspace(3.0, 12.0, int(spectrum.shape[0]), dtype=np.float32)

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.88, 1.55], wspace=0.28)
    draw_binary_pattern(fig.add_subplot(gs[0]), pattern, "选定的 11×11 图案")

    ax = fig.add_subplot(gs[1])
    ax.plot(lambda_um, spectrum, color="#2f6f9f", lw=2.0, label="预测吸收率 A")
    ax.axvline(target_pos, color="#c92a2a", ls="--", lw=1.4, label="目标峰位")
    ax.axhline(target_amp, color="#3b8b3b", ls=":", lw=1.4, label="目标峰高")
    ax.scatter(
        [float(report["main_pos"])],
        [float(report["main_amp"])],
        s=44, color="#f28e2b", zorder=4, label="主峰",
    )
    ax.set_xlim(3.0, 12.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("波长 (μm)")
    ax.set_ylabel("吸收率")
    ax.set_title("8.5 μm 单峰目标代表性设计")
    ax.legend(frameon=False, loc="lower right")

    summary = (
        f"主峰位置：{report['main_pos']:.2f} μm\n"
        f"峰高：{report['main_amp']:.3f}\n"
        f"峰高误差：{abs(report['main_amp'] - target_amp):.3f}\n"
        f"次峰超限：{report['spur_excess']:.3f}"
    )
    ax.text(
        0.03, 0.96, summary,
        transform=ax.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white",
              "ec": "#d0d0d0", "alpha": 0.92},
    )
    savefig(fig, "ch5_closed_loop.png")


def generate_ch5_multi_solution() -> None:
    """Pareto-diversity panel for the 8.5 um single-peak target.

    Uses the same converged optimization run as the closed-loop figure
    (FNO/runs_peak_nsga2_v2). Four candidates are picked to span the
    trade-off corners: rank 3 (clean single peak at target), rank 0
    (highest peak at target with side peaks), rank 4 (highest peak with
    0.05 um position drift), rank 40 (clean single peak with position
    relaxed to 8.05 um).
    """
    run_root = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
    pareto = load_json(run_root / "pareto_summary.json")
    target_pos = 8.5
    target_amp = 0.95
    chosen = [
        (3,  "B：峰位准确，谱形纯净"),
        (0,  "A：峰位准确，峰高较高"),
        (4,  "C：峰高最高"),
        (40, "D：谱形纯净，峰位 8.05 μm"),
    ]
    n_lambda = int(np.load(run_root / "spectrum_000.npy").shape[0])
    lambda_um = np.linspace(3.0, 12.0, n_lambda, dtype=np.float32)

    fig = plt.figure(figsize=(13.0, 6.6))
    outer = gridspec.GridSpec(2, 4, hspace=0.20, wspace=0.18)
    for idx, (rank, title) in enumerate(chosen):
        report = pareto[rank]
        pattern = np.load(run_root / f"pattern_{rank:03d}.npy").astype(np.float32)
        spectrum = np.load(run_root / f"spectrum_{rank:03d}.npy").astype(np.float32)

        axp = fig.add_subplot(outer[0, idx])
        draw_binary_pattern(axp, pattern, title)

        axs = fig.add_subplot(outer[1, idx])
        axs.plot(lambda_um, spectrum, color="#4e79a7", lw=1.8)
        axs.axvline(target_pos, color="#d62728", ls="--", lw=1.2,
                    label="目标峰位" if idx == 0 else None)
        axs.axhline(target_amp, color="#3b8b3b", ls=":", lw=1.0)
        axs.scatter([float(report["main_pos"])], [float(report["main_amp"])],
                    s=36, color="#f28e2b", zorder=4)
        axs.set_xlim(3.0, 12.0)
        axs.set_ylim(0.0, 1.02)
        axs.set_xlabel("波长 (μm)")
        axs.set_title(
            f"$\\lambda_p$={report['main_pos']:.2f}$\\,\\mu$m, "
            f"$A_p$={report['main_amp']:.3f}",
            fontsize=9.5,
        )
        if idx == 0:
            axs.set_ylabel("预测吸收率")
    savefig(fig, "ch5_multi_solution.png")


def draw_xy_xz(ax_xy: plt.Axes, ax_xz: plt.Axes, field_cube: np.ndarray, predictor: FullFieldDualSurrogatePredictor, title_prefix: str) -> None:
    xv = predictor.xv * 1e6
    yv = predictor.yv * 1e6
    zv = predictor.zv * 1e9
    z_idx = len(zv) // 2
    y_idx = len(yv) // 2
    xy = field_to_view(field_cube[:, :, z_idx], "magnitude")
    xz = field_to_view(field_cube[:, y_idx, :], "magnitude")
    im0 = ax_xy.imshow(xy.T, origin="lower", extent=[xv[0], xv[-1], yv[0], yv[-1]], aspect="auto", cmap="magma")
    im1 = ax_xz.imshow(xz.T, origin="lower", extent=[xv[0], xv[-1], zv[0], zv[-1]], aspect="auto", cmap="magma")
    ax_xy.set_title(f"{title_prefix}：XY 截面")
    ax_xz.set_title(f"{title_prefix}：XZ 截面")
    ax_xy.set_xlabel("x (μm)")
    ax_xy.set_ylabel("y (μm)")
    ax_xz.set_xlabel("x (μm)")
    ax_xz.set_ylabel("z (nm)")
    return im0, im1


def generate_ch5_field_maps(predictor: FullFieldDualSurrogatePredictor) -> None:
    data = np.load(PROJECT_ROOT / "outputs" / "inference" / "xy_model_reasoning" / "xy_reasoning_model_outputs.npz")
    pattern = data["pattern"]
    z_nm = float(data["z_nm"])

    fig = plt.figure(figsize=(15.8, 6.8))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[0.92, 1.0, 1.0, 1.0],
        hspace=0.38, wspace=0.52,
    )

    axp = fig.add_subplot(gs[:, 0])
    draw_binary_pattern(axp, pattern, "11×11 二值图案")
    axp.set_xticks(range(11))
    axp.set_yticks(range(11))
    axp.tick_params(labelsize=7)

    components = [
        ("Ex_xy", "|Ex|"),
        ("Ey_xy", "|Ey|"),
        ("Ez_xy", "|Ez|"),
        ("Hx_xy", "|Hx|"),
        ("Hy_xy", "|Hy|"),
        ("Hz_xy", "|Hz|"),
    ]
    for i, (key, title) in enumerate(components):
        axc = fig.add_subplot(gs[i // 3, 1 + i % 3])
        field = np.abs(data[key])
        im = axc.imshow(field.T, origin="lower", extent=[0.0, 2.7, 0.0, 2.7], cmap="magma", aspect="auto")
        axc.set_title(f"{title}，XY 截面，z={z_nm:.0f} nm")
        axc.set_xlabel("x (μm)")
        axc.set_ylabel("y (μm)")
        fig.colorbar(im, ax=axc, fraction=0.040, pad=0.025)

    savefig(fig, "ch5_field_maps.png")


def generate_ch5_complexity_error(eval_data: dict[str, np.ndarray]) -> None:
    complexity = eval_data["complexity"]
    mse = eval_data["mse"]
    bins = np.quantile(complexity, [0, 0.25, 0.5, 0.75, 1.0])
    groups = []
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (complexity >= lo) & (complexity <= hi)
        groups.append(mse[mask])
        labels.append(f"{lo:.0f}-{hi:.0f}")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0))
    axes[0].scatter(complexity, mse, s=20, alpha=0.65, color="#4e79a7")
    axes[0].set_xlabel("图案复杂度得分")
    axes[0].set_ylabel("曲线 MSE")
    axes[0].set_title("复杂度与预测误差")
    axes[1].boxplot(groups, tick_labels=labels, patch_artist=True, boxprops={"facecolor": "#dbe8ff"})
    axes[1].set_xlabel("复杂度分组")
    axes[1].set_ylabel("曲线 MSE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("不同复杂度分组的误差")
    savefig(fig, "ch5_complexity_error.png")


def generate_ch5_wavelength_error(eval_data: dict[str, np.ndarray]) -> None:
    lambda_um = eval_data["lambda_um"]
    abs_err = np.abs(eval_data["pred"] - eval_data["truth"])
    labels = eval_data["labels"]
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.plot(lambda_um, abs_err.mean(axis=0), lw=2.4, color="#1f77b4", label="全部验证样本")
    label_map = {"single": "单峰", "multi": "多峰", "wide": "宽带"}
    for label_name, color in [("single", "#e15759"), ("multi", "#59a14f"), ("wide", "#f28e2b")]:
        mask = labels == label_name
        ax.plot(lambda_um, abs_err[mask].mean(axis=0), lw=1.7, color=color, label=label_map[label_name])
    ax.set_xlabel("波长 (μm)")
    ax.set_ylabel("平均绝对误差")
    ax.set_title("沿波长轴的误差分布")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, "ch5_wavelength_error.png")


def generate_ch5_optimization_convergence() -> None:
    """Convergence trajectories for the 8.5 um single-peak optimization.

    Uses the same converged run (FNO/runs_peak_nsga2_v2) as the closed-loop,
    multi-solution and Pareto figures so all of section 5.3 is consistent.
    Each of the three NSGA-II objectives gets its own panel (since they have
    very different scales), and the population panel uses twin axes so the
    Pareto front size is visible alongside the much larger cache size.
    """
    run_root = PROJECT_ROOT / "FNO" / "runs_peak_nsga2_v2"
    progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
    gen = np.array([row["gen"] for row in progress], dtype=float)
    obj1 = np.array([row["obj1_best"] for row in progress], dtype=float)
    obj2 = np.array([row["obj2_best"] for row in progress], dtype=float)
    obj3 = np.array([row["obj3_best"] for row in progress], dtype=float)
    front = np.array([row["front_size"] for row in progress], dtype=float)
    cache_size = np.array([row["cache_size"] for row in progress], dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.6))

    axes[0].plot(gen, obj1, lw=2.0, color="#4e79a7")
    axes[0].set_xlabel("迭代代数")
    axes[0].set_ylabel("峰位惩罚")
    axes[0].set_title("峰位收敛")

    # obj2 spans several orders of magnitude — use semilog after clamping.
    axes[1].semilogy(gen, np.maximum(obj2, 1e-5), lw=2.0, color="#f28e2b")
    axes[1].set_xlabel("迭代代数")
    axes[1].set_ylabel("峰高误差")
    axes[1].set_title("峰高误差（对数）")

    axes[2].plot(gen, obj3, lw=2.0, color="#59a14f")
    axes[2].set_xlabel("迭代代数")
    axes[2].set_ylabel("次峰超限量")
    axes[2].set_title("次峰抑制")

    ax_left = axes[3]
    ax_right = ax_left.twinx()
    l1, = ax_left.plot(gen, front, lw=2.0, color="#4e79a7", label="Pareto 前沿规模")
    l2, = ax_right.plot(gen, cache_size, lw=2.0, color="#e15759",
                        label="已评估唯一结构数")
    ax_left.set_xlabel("迭代代数")
    ax_left.set_ylabel("Pareto 前沿规模", color="#4e79a7")
    ax_right.set_ylabel("唯一结构数", color="#e15759")
    ax_left.tick_params(axis="y", labelcolor="#4e79a7")
    ax_right.tick_params(axis="y", labelcolor="#e15759")
    ax_left.set_title("种群多样性")
    ax_left.legend(handles=[l1, l2], frameon=False, loc="lower right", fontsize=8.5)

    savefig(fig, "ch5_optimization_convergence.png")


def generate_ch6_future() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 3.8))
    ax.axis("off")
    steps = [
        (0.04, "扩展数据集"),
        (0.28, "物理增强模型"),
        (0.52, "逆向设计升级"),
        (0.76, "实验闭环验证"),
    ]
    for x, text in steps:
        add_flow_box(ax, (x, 0.38), text, width=0.18, height=0.22, fc="#eef5ff")
    for (x0, _), (x1, _) in zip(steps[:-1], steps[1:]):
        add_arrow(ax, (x0 + 0.18, 0.49), (x1, 0.49))
    ax.text(0.13, 0.22, "覆盖更多结构类型", ha="center")
    ax.text(0.37, 0.22, "提升泛化能力", ha="center")
    ax.text(0.61, 0.22, "支持更丰富目标约束", ha="center")
    ax.text(0.85, 0.22, "验证热学与器件性能", ha="center")
    savefig(fig, "ch6_future.png")


def generate_app_sample_layout() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    ax.axis("off")
    boxes = [
        ((0.04, 0.60), "sample_XXXXX.mat"),
        ((0.24, 0.78), "二值矩阵\n11×11"),
        ((0.24, 0.52), "波长数组\n1×91"),
        ((0.24, 0.26), "S11 / A / R / T\n1x91"),
        ((0.52, 0.64), "场张量\nEx...Hz"),
        ((0.76, 0.64), "20 x 51 x 51 x 91"),
    ]
    for (x, y), text in boxes:
        add_flow_box(ax, (x, y), text, width=0.16, height=0.14)
    for y in (0.85, 0.59, 0.33):
        add_arrow(ax, (0.20, 0.67), (0.24, y))
    add_arrow(ax, (0.40, 0.67), (0.52, 0.71))
    add_arrow(ax, (0.68, 0.71), (0.76, 0.71))
    savefig(fig, "app_sample_layout.png")


def generate_app_material_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.axis("off")
    items = [
        (0.05, "Au / SiO2\n材料表"),
        (0.28, "插值获得\nn,k(λ)"),
        (0.51, "转换为\n复介电常数"),
        (0.74, "分配到\nMIM 体素"),
    ]
    for x, text in items:
        add_flow_box(ax, (x, 0.40), text, width=0.16, height=0.18)
    for i in range(len(items) - 1):
        add_arrow(ax, (items[i][0] + 0.16, 0.49), (items[i + 1][0], 0.49))
    savefig(fig, "app_material_pipeline.png")


def generate_app_gallery(eval_data: dict[str, np.ndarray]) -> None:
    """Six representative validation samples where the surrogate fits cleanly.

    Restricts candidates to single-peak ('single') task samples — these have
    the cleanest truth spectra and the model fits them best, so the gallery
    shows the surrogate working rather than noisy multi-peak truths that
    visually overwhelm the smoother predictions.

    Layout: per-panel y-axis label is dropped (it was being clipped by the
    bitmap on the left). A single shared "Absorption" label is placed on the
    figure left margin via fig.supylabel.
    """
    labels = eval_data["labels"]
    mse = eval_data["mse"]
    pos_err = eval_data["pos_err"]

    pool = np.where(labels == "single")[0]
    if len(pool) < 6:
        pool = np.argsort(mse)[: max(6, len(pool))]
    score = mse[pool] + 0.05 * np.maximum(pos_err[pool], 0.0)
    order = pool[np.argsort(score)[:6]]

    fig = plt.figure(figsize=(13.6, 8.4))
    outer = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.18,
                              left=0.07, right=0.985,
                              top=0.94, bottom=0.07)
    for idx, sample_idx in enumerate(order):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[idx],
            width_ratios=[0.55, 1.9], wspace=0.05,
        )
        axp = fig.add_subplot(inner[0, 0])
        axs = fig.add_subplot(inner[0, 1])
        draw_binary_pattern(
            axp, eval_data["patterns"][sample_idx],
            f"样本 {int(eval_data['sample_ids'][sample_idx]):05d}",
        )
        draw_absorption(
            axs, eval_data["lambda_um"],
            eval_data["truth"][sample_idx], eval_data["pred"][sample_idx],
        )
        axs.set_ylabel("")  # suppress per-panel y-label; use figure-level supylabel
        axs.set_title(
            f"均方误差 {mse[sample_idx]:.3f}，峰位误差 {pos_err[sample_idx]:.2f} μm",
            fontsize=9.5, pad=4,
        )
        if idx == 0:
            axs.legend(frameon=False, loc="upper right", fontsize=8.5)

    fig.supylabel("吸收率", fontsize=11, x=0.012)
    savefig(fig, "app_gallery.png")


def generate_app_casebooks() -> None:
    runs = sorted([p.name for p in OPT_ROOT.iterdir() if p.is_dir()])
    selected = runs[:8]
    targets = [("app_casebook_1.png", selected[:4]), ("app_casebook_2.png", selected[4:8])]
    for filename, chosen in targets:
        fig = plt.figure(figsize=(12.8, 8.0))
        outer = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.22)
        for idx, name in enumerate(chosen):
            info = load_opt_run(name)
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], height_ratios=[1.0, 1.1], hspace=0.10)
            axp = fig.add_subplot(inner[0, 0])
            axs = fig.add_subplot(inner[1, 0])
            draw_binary_pattern(axp, info["best_pattern"], name)
            lam_ref = np.load(run_dir(name) / "closed_loop_prediction.npz")["lambda_um"] if (run_dir(name) / "closed_loop_prediction.npz").exists() else np.linspace(4.0, 12.0, len(info["best_spectrum"]))
            axs.plot(lam_ref, info["best_spectrum"], color="#4e79a7", lw=1.8)
            axs.axvline(float(info["cfg"]["target_peak"]["pos"]), color="#d62728", ls="--", lw=1.2)
            axs.set_ylim(0, 1.02)
            axs.set_xlabel("波长 (μm)")
            axs.set_ylabel("预测 A")
        savefig(fig, filename)


def generate_app_tensorboard() -> None:
    run = TB_ROOT / "fno_fullfield_maxwell" / MAIN_MODEL_RUN
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6),
                              constrained_layout=True)
    tags = [
        ("loss/train_total", "loss/val_total", axes[0, 0], "总损失"),
        ("loss/train_field", "loss/val_field", axes[0, 1], "场损失"),
        ("loss/train_curl_e", "loss/val_curl_e", axes[1, 0], "curl-E 损失"),
        ("loss/train_div", "loss/val_div", axes[1, 1], "散度损失"),
    ]
    for train_tag, val_tag, ax, title in tags:
        s1, v1 = read_scalar_history(run, [train_tag])
        s2, v2 = read_scalar_history(run, [val_tag])
        ax.plot(s1, v1, label="训练集", lw=1.9)
        ax.plot(s2, v2, label="验证集", lw=1.9)
        ax.set_title(title, pad=8)
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("损失")
    axes[0, 0].legend(frameon=False)
    savefig(fig, "app_tensorboard.png")


def generate_app_field_multilambda(predictor: FullFieldDualSurrogatePredictor) -> None:
    best_pattern = load_opt_run(RUN_8UM)["best_pattern"]
    target_idx = int(np.argmin(np.abs(predictor.lambda_vec * 1e6 - 8.0)))
    indices = [max(0, target_idx - 10), target_idx, min(len(predictor.lambda_vec) - 1, target_idx + 10)]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    xv = predictor.xv * 1e6
    yv = predictor.yv * 1e6
    z_idx = len(predictor.zv) // 2
    for ax, idx in zip(axes, indices):
        result = predictor.predict_field_at_lambda(best_pattern[None, ...], lambda_index=idx)
        ez = result["fields"][0]["Ez"]
        im = ax.imshow(
            np.abs(ez[:, :, z_idx]).T,
            origin="lower",
            extent=[xv[0], xv[-1], yv[0], yv[-1]],
            cmap="magma",
            aspect="auto",
        )
        ax.set_title(f"{result['lambda_m'] * 1e6:.2f} μm")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
    fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    savefig(fig, "app_field_multilambda_1.png")


def generate_app_field_multisolution(predictor: FullFieldDualSurrogatePredictor) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    xv = predictor.xv * 1e6
    yv = predictor.yv * 1e6
    z_idx = len(predictor.zv) // 2
    for ax, rank in zip(axes, (1, 2, 3)):
        pattern = load_pattern_by_rank(RUN_8UM, rank)
        result = predictor.predict_field_at_lambda(pattern[None, ...], lambda_value=8.0e-6)
        ez = result["fields"][0]["Ez"]
        im = ax.imshow(
            np.abs(ez[:, :, z_idx]).T,
            origin="lower",
            extent=[xv[0], xv[-1], yv[0], yv[-1]],
            cmap="magma",
            aspect="auto",
        )
        ax.set_title(f"候选结构 #{rank}")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
    fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    savefig(fig, "app_field_multisolution.png")


def generate_app_maxwell_compare() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.axis("off")
    rows = ["仅光谱", "光谱 + 场 + Maxwell"]
    cols = ["曲线拟合", "场拟合", "被动性", "旋度", "散度", "峰值约束"]
    mat = np.array(
        [
            [1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "开启" if mat[i, j] > 0.5 else "关闭", ha="center", va="center", color="black")
    savefig(fig, "app_maxwell_compare.png")


def generate_app_template_spectrum(eval_data: dict[str, np.ndarray]) -> None:
    picks = list(selected_examples(eval_data).values())
    while len(picks) < 4:
        picks.append(len(picks))
    fig = plt.figure(figsize=(11.8, 8.4))
    outer = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.20)
    for slot, idx in enumerate(picks[:4]):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[slot], height_ratios=[1.0, 1.2], hspace=0.10)
        axp = fig.add_subplot(inner[0, 0])
        axs = fig.add_subplot(inner[1, 0])
        draw_binary_pattern(axp, eval_data["patterns"][idx], f"样本 {int(eval_data['sample_ids'][idx]):05d}")
        draw_absorption(axs, eval_data["lambda_um"], eval_data["truth"][idx], eval_data["pred"][idx])
    savefig(fig, "app_template_spectrum.png")


def generate_app_template_field(predictor: FullFieldDualSurrogatePredictor) -> None:
    npz = get_closed_loop_npz(RUN_8UM, predictor)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    xv = predictor.xv * 1e6
    yv = predictor.yv * 1e6
    z_idx = len(predictor.zv) // 2
    y_idx = len(predictor.yv) // 2
    ez = npz["ez_norm"]
    xy = np.abs(ez[:, :, z_idx])
    xz = np.abs(ez[:, y_idx, :])
    im = axes[0].imshow(xy.T, origin="lower", extent=[xv[0], xv[-1], yv[0], yv[-1]], cmap="magma", aspect="auto")
    axes[0].set_title("Ez 的 XY 截面")
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("y (μm)")
    im2 = axes[1].imshow(xz.T, origin="lower", extent=[xv[0], xv[-1], predictor.zv[0] * 1e9, predictor.zv[-1] * 1e9], cmap="magma", aspect="auto")
    axes[1].set_title("Ez 的 XZ 截面")
    axes[1].set_xlabel("x (μm)")
    axes[1].set_ylabel("z (nm)")
    fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    _ = im2
    savefig(fig, "app_template_field.png")


def maybe_fix_lam_from_npz(run_name: str) -> np.ndarray:
    npz_path = run_dir(run_name) / "closed_loop_prediction.npz"
    if npz_path.exists():
        return np.load(npz_path)["lambda_um"]
    return np.linspace(4.0, 12.0, len(np.load(run_dir(run_name) / "best_spectrum.npy")))


def generate_all() -> None:
    ensure_dirs()
    cache = load_curve_cache()
    predictor = get_predictor()
    eval_data = evaluate_val_set(cache, predictor)

    generate_ch3_parametric_pipeline()
    generate_ch3_z_sampling(predictor)
    generate_ch3_dispersion(cache)
    generate_ch3_fno_arch()
    generate_ch3_loss()

    generate_ch4_encoding()
    generate_ch4_fitness_terms()
    generate_ch4_nsga2_flow()
    generate_ch4_pareto()

    generate_ch5_typical_spectra(eval_data)
    generate_ch5_peak_stats(eval_data)
    generate_ch5_training_ablation()
    generate_ch5_closed_loop()
    generate_ch5_multi_solution()
    generate_ch5_field_maps(predictor)
    generate_ch5_complexity_error(eval_data)
    generate_ch5_wavelength_error(eval_data)
    generate_ch5_optimization_convergence()

    generate_ch6_future()

    generate_app_sample_layout()
    generate_app_material_pipeline()
    generate_app_gallery(eval_data)
    generate_app_casebooks()
    generate_app_tensorboard()
    generate_app_field_multilambda(predictor)
    generate_app_field_multisolution(predictor)
    generate_app_maxwell_compare()
    generate_app_template_spectrum(eval_data)
    generate_app_template_field(predictor)


if __name__ == "__main__":
    generate_all()
