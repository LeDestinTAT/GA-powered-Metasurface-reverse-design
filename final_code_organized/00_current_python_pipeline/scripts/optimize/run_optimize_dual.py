import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

try:
    from deap import base, creator, tools
except ImportError as exc:
    raise ImportError("缺少依赖 deap。请先安装：pip install deap") from exc

from src.checkpoint_utils import resolve_checkpoint_choice
from src.fullfield_dual_surrogate import FullFieldDualSurrogatePredictor
from src.project_paths import BEST_MODEL_HISTORY_ROOT, MODELS_CURRENT_DIR, OPTIMIZATION_OUTPUTS_DIR, SAMPLING_META_PATH, ensure_standard_dirs


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def smooth_curve(y, k=5):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if k <= 1:
        return y
    kernel = np.ones(k, dtype=np.float32) / float(k)
    y_pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def find_peaks_np(y, min_height=0.08, min_prom_ratio=0.08, min_distance=2):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    ymax = float(np.max(y)) if len(y) > 0 else 0.0
    if ymax <= 1e-8:
        return np.array([], dtype=np.int64)

    height = max(float(min_height), 0.10 * ymax)
    prominence = max(0.02, float(min_prom_ratio) * ymax)
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(y, height=height, prominence=prominence, distance=int(min_distance))
        return peaks.astype(np.int64)
    except Exception:
        idx = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= height:
                idx.append(i)
        return np.array(idx, dtype=np.int64)


def extract_peaks(lambda_vec, A, peak_cfg):
    A_used = smooth_curve(A, k=int(peak_cfg.get("smooth_k", 5)))
    idx = find_peaks_np(
        A_used,
        min_height=float(peak_cfg.get("min_height", 0.08)),
        min_prom_ratio=float(peak_cfg.get("min_prom_ratio", 0.08)),
        min_distance=int(peak_cfg.get("min_distance", 2)),
    )
    peaks = [{"idx": int(i), "pos": float(lambda_vec[i]), "amp": float(A_used[i])} for i in idx]
    peaks = sorted(peaks, key=lambda p: p["amp"], reverse=True)
    return peaks, A_used


def interpolate_crossing(x1, y1, x2, y2, y_target):
    if abs(y2 - y1) < 1e-12:
        return 0.5 * (x1 + x2)
    t = (y_target - y1) / (y2 - y1)
    t = float(np.clip(t, 0.0, 1.0))
    return float(x1 + t * (x2 - x1))


def estimate_fwhm(lambda_vec, y, peak_idx):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)
    peak_idx = int(peak_idx)

    if y.size < 3 or peak_idx <= 0 or peak_idx >= y.size - 1:
        return None

    peak_amp = float(y[peak_idx])
    if peak_amp <= 1e-8:
        return None

    half_level = 0.5 * peak_amp

    left_cross = None
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= half_level:
            j = min(i + 1, peak_idx)
            left_cross = interpolate_crossing(lambda_vec[i], y[i], lambda_vec[j], y[j], half_level)
            break

    right_cross = None
    for i in range(peak_idx + 1, y.size):
        if y[i] <= half_level:
            j = max(i - 1, peak_idx)
            right_cross = interpolate_crossing(lambda_vec[j], y[j], lambda_vec[i], y[i], half_level)
            break

    if left_cross is None or right_cross is None or right_cross < left_cross:
        return None
    return float(right_cross - left_cross)


def compute_objectives(lambda_vec, A, cfg):
    peak_cfg = cfg["peak_detect"]
    fit_cfg = cfg["fitness"]
    target_cfg = cfg["target_peak"]
    secondary_cfg = cfg.get("secondary_peak", {})
    fwhm_cfg = cfg.get("fwhm", {})

    A = np.clip(np.asarray(A, dtype=np.float32), 0.0, 1.0)
    peaks, A_used = extract_peaks(lambda_vec, A, peak_cfg)

    if len(peaks) == 0:
        big = float(fit_cfg["missing_peak_penalty"])
        return (big, big, big, big, big), {
            "main_pos": None,
            "main_amp": None,
            "n_peaks": 0,
            "secondary_peak_amp": None,
            "secondary_peak_excess": big,
            "max_spur_amp": None,
            "spur_ratio": None,
            "spur_excess": big,
            "main_fwhm": None,
            "fwhm_err": big,
            "peaks": [],
        }

    main_peak = peaks[0]
    pos_err = abs(main_peak["pos"] - float(target_cfg["pos"]))
    amp_err = abs(main_peak["amp"] - float(target_cfg["amp"]))
    max_pos_err = target_cfg.get("max_pos_err")
    min_amp = target_cfg.get("min_amp")
    peak_hard_weight = float(fit_cfg.get("w_peak_hard", 0.0))
    pos_hard_excess = max(0.0, pos_err - float(max_pos_err)) if max_pos_err is not None else 0.0
    amp_hard_excess = max(0.0, float(min_amp) - float(main_peak["amp"])) if min_amp is not None else 0.0
    peak_hard_penalty = peak_hard_weight * (pos_hard_excess ** 2 + amp_hard_excess ** 2)
    secondary_peak_amp = float(peaks[1]["amp"]) if len(peaks) >= 2 else 0.0
    secondary_peak_excess = max(0.0, secondary_peak_amp - float(secondary_cfg.get("max_amp", 0.2)))
    max_spur_amp = max((p["amp"] for p in peaks[1:]), default=0.0)
    spur_ratio = max_spur_amp / max(main_peak["amp"], 1e-8)
    spur_excess = max(0.0, spur_ratio - float(cfg["spur_ratio_max"]))
    main_fwhm = estimate_fwhm(lambda_vec, A_used, main_peak["idx"])
    if main_fwhm is None:
        fwhm_err = float(fwhm_cfg.get("missing_penalty", fit_cfg["missing_peak_penalty"]))
    else:
        target_fwhm = fwhm_cfg.get("target")
        if target_fwhm is not None:
            fwhm_err = abs(float(main_fwhm) - float(target_fwhm))
        else:
            fwhm_err = 0.0

        min_fwhm = fwhm_cfg.get("min")
        max_fwhm = fwhm_cfg.get("max")
        if min_fwhm is not None:
            fwhm_err += max(0.0, float(min_fwhm) - float(main_fwhm))
        if max_fwhm is not None:
            fwhm_err += max(0.0, float(main_fwhm) - float(max_fwhm))

    obj1 = float(fit_cfg["w_pos"]) * pos_err + peak_hard_penalty
    obj2 = float(fit_cfg["w_amp"]) * amp_err + peak_hard_penalty
    obj3 = float(fit_cfg.get("w_secondary", 1.0)) * secondary_peak_excess
    obj4 = float(fit_cfg.get("w_fwhm", 1.0)) * fwhm_err
    obj5 = float(fit_cfg["w_spur"]) * spur_excess
    info = {
        "main_pos": main_peak["pos"],
        "main_amp": main_peak["amp"],
        "pos_err": pos_err,
        "amp_err": amp_err,
        "pos_hard_excess": pos_hard_excess,
        "amp_hard_excess": amp_hard_excess,
        "peak_hard_penalty": peak_hard_penalty,
        "n_peaks": len(peaks),
        "secondary_peak_amp": secondary_peak_amp,
        "secondary_peak_excess": secondary_peak_excess,
        "max_spur_amp": max_spur_amp,
        "spur_ratio": spur_ratio,
        "spur_excess": spur_excess,
        "main_fwhm": main_fwhm,
        "fwhm_err": fwhm_err,
        "peaks": peaks,
    }
    return (obj1, obj2, obj3, obj4, obj5), info


class Pixel11x11Encoder:
    def __init__(self, height=11, width=11, symmetry="none"):
        self.height = int(height)
        self.width = int(width)
        self.symmetry = str(symmetry)

    @property
    def genome_length(self):
        return self.height * self.width

    def sample(self):
        return np.random.randint(0, 2, size=(self.genome_length,), dtype=np.int64).tolist()

    def repair(self, ind):
        mat = np.array(ind, dtype=np.int64).reshape(self.height, self.width)
        mat = (mat > 0).astype(np.int64)

        if self.symmetry in ("h", "hv"):
            mat = np.maximum(mat, np.flipud(mat))
        if self.symmetry in ("v", "hv"):
            mat = np.maximum(mat, np.fliplr(mat))

        return mat.reshape(-1).tolist()

    def to_pattern_batch(self, inds):
        mats = []
        for ind in inds:
            fixed = self.repair(ind)
            mats.append(np.array(fixed, dtype=np.float32).reshape(self.height, self.width))
        return np.stack(mats, axis=0)

    def to_pattern(self, ind):
        fixed = self.repair(ind)
        return np.array(fixed, dtype=np.float32).reshape(self.height, self.width)


class FitnessCache:
    def __init__(self, max_size=100000):
        self.max_size = int(max_size)
        self.cache = {}

    def _key(self, ind):
        return tuple(int(x) for x in ind)

    def get(self, ind):
        return self.cache.get(self._key(ind), None)

    def set(self, ind, value):
        if len(self.cache) >= self.max_size:
            for k in list(self.cache.keys())[: max(1, self.max_size // 10)]:
                self.cache.pop(k, None)
        self.cache[self._key(ind)] = value


def format_as_matlab_matrix(mat, var_name="binary_matrix"):
    mat = np.asarray(mat)
    mat = (mat > 0.5).astype(int)
    lines = [f"{var_name} = ["]
    for i, row in enumerate(mat):
        row_str = " ".join(str(int(x)) for x in row)
        lines.append(f"    {row_str}{';' if i < mat.shape[0] - 1 else ''}")
    lines.append("];")
    return "\n".join(lines)


def evaluate_population(population, encoder, predictor, cfg, cache):
    values = [None] * len(population)
    uncached = []
    uncached_indices = []
    for i, ind in enumerate(population):
        cached = cache.get(ind)
        if cached is not None:
            values[i] = cached
        else:
            uncached.append(ind)
            uncached_indices.append(i)

    if uncached:
        pattern_batch = encoder.to_pattern_batch(uncached)
        A_batch, pred_s = predictor.predict_spectrum(pattern_batch)
        lambda_axis_um = predictor.lambda_vec * 1e6
        for local_i, global_i in enumerate(uncached_indices):
            A = A_batch[local_i]
            fit, info = compute_objectives(lambda_axis_um, A, cfg)
            result = {
                "fitness": fit,
                "info": info,
                "spectrum": A,
                "sparams": pred_s[local_i],
            }
            values[global_i] = result
            cache.set(population[global_i], result)

    return values


def save_results(run_dir, pareto, encoder, predictor, cfg):
    run_dir = Path(run_dir)
    lambda_axis_um = predictor.lambda_vec * 1e6
    for rank, ind in enumerate(pareto, start=1):
        mat = encoder.to_pattern(ind)
        A_batch, pred_s = predictor.predict_spectrum(mat[None, ...])
        A = A_batch[0]
        fit, info = compute_objectives(lambda_axis_um, A, cfg)

        np.save(run_dir / f"pattern_{rank:03d}.npy", mat)
        np.save(run_dir / f"spectrum_{rank:03d}.npy", A)
        np.save(run_dir / f"sparams_{rank:03d}.npy", pred_s[0])

        payload = {
            "rank": rank,
            "fitness": list(map(float, fit)),
            "main_pos": info["main_pos"],
            "main_amp": info["main_amp"],
            "n_peaks": info["n_peaks"],
            "secondary_peak_amp": info["secondary_peak_amp"],
            "secondary_peak_excess": info["secondary_peak_excess"],
            "main_fwhm": info["main_fwhm"],
            "fwhm_err": info["fwhm_err"],
            "max_spur_amp": info["max_spur_amp"],
            "spur_ratio": info["spur_ratio"],
            "spur_excess": info["spur_excess"],
        }
        with open(run_dir / f"report_{rank:03d}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def select_best_from_pareto(pareto, encoder, predictor, cfg):
    wcfg = cfg["best_selection"]
    wp = float(wcfg["w_pos"])
    wa = float(wcfg["w_amp"])
    wsec = float(wcfg.get("w_secondary", 1.0))
    wfwhm = float(wcfg.get("w_fwhm", 1.0))
    ws = float(wcfg["w_spur"])

    best_score = None
    best_ind = None
    best_fit = None
    best_info = None
    best_A = None
    best_s = None
    lambda_axis_um = predictor.lambda_vec * 1e6

    for ind in pareto:
        mat = encoder.to_pattern(ind)
        A_batch, pred_s = predictor.predict_spectrum(mat[None, ...])
        A = A_batch[0]
        fit, info = compute_objectives(lambda_axis_um, A, cfg)
        score = wp * fit[0] + wa * fit[1] + wsec * fit[2] + wfwhm * fit[3] + ws * fit[4]
        if best_score is None or score < best_score:
            best_score = score
            best_ind = ind
            best_fit = fit
            best_info = info
            best_A = A
            best_s = pred_s[0]

    return best_ind, best_fit, best_info, best_A, best_s, best_score


def build_optimizer_config(cfg):
    ensure_standard_dirs()

    model_cfg = dict(cfg.get("model", {}))
    resolved_ckpt_path, checkpoint_source = resolve_checkpoint_choice(
        str(model_cfg.get("choice", "current_best")),
        current_best=MODELS_CURRENT_DIR / str(model_cfg.get("current_best_name", "fno_fullfield_maxwell_dual_best.pt")),
        current_final=MODELS_CURRENT_DIR / str(model_cfg.get("current_final_name", "fno_fullfield_maxwell_dual_final.pt")),
        history_root=BEST_MODEL_HISTORY_ROOT,
        custom_path=model_cfg.get("custom_path"),
        run_name=model_cfg.get("run_name"),
        best_index=model_cfg.get("best_index"),
        project_root=PROJECT_ROOT,
    )

    cfg["resolved_ckpt_path"] = str(resolved_ckpt_path)
    cfg["checkpoint_source"] = checkpoint_source

    requested_device = cfg.get("device")
    if requested_device is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("[Warning] config 里请求了 cuda，但当前环境不可用，自动切换到 cpu")
        cfg["device"] = "cpu"

    output_root = Path(cfg.get("output_dir_root", OPTIMIZATION_OUTPUTS_DIR / "dual_nsga2"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = cfg.get("run_name") or time.strftime("%Y%m%d-%H%M%S")
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(output_root / run_name)

    cfg.setdefault("meta_path", str(SAMPLING_META_PATH))
    return cfg


def run_nsga2(cfg):
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", False)))

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

    print("[Model] loaded.")
    print("[Model] checkpoint =", cfg["resolved_ckpt_path"])
    print("[Model] source =", cfg["checkpoint_source"])
    print("[Model] lambda points =", len(predictor.lambda_vec))

    encoder = Pixel11x11Encoder(
        height=int(cfg["encoding"]["height"]),
        width=int(cfg["encoding"]["width"]),
        symmetry=str(cfg["encoding"].get("symmetry", "none")),
    )

    pop_size = int(cfg["nsga2"]["pop_size"])
    ngen = int(cfg["nsga2"]["ngen"])
    cxpb = float(cfg["nsga2"]["cxpb"])
    mutpb = float(cfg["nsga2"]["mutpb"])
    indpb = float(cfg["nsga2"]["indpb_bit"])
    cache = FitnessCache(max_size=int(cfg["nsga2"].get("cache_size", 100000)))

    fit_name = "FitnessMin5Dual"
    ind_name = "Individual11x11Dual"
    if not hasattr(creator, fit_name):
        creator.create(fit_name, base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
    if not hasattr(creator, ind_name):
        creator.create(ind_name, list, fitness=getattr(creator, fit_name))

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: getattr(creator, ind_name)(encoder.sample()))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    eval_results = evaluate_population(pop, encoder, predictor, cfg, cache)
    for ind, res in zip(pop, eval_results):
        ind.fitness.values = res["fitness"]

    pop = toolbox.select(pop, len(pop))
    pareto = tools.ParetoFront()
    pareto.update(pop)

    log = []
    t0 = time.time()
    for gen in range(1, ngen + 1):
        k = len(pop)
        k4 = k - (k % 4)
        offspring = tools.selTournamentDCD(pop, k4)
        if k4 < k:
            offspring += random.sample(pop, k - k4)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                ind1[:] = encoder.repair(ind1)
                ind2[:] = encoder.repair(ind2)
                if ind1.fitness.valid:
                    del ind1.fitness.values
                if ind2.fitness.valid:
                    del ind2.fitness.values

        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                ind[:] = encoder.repair(ind)
                if ind.fitness.valid:
                    del ind.fitness.values
            else:
                ind[:] = encoder.repair(ind)

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_inds:
            eval_results = evaluate_population(invalid_inds, encoder, predictor, cfg, cache)
            for ind, res in zip(invalid_inds, eval_results):
                ind.fitness.values = res["fitness"]

        pop = toolbox.select(pop + offspring, pop_size)
        pareto.update(pop)

        front0 = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        best_front_fit = [ind.fitness.values for ind in front0]
        obj1_best = min(v[0] for v in best_front_fit)
        obj2_best = min(v[1] for v in best_front_fit)
        obj3_best = min(v[2] for v in best_front_fit)
        obj4_best = min(v[3] for v in best_front_fit)
        obj5_best = min(v[4] for v in best_front_fit)

        log_item = {
            "gen": gen,
            "front_size": len(front0),
            "obj1_best": float(obj1_best),
            "obj2_best": float(obj2_best),
            "obj3_best": float(obj3_best),
            "obj4_best": float(obj4_best),
            "obj5_best": float(obj5_best),
            "cache_size": len(cache.cache),
            "elapsed_sec": float(time.time() - t0),
        }
        log.append(log_item)

        print(
            f"Gen {gen:03d} | front={len(front0)} | "
            f"best_pos_obj={obj1_best:.6f} | "
            f"best_amp_obj={obj2_best:.6f} | "
            f"best_secondary_obj={obj3_best:.6f} | "
            f"best_fwhm_obj={obj4_best:.6f} | "
            f"best_spur_obj={obj5_best:.6f} | "
            f"cache={len(cache.cache)}"
        )

    run_dir = Path(cfg["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(run_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    save_results(run_dir, pareto, encoder, predictor, cfg)

    best_ind, best_fit, best_info, best_A, best_s, best_score = select_best_from_pareto(pareto, encoder, predictor, cfg)
    if best_ind is not None:
        best_pattern = encoder.to_pattern(best_ind)
        np.save(run_dir / "best_pattern.npy", best_pattern)
        np.save(run_dir / "best_spectrum.npy", best_A)
        np.save(run_dir / "best_sparams.npy", best_s)

        best_report = {
            "fitness": list(map(float, best_fit)),
            "post_score": float(best_score),
            "main_pos": best_info["main_pos"],
            "main_amp": best_info["main_amp"],
            "pos_err": best_info.get("pos_err"),
            "amp_err": best_info.get("amp_err"),
            "pos_hard_excess": best_info.get("pos_hard_excess"),
            "amp_hard_excess": best_info.get("amp_hard_excess"),
            "peak_hard_penalty": best_info.get("peak_hard_penalty"),
            "n_peaks": best_info["n_peaks"],
            "secondary_peak_amp": best_info["secondary_peak_amp"],
            "secondary_peak_excess": best_info["secondary_peak_excess"],
            "main_fwhm": best_info["main_fwhm"],
            "fwhm_err": best_info["fwhm_err"],
            "max_spur_amp": best_info["max_spur_amp"],
            "spur_ratio": best_info["spur_ratio"],
            "spur_excess": best_info["spur_excess"],
            "best_selection": cfg["best_selection"],
            "checkpoint_source": cfg["checkpoint_source"],
            "resolved_ckpt_path": cfg["resolved_ckpt_path"],
        }
        with open(run_dir / "best_report.json", "w", encoding="utf-8") as f:
            json.dump(best_report, f, indent=2, ensure_ascii=False)
        with open(run_dir / "best_matrix.m", "w", encoding="utf-8") as f:
            f.write(format_as_matlab_matrix(best_pattern, var_name="binary_matrix"))

        print("\n[Best solution]")
        print(json.dumps(best_report, indent=2, ensure_ascii=False))

    return run_dir, predictor.lambda_vec * 1e6


def plot_best(run_dir, cfg, lambda_um):
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    best_pattern = np.load(run_dir / "best_pattern.npy")
    best_A = np.load(run_dir / "best_spectrum.npy")

    lambda_um = np.asarray(lambda_um, dtype=np.float32).reshape(-1)
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(lambda_um, best_A, label="Pred A")
    plt.axvline(float(cfg["target_peak"]["pos"]), linestyle="--", label="Target pos")
    plt.axhline(float(cfg["target_peak"]["amp"]), linestyle="--", label="Target amp")
    plt.xlabel("lambda (um)")
    plt.ylabel("A")
    plt.title("Best Predicted Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "best_spectrum.png", dpi=200)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow(best_pattern, cmap="gray")
    plt.title("Best 11x11 Pattern")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "best_pattern.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg = build_optimizer_config(cfg)

    print("[Config]")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    run_dir, lambda_vec = run_nsga2(cfg)
    plot_best(run_dir, cfg, lambda_vec)
    print(f"\nFinished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
