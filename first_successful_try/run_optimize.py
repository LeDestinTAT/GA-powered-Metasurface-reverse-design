import os
import json
import math
import time
import copy
import random
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from deap import base, creator, tools

# ==========================================================
# 1. 工具
# ==========================================================
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


def normalize_lambda(lam, lam_min, lam_max):
    return 2.0 * (lam - lam_min) / (lam_max - lam_min + 1e-12) - 1.0


def choose_gn_groups(width):
    for g in [8, 4, 2, 1]:
        if width % g == 0:
            return g
    return 1


# ==========================================================
# 2. 你的 FNO 模型定义（按训练代码重建）
# ==========================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)

        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def forward(self, x):
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            x_ft.shape[0], self.out_channels, x_ft.shape[2], x_ft.shape[3],
            dtype=torch.complex64, device=x.device
        )

        weight = torch.complex(self.weight_real, self.weight_imag)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            weight
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class LambdaFourierFeatures(nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        freqs = (2.0 ** torch.arange(n_freq).float()) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam_norm):
        x = lam_norm * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class FNOEncoder(nn.Module):
    def __init__(self, modes=6, width=64, depth=4):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.out_norm = nn.GroupNorm(choose_gn_groups(width), width)

    def forward(self, x):
        x = self.in_proj(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = self.act(spec(x) + pw(x))
        x = self.out_norm(x)
        return x.mean(dim=(-2, -1))


class FNO_LambdaConditional_SParams(nn.Module):
    def __init__(self, modes=6, width=64, depth=4, lam_ff=8, head_hidden=256):
        super().__init__()
        self.encoder = FNOEncoder(modes=modes, width=width, depth=depth)
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)

        head_in = width + 2 * lam_ff
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode_from_latent(self, z, lam_norm):
        B, K, _ = lam_norm.shape
        z_rep = z.unsqueeze(1).expand(B, K, z.shape[-1])
        le = self.lam_embed(lam_norm.reshape(B * K, 1)).reshape(B, K, -1)
        h = torch.cat([z_rep, le], dim=-1).reshape(B * K, -1)
        out = self.head(h).reshape(B, K, 4)
        return out

    def forward_curve(self, x, lam_norm):
        z = self.encode(x)
        return self.decode_from_latent(z, lam_norm)


# ==========================================================
# 3. checkpoint 加载
# ==========================================================
def load_peak_fno_checkpoint(ckpt_path, device="cpu"):
    """
    兼容 PyTorch 2.6+:
    1) 先尝试默认安全加载
    2) 若因 numpy 等对象失败，在“本地可信 checkpoint”前提下回退到 weights_only=False
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError as e:
        msg = str(e)
        if "Weights only load failed" in msg:
            print("[Info] torch.load 默认 weights_only=True 失败，尝试以 weights_only=False 重新加载（仅限可信 checkpoint）")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        else:
            raise

    if "state_dict" not in ckpt or "config" not in ckpt:
        raise RuntimeError("checkpoint 格式不匹配：必须包含 state_dict 和 config")

    cfg = ckpt["config"]
    state_dict = ckpt["state_dict"]

    model = FNO_LambdaConditional_SParams(
        modes=int(cfg["MODES"]),
        width=int(cfg["WIDTH"]),
        depth=int(cfg["DEPTH"]),
        lam_ff=int(cfg["LAM_FF"]),
        head_hidden=int(cfg["HEAD_HIDDEN"]),
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print("[Warning] missing keys:", missing)
    if len(unexpected) > 0:
        print("[Warning] unexpected keys:", unexpected)

    model.eval()

    if "lambda_vec" not in ckpt:
        raise RuntimeError("checkpoint 中没有 lambda_vec")

    lambda_vec = ckpt["lambda_vec"]
    if isinstance(lambda_vec, torch.Tensor):
        lambda_vec = lambda_vec.detach().cpu().numpy()
    else:
        lambda_vec = np.asarray(lambda_vec, dtype=np.float32)

    lambda_vec = lambda_vec.astype(np.float32).reshape(-1)

    meta = {
        "best_epoch": ckpt.get("best_epoch", None),
        "best_peak_score": ckpt.get("best_peak_score", None),
    }
    return model, lambda_vec, meta


# ==========================================================
# 4. surrogate 推理
# ==========================================================
@torch.no_grad()
def predict_absorption_batch(model, pattern_batch, lambda_vec, device):
    """
    pattern_batch: [B,1,11,11] numpy/torch
    return:
        A:     [B,K]
        predS: [B,K,4]
    """
    if isinstance(pattern_batch, np.ndarray):
        x = torch.tensor(pattern_batch, dtype=torch.float32, device=device)
    else:
        x = pattern_batch.to(device=device, dtype=torch.float32)

    lam_min = float(lambda_vec.min())
    lam_max = float(lambda_vec.max())

    lam_n = normalize_lambda(lambda_vec, lam_min, lam_max).astype(np.float32)
    lam_n = torch.tensor(lam_n, dtype=torch.float32, device=device)[None, :, None]
    lam_n = lam_n.expand(x.shape[0], -1, -1)  # [B,K,1]

    predS = model.forward_curve(x, lam_n)  # [B,K,4]

    re11, im11, re21, im21 = predS[..., 0], predS[..., 1], predS[..., 2], predS[..., 3]
    A = 1.0 - (re11**2 + im11**2 + re21**2 + im21**2)

    return A.detach().cpu().numpy().astype(np.float32), predS.detach().cpu().numpy().astype(np.float32)


# ==========================================================
# 5. 峰检测与 fitness
# ==========================================================
def find_peaks_np(y, min_height=0.08, min_prom_ratio=0.08, min_distance=2):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_max = float(np.max(y)) if len(y) > 0 else 0.0

    if y_max <= 1e-8:
        return np.array([], dtype=np.int64)

    height = max(min_height, 0.10 * y_max)
    prominence = max(0.02, min_prom_ratio * y_max)

    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            y,
            height=height,
            prominence=prominence,
            distance=min_distance
        )
        return peaks.astype(np.int64)
    except Exception:
        idx = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= height:
                idx.append(i)
        return np.array(idx, dtype=np.int64)


def smooth_curve(y, k=5):
    y = np.asarray(y, dtype=np.float32)
    if k <= 1:
        return y
    kernel = np.ones(k, dtype=np.float32) / float(k)
    y_pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def extract_peaks(lambda_vec, A, peak_cfg):
    A_used = smooth_curve(A, k=int(peak_cfg.get("smooth_k", 5)))
    idx = find_peaks_np(
        A_used,
        min_height=float(peak_cfg.get("min_height", 0.08)),
        min_prom_ratio=float(peak_cfg.get("min_prom_ratio", 0.08)),
        min_distance=int(peak_cfg.get("min_distance", 2)),
    )

    peaks = []
    for i in idx:
        peaks.append({
            "idx": int(i),
            "pos": float(lambda_vec[i]),
            "amp": float(A_used[i]),
        })

    peaks = sorted(peaks, key=lambda p: p["amp"], reverse=True)
    return peaks


def compute_objectives(lambda_vec, A, cfg):
    """
    返回两个目标：
      obj1 = 主峰位置误差
      obj2 = 主峰高度误差 + 杂峰惩罚
    越小越好
    """
    peak_cfg = cfg["peak_detect"]
    fit_cfg = cfg["fitness"]
    target_cfg = cfg["target_peak"]

    peaks = extract_peaks(lambda_vec, A, peak_cfg)

    if len(peaks) == 0:
        return (
            float(fit_cfg["missing_peak_penalty"]),
            float(fit_cfg["missing_peak_penalty"])
        ), {
            "main_pos": None,
            "main_amp": None,
            "n_peaks": 0,
            "spur_penalty": None,
            "peaks": []
        }

    main_peak = peaks[0]
    pos_err = abs(main_peak["pos"] - float(target_cfg["pos"]))
    amp_err = abs(main_peak["amp"] - float(target_cfg["amp"]))

    spur_limit = float(cfg["spur_ratio_max"]) * main_peak["amp"]
    spur_penalty = 0.0
    for p in peaks[1:]:
        spur_penalty += max(0.0, p["amp"] - spur_limit)

    obj1 = float(fit_cfg["w_pos"]) * pos_err
    obj2 = float(fit_cfg["w_amp"]) * amp_err + float(fit_cfg["w_spur"]) * spur_penalty

    info = {
        "main_pos": main_peak["pos"],
        "main_amp": main_peak["amp"],
        "n_peaks": len(peaks),
        "spur_penalty": spur_penalty,
        "peaks": peaks
    }
    return (obj1, obj2), info


# ==========================================================
# 6. 11x11 二值编码
# ==========================================================
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
            mat = np.array(fixed, dtype=np.float32).reshape(self.height, self.width)
            mats.append(mat)
        mats = np.stack(mats, axis=0)[:, None, :, :]  # [B,1,H,W]
        return mats

    def to_pattern(self, ind):
        fixed = self.repair(ind)
        return np.array(fixed, dtype=np.float32).reshape(self.height, self.width)


# ==========================================================
# 7. 缓存
# ==========================================================
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
            # 简单裁剪
            for k in list(self.cache.keys())[: max(1, self.max_size // 10)]:
                self.cache.pop(k, None)
        self.cache[self._key(ind)] = value


# ==========================================================
# 8. NSGA-II
# ==========================================================
def evaluate_population(population, encoder, model, lambda_vec, device, cfg, cache):
    values = [None] * len(population)
    uncached_inds = []
    uncached_idx = []

    for i, ind in enumerate(population):
        cached = cache.get(ind)
        if cached is not None:
            values[i] = cached
        else:
            uncached_inds.append(ind)
            uncached_idx.append(i)

    if len(uncached_inds) > 0:
        pattern_batch = encoder.to_pattern_batch(uncached_inds)
        A_batch, _ = predict_absorption_batch(model, pattern_batch, lambda_vec, device)

        for local_i, A in enumerate(A_batch):
            fit, info = compute_objectives(lambda_vec, A, cfg)
            result = {
                "fitness": fit,
                "info": info,
                "A": A
            }
            global_i = uncached_idx[local_i]
            values[global_i] = result
            cache.set(population[global_i], result)

    return values


def save_results(run_dir, pareto, encoder, model, lambda_vec, device, cfg):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for rank, ind in enumerate(pareto):
        mat = encoder.to_pattern(ind)
        A_batch, _ = predict_absorption_batch(model, mat[None, None, :, :], lambda_vec, device)
        A = A_batch[0]
        fit, info = compute_objectives(lambda_vec, A, cfg)

        np.save(run_dir / f"pattern_{rank:03d}.npy", mat)
        np.save(run_dir / f"spectrum_{rank:03d}.npy", A)

        all_records.append({
            "rank": rank,
            "fitness": list(map(float, fit)),
            "main_pos": info["main_pos"],
            "main_amp": info["main_amp"],
            "n_peaks": info["n_peaks"],
            "spur_penalty": info["spur_penalty"],
            "pattern_file": f"pattern_{rank:03d}.npy",
            "spectrum_file": f"spectrum_{rank:03d}.npy",
        })

    with open(run_dir / "pareto_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)


def run_nsga2(cfg):
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", False)))

    device = cfg["device"]
    model, lambda_vec, meta = load_peak_fno_checkpoint(cfg["ckpt_path"], device=device)
    print("[Model] loaded.")
    print("[Model] best_epoch =", meta["best_epoch"])
    print("[Model] best_peak_score =", meta["best_peak_score"])
    print("[Model] lambda points =", len(lambda_vec))

    encoder = Pixel11x11Encoder(
        height=int(cfg["encoding"]["height"]),
        width=int(cfg["encoding"]["width"]),
        symmetry=str(cfg["encoding"].get("symmetry", "none"))
    )

    pop_size = int(cfg["nsga2"]["pop_size"])
    ngen = int(cfg["nsga2"]["ngen"])
    cxpb = float(cfg["nsga2"]["cxpb"])
    mutpb = float(cfg["nsga2"]["mutpb"])
    indpb = float(cfg["nsga2"]["indpb_bit"])
    cache = FitnessCache(max_size=int(cfg["nsga2"].get("cache_size", 100000)))

    # 动态 creator，避免重复定义报错
    creator_name_fit = "FitnessMin2"
    creator_name_ind = "Individual11x11"
    if not hasattr(creator, creator_name_fit):
        creator.create(creator_name_fit, base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, creator_name_ind):
        creator.create(creator_name_ind, list, fitness=getattr(creator, creator_name_fit))

    toolbox = base.Toolbox()

    def init_individual():
        return getattr(creator, creator_name_ind)(encoder.sample())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)

    # 初始评估
    eval_results = evaluate_population(pop, encoder, model, lambda_vec, device, cfg, cache)
    for ind, res in zip(pop, eval_results):
        ind.fitness.values = res["fitness"]

    pop = toolbox.select(pop, len(pop))
    pareto = tools.ParetoFront()

    log = []
    t0 = time.time()

    for gen in range(1, ngen + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # crossover
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                ind1[:] = encoder.repair(ind1)
                ind2[:] = encoder.repair(ind2)
                if hasattr(ind1.fitness, "values"):
                    del ind1.fitness.values
                if hasattr(ind2.fitness, "values"):
                    del ind2.fitness.values

        # mutation
        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                ind[:] = encoder.repair(ind)
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values
            else:
                ind[:] = encoder.repair(ind)

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        if len(invalid_inds) > 0:
            eval_results = evaluate_population(invalid_inds, encoder, model, lambda_vec, device, cfg, cache)
            for ind, res in zip(invalid_inds, eval_results):
                ind.fitness.values = res["fitness"]

        pop = toolbox.select(pop + offspring, pop_size)
        pareto.update(pop)

        front0 = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        best_front_fit = [ind.fitness.values for ind in front0]
        obj1_best = min(v[0] for v in best_front_fit)
        obj2_best = min(v[1] for v in best_front_fit)

        log_item = {
            "gen": gen,
            "front_size": len(front0),
            "obj1_best": float(obj1_best),
            "obj2_best": float(obj2_best),
            "cache_size": len(cache.cache),
            "elapsed_sec": float(time.time() - t0),
        }
        log.append(log_item)

        print(
            f"Gen {gen:03d} | front={len(front0)} | "
            f"best_pos_obj={obj1_best:.6f} | "
            f"best_ampspur_obj={obj2_best:.6f} | "
            f"cache={len(cache.cache)}"
        )

    run_dir = cfg["output_dir"]
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(run_dir) / "progress.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    save_results(run_dir, pareto, encoder, model, lambda_vec, device, cfg)

    # 额外保存最优单解（按 obj1+obj2 最小排序）
    scored = []
    for ind in pareto:
        s = float(ind.fitness.values[0] + ind.fitness.values[1])
        scored.append((s, ind))
    scored.sort(key=lambda x: x[0])

    if len(scored) > 0:
        best_ind = scored[0][1]
        best_pattern = encoder.to_pattern(best_ind)
        A_batch, _ = predict_absorption_batch(model, best_pattern[None, None, :, :], lambda_vec, device)
        best_A = A_batch[0]
        fit, info = compute_objectives(lambda_vec, best_A, cfg)

        np.save(Path(run_dir) / "best_pattern.npy", best_pattern)
        np.save(Path(run_dir) / "best_spectrum.npy", best_A)

        best_report = {
            "fitness": list(map(float, fit)),
            "main_pos": info["main_pos"],
            "main_amp": info["main_amp"],
            "n_peaks": info["n_peaks"],
            "spur_penalty": info["spur_penalty"],
        }
        with open(Path(run_dir) / "best_report.json", "w", encoding="utf-8") as f:
            json.dump(best_report, f, indent=2, ensure_ascii=False)

        print("\n[Best solution]")
        print(json.dumps(best_report, indent=2, ensure_ascii=False))

    return run_dir, lambda_vec


# ==========================================================
# 9. 画图
# ==========================================================
def plot_best(run_dir, cfg, lambda_vec):
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    best_pattern = np.load(run_dir / "best_pattern.npy")
    best_A = np.load(run_dir / "best_spectrum.npy")

    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(lambda_vec, best_A, label="Pred A")
    plt.axvline(float(cfg["target_peak"]["pos"]), linestyle="--", label="Target pos")
    plt.axhline(float(cfg["target_peak"]["amp"]), linestyle="--", label="Target amp")
    plt.xlabel("lambda")
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

# ==========================================================
# 10. main
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "device" not in cfg or cfg["device"] is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print("[Config]")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    run_dir, lambda_vec = run_nsga2(cfg)
    plot_best(run_dir, cfg, lambda_vec)
    print(f"\nFinished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()