# ==========================================================
# Peak-aware FNO training for 11x11 -> complex S-parameters / A(lambda)
# 目标：
# - 保留 FNO 创新性
# - 不强求全谱逐点极高精度
# - 优先保证峰数量、峰位置、峰高
# - 数据提取更稳健（MATLAB v7.3 / 非v7.3，复数兼容，维度自动识别）
# - 使用离散差分约束，避免 d2 数值主导训练
# ==========================================================

import os
import time
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

print("TensorBoard：终端运行 tensorboard --logdir runs  打开 http://localhost:6006/")

# ==========================================================
# 0) 配置
# ==========================================================
patterns_path = r"C:\Users\90740\Desktop\final\training_patterns_11x11.mat"
sparams_path  = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"

SAVE_PATH_FINAL = r"C:\Users\90740\Desktop\final\fno_peak_curve_final.pt"
SAVE_PATH_BEST  = r"C:\Users\90740\Desktop\final\fno_peak_curve_best.pt"

SEED = 42

EPOCHS = 300
BATCH_SIZE = 64
VAL_BATCH_SIZE = 16

# 训练时每个结构采样 K 个 λ 点：一半均匀覆盖，一半峰区优先
K_TOTAL = 64
K_UNIFORM = 16
K_PEAK = 48

# 若想只用前 N_USE 个有效样本，填整数；用全部写 None
N_USE = None

TRAIN_RATIO = 0.8
EPS_VALID = 1e-12

# 可视化间隔
PLOT_EVERY = 10

# FNO 结构
MODES = 6
WIDTH = 64
DEPTH = 4
LAM_FF = 8
HEAD_HIDDEN = 256

# 优化
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 40
MIN_EPOCHS = 60

# loss 权重（已调稳）
HUBER_BETA = 0.02
LAMBDA_A = 1.2
LAMBDA_D1 = 0.02
LAMBDA_D2 = 0.005
LAMBDA_PHYS = 0.05

ALPHA_PEAK_WEIGHT = 4.0
P_PEAK_WEIGHT = 2.0
BETA_CURV_WEIGHT = 1.0

# 整谱峰检测参数（用于验证）
PEAK_MIN_HEIGHT = 0.08
PEAK_MIN_PROM_RATIO = 0.08
PEAK_MIN_DISTANCE = 2

LOG_ROOT = "runs/fno_peak_curve"

# ==========================================================
# 1) 随机种子
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==========================================================
# 2) .mat 读取与稳健提取
# ==========================================================
def load_mat_auto(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    if os.path.getsize(path) < 1024:
        raise OSError(f"文件过小/可能损坏：{path}")

    import h5py
    if h5py.is_hdf5(path):
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = f[k][:]
        return out
    else:
        from scipy.io import loadmat
        out = loadmat(path)
        return {k: v for k, v in out.items() if not k.startswith("__")}

def find_key_exact_or_contains(d, candidates):
    lower_to_key = {str(k).lower(): k for k in d.keys()}

    for c in candidates:
        if c.lower() in lower_to_key:
            return lower_to_key[c.lower()]

    for k in d.keys():
        lk = str(k).lower()
        for c in candidates:
            if c.lower() in lk:
                return k
    return None

def load_complex_struct(arr):
    arr = np.array(arr)

    # 已经是复数
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)

    # MATLAB struct complex: fields real / imag
    if hasattr(arr, "dtype") and arr.dtype.fields is not None:
        fields = set(arr.dtype.fields.keys())
        if "real" in fields and "imag" in fields:
            return (arr["real"] + 1j * arr["imag"]).astype(np.complex64)

    # 最后一维为 [real, imag]
    if arr.ndim >= 1 and arr.shape[-1] == 2 and arr.dtype.kind in ("f", "i", "u"):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)

    # 退化成实数
    return arr.astype(np.float32).astype(np.complex64)

def extract_selected_11x11xN(patterns_dict):
    key = find_key_exact_or_contains(patterns_dict, ["selected", "pattern", "patterns"])
    if key is None:
        raise KeyError(f"未找到 selected / pattern / patterns 键，现有键：{list(patterns_dict.keys())}")

    arr = np.array(patterns_dict[key])
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"selected 维度应为3维，当前 shape={arr.shape}")

    # 统一成 (11,11,N)
    if arr.shape[0] == 11 and arr.shape[1] == 11:
        out = arr
    elif arr.shape[1] == 11 and arr.shape[2] == 11:
        out = np.transpose(arr, (1, 2, 0))
    elif arr.shape[0] == 11 and arr.shape[2] == 11:
        out = np.transpose(arr, (0, 2, 1))
    else:
        dims_11 = [i for i, s in enumerate(arr.shape) if s == 11]
        if len(dims_11) != 2:
            raise ValueError(f"无法自动识别 11x11 结构维度，shape={arr.shape}")
        other = [i for i in range(3) if i not in dims_11][0]
        out = np.transpose(arr, (dims_11[0], dims_11[1], other))

    out = (out != 0).astype(np.float32)
    return out  # (11,11,N)

def extract_lambda_vec(sp_dict):
    key = find_key_exact_or_contains(
        sp_dict,
        ["lambda_vec", "lambda", "lambdas", "wavelength", "wl"]
    )
    if key is None:
        raise KeyError(f"未找到 lambda_vec / lambda / wavelength 键，现有键：{list(sp_dict.keys())}")

    lam = np.array(sp_dict[key]).squeeze().astype(np.float32)
    if lam.ndim != 1:
        lam = lam.reshape(-1)
    return lam

def extract_complex_sparam(sp_dict, base_name):
    key = find_key_exact_or_contains(
        sp_dict,
        [base_name, f"{base_name}_all", base_name.lower(), f"{base_name.lower()}_all"]
    )
    if key is not None:
        return load_complex_struct(np.array(sp_dict[key]))

    real_key = None
    imag_key = None
    for k in sp_dict.keys():
        lk = str(k).lower()
        if base_name.lower() in lk and "real" in lk:
            real_key = k
        if base_name.lower() in lk and "imag" in lk:
            imag_key = k

    if real_key is not None and imag_key is not None:
        real = np.array(sp_dict[real_key]).astype(np.float32)
        imag = np.array(sp_dict[imag_key]).astype(np.float32)
        return (real + 1j * imag).astype(np.complex64)

    raise KeyError(f"未找到 {base_name} 复数数据，现有键：{list(sp_dict.keys())}")

def standardize_sparam_shape(arr_complex, M, name="S"):
    arr = np.array(arr_complex)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"{name} 应为二维数组，当前 shape={arr.shape}")

    if arr.shape[1] == M:
        out = arr
    elif arr.shape[0] == M:
        out = arr.T
    else:
        raise ValueError(f"{name} 无法与 lambda_vec 长度 M={M} 对齐，当前 shape={arr.shape}")

    return out.astype(np.complex64)

def normalize_lambda(lam, lam_min, lam_max):
    return 2.0 * (lam - lam_min) / (lam_max - lam_min + 1e-12) - 1.0

# ==========================================================
# 3) 训练集：混合采样（均匀 + 峰区）
# ==========================================================
class PeakAwareTrainDataset(Dataset):
    """
    输入：
      x:   (N,11,11)
      S11: (N,M)
      S21: (N,M)
      lam: (M,)
    输出：
      x:      (1,11,11)
      lam_n:  (K,1)  已排序
      yS:     (K,4)  [ReS11, ImS11, ReS21, ImS21]
      a_true: (K,1)
      lam_raw:(K,1)  仅保留接口
    """
    def __init__(self, x_N11x11, lambda_vec, S11, S21, k_uniform=24, k_peak=24):
        self.x = x_N11x11.astype(np.float32)     # (N,11,11)
        self.lam = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)  # (M,)
        self.S11 = S11.astype(np.complex64)      # (N,M)
        self.S21 = S21.astype(np.complex64)      # (N,M)

        self.N = self.x.shape[0]
        self.M = self.lam.shape[0]

        self.k_uniform = int(k_uniform)
        self.k_peak = int(k_peak)
        self.K = self.k_uniform + self.k_peak

        self.lam_min = float(self.lam.min())
        self.lam_max = float(self.lam.max())

        R = np.abs(self.S11) ** 2
        T = np.abs(self.S21) ** 2
        self.A = (1.0 - R - T).astype(np.float32)  # (N,M)

        # 峰区采样得分：吸收峰 + 离散曲率
        curv = np.zeros_like(self.A, dtype=np.float32)
        if self.M >= 3:
            curv[:, 1:-1] = np.abs(self.A[:, 2:] - 2.0 * self.A[:, 1:-1] + self.A[:, :-2])

        curv_max = np.max(curv, axis=1, keepdims=True)
        curv = curv / (curv_max + 1e-8)

        self.sample_score = np.clip(self.A, 0.0, 1.0) ** 1.5 + 1.5 * curv + 1e-8

    def __len__(self):
        return self.N

    def _stratified_uniform_idx(self):
        edges = np.linspace(0, self.M, self.k_uniform + 1, dtype=int)
        idxs = []
        for l, r in zip(edges[:-1], edges[1:]):
            r = max(r, l + 1)
            idxs.append(np.random.randint(l, r))
        return np.array(idxs, dtype=np.int64)

    def _peak_idx(self, i, occupied):
        score = self.sample_score[i].copy()
        if len(occupied) > 0:
            score[np.array(list(occupied), dtype=np.int64)] = 0.0

        valid = np.where(score > 0)[0]
        if len(valid) == 0:
            remain = np.setdiff1d(np.arange(self.M), np.array(sorted(list(occupied))), assume_unique=False)
            take = min(self.k_peak, len(remain))
            if take <= 0:
                return np.array([], dtype=np.int64)
            return np.random.choice(remain, size=take, replace=False)

        prob = score / score.sum()
        take = min(self.k_peak, np.count_nonzero(score > 0))
        return np.random.choice(self.M, size=take, replace=False, p=prob)

    def __getitem__(self, i):
        x = self.x[i]  # (11,11)

        idx_u = self._stratified_uniform_idx()
        occupied = set(idx_u.tolist())
        idx_p = self._peak_idx(i, occupied)

        idx = np.unique(np.concatenate([idx_u, idx_p], axis=0))
        idx = np.sort(idx)

        if len(idx) < self.K:
            remain = np.setdiff1d(np.arange(self.M), idx, assume_unique=False)
            need = min(self.K - len(idx), len(remain))
            if need > 0:
                extra = np.random.choice(remain, size=need, replace=False)
                idx = np.sort(np.concatenate([idx, extra], axis=0))

        lam_raw = self.lam[idx].astype(np.float32)
        lam_n = normalize_lambda(lam_raw, self.lam_min, self.lam_max)

        s11 = self.S11[i, idx]
        s21 = self.S21[i, idx]

        yS = np.stack(
            [np.real(s11), np.imag(s11), np.real(s21), np.imag(s21)],
            axis=-1
        ).astype(np.float32)

        a_true = self.A[i, idx].astype(np.float32).reshape(-1, 1)

        x = torch.from_numpy(x).unsqueeze(0)            # (1,11,11)
        lam_n = torch.from_numpy(lam_n).unsqueeze(1)    # (K,1)
        yS = torch.from_numpy(yS)                       # (K,4)
        a_true = torch.from_numpy(a_true)               # (K,1)
        lam_raw = torch.from_numpy(lam_raw).unsqueeze(1)

        return x, lam_n, yS, a_true, lam_raw

# ==========================================================
# 4) 验证集：固定整谱，不随机抽点
# ==========================================================
class FullSpectrumValDataset(Dataset):
    def __init__(self, x_N11x11, lambda_vec, S11, S21):
        self.x = x_N11x11.astype(np.float32)
        self.lam = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)
        self.S11 = S11.astype(np.complex64)
        self.S21 = S21.astype(np.complex64)

        self.N = self.x.shape[0]
        self.M = self.lam.shape[0]

        self.lam_min = float(self.lam.min())
        self.lam_max = float(self.lam.max())
        self.lam_n_full = normalize_lambda(self.lam, self.lam_min, self.lam_max).astype(np.float32)

        R = np.abs(self.S11) ** 2
        T = np.abs(self.S21) ** 2
        self.A = (1.0 - R - T).astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).unsqueeze(0)
        lam_n = torch.from_numpy(self.lam_n_full).unsqueeze(1)
        lam_raw = torch.from_numpy(self.lam).unsqueeze(1)

        s11 = self.S11[i]
        s21 = self.S21[i]
        yS = np.stack(
            [np.real(s11), np.imag(s11), np.real(s21), np.imag(s21)],
            axis=-1
        ).astype(np.float32)
        a_true = self.A[i].astype(np.float32).reshape(-1, 1)

        yS = torch.from_numpy(yS)
        a_true = torch.from_numpy(a_true)

        return x, lam_n, yS, a_true, lam_raw

# ==========================================================
# 5) FNO 模型
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

def choose_gn_groups(width):
    for g in [8, 4, 2, 1]:
        if width % g == 0:
            return g
    return 1

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
# 6) 物理量、差分、损失、峰指标
# ==========================================================
def s_to_A(predS):
    ReS11, ImS11, ReS21, ImS21 = predS[..., 0], predS[..., 1], predS[..., 2], predS[..., 3]
    A = 1.0 - (ReS11 ** 2 + ImS11 ** 2 + ReS21 ** 2 + ImS21 ** 2)
    return A

def power_excess(predS):
    ReS11, ImS11, ReS21, ImS21 = predS[..., 0], predS[..., 1], predS[..., 2], predS[..., 3]
    return F.relu((ReS11 ** 2 + ImS11 ** 2 + ReS21 ** 2 + ImS21 ** 2) - 1.0)

def first_diff(y):
    return y[:, 1:] - y[:, :-1]

def second_diff(y):
    return y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]

def make_peak_weights(a_true):
    """
    a_true: (B, K)
    1) 峰高区域权重大
    2) 曲率大区域权重大
    使用离散二阶差分，不再除以波长步长，避免数值爆炸
    """
    w_peak = 1.0 + ALPHA_PEAK_WEIGHT * (a_true.clamp(0.0, 1.0) ** P_PEAK_WEIGHT)

    if a_true.shape[1] >= 3:
        curv = torch.abs(second_diff(a_true))  # (B, K-2)

        curv_full = torch.zeros_like(a_true)
        curv_max = curv.max(dim=1, keepdim=True).values
        curv_norm = curv / (curv_max + 1e-8)
        curv_full[:, 1:-1] = curv_norm

        w = w_peak + BETA_CURV_WEIGHT * curv_full
    else:
        w = w_peak

    return w

def manual_huber_loss(pred, target, delta=0.02):
    diff = pred - target
    abs_diff = torch.abs(diff)
    quad = torch.clamp(abs_diff, max=delta)
    lin = abs_diff - quad
    loss = 0.5 * quad ** 2 + delta * lin
    return loss.mean()

def peak_aware_loss(predS, yS, a_true, lam_raw):
    """
    predS:  (B, K, 4)
    yS:     (B, K, 4)
    a_true: (B, K, 1)
    lam_raw: 保留接口，不再用于导数缩放
    """
    A_true = a_true.squeeze(-1)   # (B, K)
    A_pred = s_to_A(predS)        # (B, K)

    # 1) 复数 S 参数主损失
    loss_s = manual_huber_loss(predS, yS, delta=HUBER_BETA)

    # 2) 峰优先 A 损失
    w = make_peak_weights(A_true)
    loss_a = (w * (A_pred - A_true) ** 2).mean()

    # 3) 离散一阶差分损失
    if A_true.shape[1] >= 2:
        d1_true = first_diff(A_true)
        d1_pred = first_diff(A_pred)
        loss_d1 = F.mse_loss(d1_pred, d1_true)
    else:
        loss_d1 = torch.tensor(0.0, device=predS.device, dtype=predS.dtype)

    # 4) 离散二阶差分损失
    if A_true.shape[1] >= 3:
        d2_true = second_diff(A_true)
        d2_pred = second_diff(A_pred)
        loss_d2 = F.mse_loss(d2_pred, d2_true)
    else:
        loss_d2 = torch.tensor(0.0, device=predS.device, dtype=predS.dtype)

    # 5) 被动性约束：|S11|^2 + |S21|^2 <= 1
    loss_phys = (power_excess(predS) ** 2).mean()

    total = (
        loss_s
        + LAMBDA_A * loss_a
        + LAMBDA_D1 * loss_d1
        + LAMBDA_D2 * loss_d2
        + LAMBDA_PHYS * loss_phys
    )

    stats = {
        "loss_s": float(loss_s.item()),
        "loss_a": float(loss_a.item()),
        "loss_d1": float(loss_d1.item()),
        "loss_d2": float(loss_d2.item()),
        "loss_phys": float(loss_phys.item()),
        "loss_total": float(total.item()),
    }
    return total, stats

def find_peaks_np(y):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_max = float(np.max(y)) if len(y) > 0 else 0.0

    if y_max <= 1e-8:
        return np.array([], dtype=np.int64)

    height = max(PEAK_MIN_HEIGHT, 0.10 * y_max)
    prominence = max(0.02, PEAK_MIN_PROM_RATIO * y_max)

    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            y,
            height=height,
            prominence=prominence,
            distance=PEAK_MIN_DISTANCE
        )
        return peaks.astype(np.int64)
    except Exception:
        idx = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= height:
                idx.append(i)
        return np.array(idx, dtype=np.int64)

def match_peaks_greedy(lam, A_true, A_pred):
    idx_t = find_peaks_np(A_true)
    idx_p = find_peaks_np(A_pred)

    n_t = len(idx_t)
    n_p = len(idx_p)
    count_err = abs(n_t - n_p)

    lam_range = float(lam[-1] - lam[0]) if len(lam) >= 2 else 1.0

    if n_t == 0 and n_p == 0:
        return 0.0, 0.0, 0.0

    if n_t == 0 or n_p == 0:
        loc_err = lam_range
        if n_t > 0:
            height_err = float(np.mean(np.abs(A_true[idx_t])))
        else:
            height_err = float(np.mean(np.abs(A_pred[idx_p])))
        return float(count_err), loc_err, height_err

    order = np.argsort(-A_true[idx_t])
    used_pred = set()
    loc_errs = []
    height_errs = []

    for oi in order:
        it = idx_t[oi]
        pred_candidates = [ip for ip in idx_p if ip not in used_pred]
        if len(pred_candidates) == 0:
            break

        best_ip = min(pred_candidates, key=lambda j: abs(lam[j] - lam[it]))
        used_pred.add(best_ip)

        loc_errs.append(abs(float(lam[best_ip] - lam[it])))
        height_errs.append(abs(float(A_pred[best_ip] - A_true[it])))

    if len(loc_errs) == 0:
        loc_err = lam_range
        height_err = 1.0
    else:
        loc_err = float(np.mean(loc_errs))
        height_err = float(np.mean(height_errs))

    return float(count_err), loc_err, height_err

# ==========================================================
# 7) 画图
# ==========================================================
def make_A_figure(lambda_vec, true_A, pred_A, title="A(lambda)"):
    fig = plt.figure(figsize=(6.8, 4.4))
    plt.plot(lambda_vec, true_A, label="true A")
    plt.plot(lambda_vec, pred_A, "--", label="pred A")
    plt.xlabel("lambda")
    plt.ylabel("A")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig

# ==========================================================
# 8) 读数据 + 对齐 + 过滤
# ==========================================================
patterns = load_mat_auto(patterns_path)
sp = load_mat_auto(sparams_path)

selected_11x11xN = extract_selected_11x11xN(patterns)   # (11,11,N_total)
lambda_vec = extract_lambda_vec(sp)                     # (M,)

S11 = extract_complex_sparam(sp, "S11")
S21 = extract_complex_sparam(sp, "S21")

M = len(lambda_vec)
S11 = standardize_sparam_shape(S11, M, name="S11")
S21 = standardize_sparam_shape(S21, M, name="S21")

N_total = min(selected_11x11xN.shape[2], S11.shape[0], S21.shape[0])
selected_11x11xN = selected_11x11xN[:, :, :N_total]
S11 = S11[:N_total, :]
S21 = S21[:N_total, :]

# 变成 (N,11,11)
x_all = np.transpose(selected_11x11xN, (2, 0, 1)).astype(np.float32)

finite_mask = (
    np.isfinite(np.real(S11)).all(axis=1) &
    np.isfinite(np.imag(S11)).all(axis=1) &
    np.isfinite(np.real(S21)).all(axis=1) &
    np.isfinite(np.imag(S21)).all(axis=1)
)

valid_nonzero = (
    np.any(np.abs(S11) > EPS_VALID, axis=1) |
    np.any(np.abs(S21) > EPS_VALID, axis=1)
)

valid = finite_mask & valid_nonzero
idx_valid = np.where(valid)[0]

print("总样本数 =", N_total, "| 有效样本数 =", len(idx_valid))

if N_USE is not None:
    idx_valid = idx_valid[:N_USE]
    print(f"仅使用前 {len(idx_valid)} 个有效样本")

x_all = x_all[idx_valid]
S11 = S11[idx_valid]
S21 = S21[idx_valid]

N = x_all.shape[0]
print("最终训练样本 N =", N, "| 谱点 M =", M)

# ==========================================================
# 9) 划分 train / val
# ==========================================================
np.random.seed(SEED)
perm = np.random.permutation(N)

n_train = int(TRAIN_RATIO * N)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

x_train = x_all[train_idx]
x_val   = x_all[val_idx]

S11_train = S11[train_idx]
S21_train = S21[train_idx]
S11_val   = S11[val_idx]
S21_val   = S21[val_idx]

if len(x_val) == 0:
    raise RuntimeError("验证集为空，请检查 TRAIN_RATIO / N_USE")

train_ds = PeakAwareTrainDataset(
    x_train, lambda_vec, S11_train, S21_train,
    k_uniform=K_UNIFORM, k_peak=K_PEAK
)

val_ds = FullSpectrumValDataset(
    x_val, lambda_vec, S11_val, S21_val
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================================
# 10) 模型 / 优化器 / 调度器
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

model = FNO_LambdaConditional_SParams(
    modes=MODES, width=WIDTH, depth=DEPTH, lam_ff=LAM_FF, head_hidden=HEAD_HIDDEN
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=12,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=0,
    min_lr=1e-6,
    eps=1e-8
)

# ==========================================================
# 11) 整谱验证
# ==========================================================
@torch.no_grad()
def evaluate_full_spectrum(loader):
    model.eval()

    total_loss = 0.0
    total_count = 0

    peak_count_err_sum = 0.0
    peak_loc_err_sum = 0.0
    peak_height_err_sum = 0.0
    n_peak_metric = 0

    vis_pack = None

    for batch_id, (x, lam_n, yS, a_true, lam_raw) in enumerate(loader):
        x = x.to(device)
        lam_n = lam_n.to(device)
        yS = yS.to(device)
        a_true = a_true.to(device)
        lam_raw = lam_raw.to(device)

        predS = model.forward_curve(x, lam_n)
        loss, _ = peak_aware_loss(predS, yS, a_true, lam_raw)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

        A_pred = s_to_A(predS).detach().cpu().numpy()
        A_true_np = a_true.squeeze(-1).detach().cpu().numpy()
        lam_np = lam_raw.squeeze(-1).detach().cpu().numpy()

        for i in range(bs):
            c_err, l_err, h_err = match_peaks_greedy(lam_np[i], A_true_np[i], A_pred[i])
            peak_count_err_sum += c_err
            peak_loc_err_sum += l_err
            peak_height_err_sum += h_err
            n_peak_metric += 1

        if vis_pack is None:
            vis_pack = (lam_np[0], A_true_np[0], A_pred[0])

    mean_loss = total_loss / max(total_count, 1)
    mean_count_err = peak_count_err_sum / max(n_peak_metric, 1)
    mean_loc_err = peak_loc_err_sum / max(n_peak_metric, 1)
    mean_height_err = peak_height_err_sum / max(n_peak_metric, 1)

    lam_range = float(lambda_vec[-1] - lambda_vec[0]) if len(lambda_vec) >= 2 else 1.0
    loc_norm = mean_loc_err / (lam_range + 1e-8)

    # 峰优先评分，越小越好
    peak_score = 0.45 * loc_norm + 0.35 * mean_height_err + 0.20 * mean_count_err

    metrics = {
        "val_total": mean_loss,
        "peak_count_err": mean_count_err,
        "peak_loc_err": mean_loc_err,
        "peak_height_err": mean_height_err,
        "peak_score": peak_score,
        "vis_pack": vis_pack,
    }
    return metrics

# ==========================================================
# 12) 训练
# ==========================================================
run_name = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"{LOG_ROOT}/{run_name}")

best_score = float("inf")
best_epoch = -1
best_state = None
bad_epochs = 0

train_hist = []
val_hist = []
score_hist = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    train_total_sum = 0.0
    train_count = 0

    train_s_sum = 0.0
    train_a_sum = 0.0
    train_d1_sum = 0.0
    train_d2_sum = 0.0
    train_phys_sum = 0.0

    for x, lam_n, yS, a_true, lam_raw in train_loader:
        x = x.to(device)
        lam_n = lam_n.to(device)
        yS = yS.to(device)
        a_true = a_true.to(device)
        lam_raw = lam_raw.to(device)

        optimizer.zero_grad()

        # 每个结构只编码一次，再对 K 个 λ 解码
        predS = model.forward_curve(x, lam_n)

        loss, stats = peak_aware_loss(predS, yS, a_true, lam_raw)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        train_total_sum += float(loss.item()) * bs
        train_count += bs

        train_s_sum += stats["loss_s"] * bs
        train_a_sum += stats["loss_a"] * bs
        train_d1_sum += stats["loss_d1"] * bs
        train_d2_sum += stats["loss_d2"] * bs
        train_phys_sum += stats["loss_phys"] * bs

    train_total = train_total_sum / max(train_count, 1)
    train_s = train_s_sum / max(train_count, 1)
    train_a = train_a_sum / max(train_count, 1)
    train_d1 = train_d1_sum / max(train_count, 1)
    train_d2 = train_d2_sum / max(train_count, 1)
    train_phys = train_phys_sum / max(train_count, 1)

    val_metrics = evaluate_full_spectrum(val_loader)
    val_total = val_metrics["val_total"]
    peak_score = val_metrics["peak_score"]

    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(peak_score)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr < old_lr:
        print(f"[Scheduler] LR reduced: {old_lr:.3e} -> {new_lr:.3e}")

    train_hist.append(train_total)
    val_hist.append(val_total)
    score_hist.append(peak_score)

    writer.add_scalar("loss/train_total", train_total, epoch)
    writer.add_scalar("loss/train_s", train_s, epoch)
    writer.add_scalar("loss/train_A", train_a, epoch)
    writer.add_scalar("loss/train_d1", train_d1, epoch)
    writer.add_scalar("loss/train_d2", train_d2, epoch)
    writer.add_scalar("loss/train_phys", train_phys, epoch)

    writer.add_scalar("loss/val_total", val_total, epoch)
    writer.add_scalar("metric/peak_score", peak_score, epoch)
    writer.add_scalar("metric/peak_count_err", val_metrics["peak_count_err"], epoch)
    writer.add_scalar("metric/peak_loc_err", val_metrics["peak_loc_err"], epoch)
    writer.add_scalar("metric/peak_height_err", val_metrics["peak_height_err"], epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    if (epoch % PLOT_EVERY == 0) or (epoch == 1):
        lam_vis, A_true_vis, A_pred_vis = val_metrics["vis_pack"]
        fig = make_A_figure(lam_vis, A_true_vis, A_pred_vis, title=f"Epoch {epoch} | A(lambda)")
        writer.add_figure("viz/val_A", fig, epoch)
        plt.close(fig)

    if peak_score < best_score:
        best_score = peak_score
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0

        torch.save({
            "state_dict": best_state,
            "config": {
                "MODES": MODES,
                "WIDTH": WIDTH,
                "DEPTH": DEPTH,
                "LAM_FF": LAM_FF,
                "HEAD_HIDDEN": HEAD_HIDDEN,
                "K_TOTAL": K_TOTAL,
                "K_UNIFORM": K_UNIFORM,
                "K_PEAK": K_PEAK,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "LAMBDA_A": LAMBDA_A,
                "LAMBDA_D1": LAMBDA_D1,
                "LAMBDA_D2": LAMBDA_D2,
                "LAMBDA_PHYS": LAMBDA_PHYS,
                "ALPHA_PEAK_WEIGHT": ALPHA_PEAK_WEIGHT,
                "P_PEAK_WEIGHT": P_PEAK_WEIGHT,
                "BETA_CURV_WEIGHT": BETA_CURV_WEIGHT,
                "HUBER_BETA": HUBER_BETA,
                "SEED": SEED,
            },
            "best_epoch": best_epoch,
            "best_peak_score": best_score,
            "lambda_vec": lambda_vec
        }, SAVE_PATH_BEST)

    else:
        bad_epochs += 1

    print(
        f"Epoch {epoch:03d} | "
        f"train={train_total:.6e} | val={val_total:.6e} | "
        f"S={train_s:.4e} | A={train_a:.4e} | d1={train_d1:.4e} | d2={train_d2:.4e} | phys={train_phys:.4e} | "
        f"peak_score={peak_score:.6e} | "
        f"count_err={val_metrics['peak_count_err']:.4f} | "
        f"loc_err={val_metrics['peak_loc_err']:.4f} | "
        f"height_err={val_metrics['peak_height_err']:.4f} | "
        f"best_epoch={best_epoch}"
    )

    if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
        break

# ==========================================================
# 13) 保存 final / best
# ==========================================================
torch.save({
    "state_dict": model.state_dict(),
    "config": {
        "MODES": MODES,
        "WIDTH": WIDTH,
        "DEPTH": DEPTH,
        "LAM_FF": LAM_FF,
        "HEAD_HIDDEN": HEAD_HIDDEN,
        "K_TOTAL": K_TOTAL,
        "K_UNIFORM": K_UNIFORM,
        "K_PEAK": K_PEAK,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LAMBDA_A": LAMBDA_A,
        "LAMBDA_D1": LAMBDA_D1,
        "LAMBDA_D2": LAMBDA_D2,
        "LAMBDA_PHYS": LAMBDA_PHYS,
        "ALPHA_PEAK_WEIGHT": ALPHA_PEAK_WEIGHT,
        "P_PEAK_WEIGHT": P_PEAK_WEIGHT,
        "BETA_CURV_WEIGHT": BETA_CURV_WEIGHT,
        "HUBER_BETA": HUBER_BETA,
        "SEED": SEED,
    },
    "best_epoch": best_epoch,
    "best_peak_score": best_score,
    "lambda_vec": lambda_vec
}, SAVE_PATH_FINAL)

writer.close()

if best_state is not None:
    model.load_state_dict(best_state)

plt.figure(figsize=(7, 4))
plt.plot(train_hist, label="train_total")
plt.plot(val_hist, label="val_total")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_peak_training.png", dpi=200)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(score_hist, label="peak_score")
plt.xlabel("epoch")
plt.ylabel("score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("peak_score_curve.png", dpi=200)
plt.show()

print("训练完成：")
print(f"  best模型:  {SAVE_PATH_BEST}")
print(f"  final模型: {SAVE_PATH_FINAL}")
print(f"  best_epoch = {best_epoch}, best_peak_score = {best_score:.6e}")