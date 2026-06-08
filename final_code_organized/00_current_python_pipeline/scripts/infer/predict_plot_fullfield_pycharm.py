import os
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_utils import resolve_checkpoint_choice
from src.material_dispersion import au_eps_from_lambda_m, sio2_eps_from_lambda_m
from src.project_paths import (
    BEST_MODEL_HISTORY_ROOT,
    FIELD_DATA_DIR,
    MODELS_CURRENT_DIR,
    OPTIONAL_PATTERNS_PATH,
    PREDICTION_OUTPUTS_DIR,
    SAMPLING_META_PATH,
)

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


# ==========================================================
# 0) 配置
# ==========================================================
CURRENT_BEST_PATH = MODELS_CURRENT_DIR / "fno_fullfield_maxwell_dual_best.pt"
CURRENT_FINAL_PATH = MODELS_CURRENT_DIR / "fno_fullfield_maxwell_dual_final.pt"
BEST_HISTORY_ROOT = BEST_MODEL_HISTORY_ROOT
DATA_DIR = FIELD_DATA_DIR
META_PATH = SAMPLING_META_PATH
PATTERNS_PATH = OPTIONAL_PATTERNS_PATH

# checkpoint 选择:
#   "default" 使用项目根目录下最新 best
#   "history" 使用 best_model_history 中的某个历史 best
#   "path"    使用 CHECKPOINT_CUSTOM_PATH
CHECKPOINT_MODE = "default"
CHECKPOINT_CUSTOM_PATH = None
CHECKPOINT_RUN_NAME = None    # 例如 "20260403-101530"，None 表示最新一轮训练
CHECKPOINT_BEST_INDEX = None  # 1-based；None 表示该轮里的最后一个 best

# 推荐直接改这 4 个量来切换模型：
MODEL_CHOICE = "current_best"
MODEL_CUSTOM_PATH = None
MODEL_RUN_NAME = None
MODEL_BEST_INDEX = None

SAMPLE_ID = 201
LAMBDA_INDEX = 45
LAMBDA_TARGET = None  # 单位 m；若不为 None，则优先按目标波长选最近点

FIELD_COMPONENT = "Ez"   # Ex / Ey / Ez / Hx / Hy / Hz
FIELD_VIEW = "magnitude"  # magnitude / real / imag

Z_INDEX = None          # None 表示自动选中间层
Y_INDEX = None          # None 表示自动选中间层

SAVE_DIR = PREDICTION_OUTPUTS_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

C0 = 299792458.0


# ==========================================================
# 1) .mat 读取
# ==========================================================
def load_mat_auto(path):
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")

    is_hdf5 = False
    if h5py is not None:
        try:
            is_hdf5 = h5py.is_hdf5(path)
        except Exception:
            is_hdf5 = False
    else:
        with open(path, "rb") as f:
            is_hdf5 = f.read(8) == b"\x89HDF\r\n\x1a\n"

    if is_hdf5:
        if h5py is None:
            raise RuntimeError(f"{path} 是 v7.3/HDF5 文件，请安装 h5py")
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = decode_h5_object(f[k])
        return out

    if loadmat is None:
        raise RuntimeError("需要 scipy.io.loadmat 读取非 v7.3 mat 文件")
    out = loadmat(path)
    return {k: v for k, v in out.items() if not k.startswith("__")}


def decode_h5_object(obj):
    if isinstance(obj, h5py.Dataset):
        return decode_complex_array(obj[()])
    if isinstance(obj, h5py.Group):
        keys = set(obj.keys())
        if {"real", "imag"}.issubset(keys):
            return np.asarray(obj["real"][()]) + 1j * np.asarray(obj["imag"][()])
        return {k: decode_h5_object(v) for k, v in obj.items()}
    return obj


def decode_complex_array(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if hasattr(arr, "dtype") and arr.dtype.fields is not None:
        fields = set(arr.dtype.fields.keys())
        if "real" in fields and "imag" in fields:
            return arr["real"] + 1j * arr["imag"]
    return arr


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


def extract_selected_11x11xN(patterns_dict):
    key = find_key_exact_or_contains(patterns_dict, ["selected", "pattern", "patterns"])
    if key is None:
        raise KeyError(f"未找到 selected/pattern/patterns 键，现有键：{list(patterns_dict.keys())}")
    arr = np.array(patterns_dict[key]).squeeze()
    if arr.ndim != 3:
        raise ValueError(f"selected 应为 3 维，当前 shape={arr.shape}")
    if arr.shape[0] == 11 and arr.shape[1] == 11:
        out = arr
    elif arr.shape[1] == 11 and arr.shape[2] == 11:
        out = np.transpose(arr, (1, 2, 0))
    elif arr.shape[0] == 11 and arr.shape[2] == 11:
        out = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(f"无法识别 11x11 维度，shape={arr.shape}")
    return (out != 0).astype(np.float32)


def standardize_coord_1d(arr):
    arr = np.array(arr).squeeze().astype(np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def standardize_pattern_11x11(arr):
    arr = np.array(arr).squeeze()
    if arr.shape != (11, 11):
        raise ValueError(f"binary_matrix shape 异常：{arr.shape}")
    return (arr != 0).astype(np.float32)


def normalize_interval(v):
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return 2.0 * (v - vmin) / (vmax - vmin + 1e-12) - 1.0


def nearest_resize_2d(pattern_11, nx, ny):
    x = torch.from_numpy(pattern_11.astype(np.float32)[None, None, ...])
    y = F.interpolate(x, size=(nx, ny), mode="nearest")
    return y[0, 0].numpy()


def build_eps_volume(pattern_11, xv, yv, zv, metal_eps, dielectric_eps, air_eps, bottom_zmax, dielectric_zmax, top_zmax):
    nx, ny, nz = len(xv), len(yv), len(zv)
    pattern_xy = nearest_resize_2d(pattern_11, nx, ny)
    eps = np.full((nx, ny, nz), air_eps, dtype=np.complex64)

    bottom_mask = zv <= bottom_zmax
    dielectric_mask = (zv > bottom_zmax) & (zv <= dielectric_zmax)
    top_mask = (zv > dielectric_zmax) & (zv <= top_zmax)

    if np.any(bottom_mask):
        eps[:, :, bottom_mask] = metal_eps
    if np.any(dielectric_mask):
        eps[:, :, dielectric_mask] = dielectric_eps
    if np.any(top_mask):
        eps[:, :, top_mask] = dielectric_eps
        metal_xy = pattern_xy > 0.5
        for zi in np.where(top_mask)[0]:
            eps[:, :, zi][metal_xy] = metal_eps

    metal_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    metal_mask[:, :, bottom_mask] = 1.0
    if np.any(top_mask):
        for zi in np.where(top_mask)[0]:
            metal_mask[:, :, zi][pattern_xy > 0.5] = 1.0

    return eps, metal_mask


def make_coord_maps(xv, yv, zv):
    x_norm = normalize_interval(xv)
    y_norm = normalize_interval(yv)
    z_norm = normalize_interval(zv)

    x_map = np.repeat(x_norm[:, None, None], len(yv), axis=1).repeat(len(zv), axis=2)
    y_map = np.repeat(y_norm[None, :, None], len(xv), axis=0).repeat(len(zv), axis=2)
    z_map = np.repeat(z_norm[None, None, :], len(xv), axis=0).repeat(len(yv), axis=1)
    return x_map.astype(np.float32), y_map.astype(np.float32), z_map.astype(np.float32)


def permute_to_xyz(arr, target_shape):
    import itertools
    arr = np.asarray(arr)
    if arr.shape == target_shape:
        return arr
    for perm in itertools.permutations(range(arr.ndim)):
        if tuple(arr.shape[p] for p in perm) == tuple(target_shape):
            return np.transpose(arr, perm)
    raise ValueError(f"无法把 shape={arr.shape} 变成 {target_shape}")


def read_field_slice(sample_path, field_key, lam_idx, target_shape):
    if h5py is not None and h5py.is_hdf5(str(sample_path)):
        with h5py.File(sample_path, "r") as f:
            ds = f[field_key]
            shape = ds.shape
            n_lambda = int(np.array(f["lambda"]).size)
            lambda_axes = [i for i, s in enumerate(shape) if s == n_lambda]
            chosen = lambda_axes[0]
            slc = [slice(None)] * len(shape)
            slc[chosen] = int(lam_idx)
            arr = decode_complex_array(ds[tuple(slc)])
    else:
        data = load_mat_auto(sample_path)
        arr = np.asarray(decode_complex_array(data[field_key]))
        n_lambda = len(np.array(data["lambda"]).reshape(-1))
        lambda_axes = [i for i, s in enumerate(arr.shape) if s == n_lambda]
        chosen = lambda_axes[0]
        arr = np.take(arr, lam_idx, axis=chosen)

    arr = np.asarray(arr).squeeze()
    arr = permute_to_xyz(arr, target_shape)
    return arr.astype(np.complex64)


def read_sample_sparams(sample_path):
    data = load_mat_auto(sample_path)
    lam = np.array(data["lambda"]).reshape(-1).astype(np.float32)
    s11 = np.array(decode_complex_array(data["S11_ref"])).reshape(-1).astype(np.complex64)
    s21 = np.array(decode_complex_array(data["S21_ref"])).reshape(-1).astype(np.complex64)
    r = np.array(data["R_ref"]).reshape(-1).astype(np.float32)
    t = np.array(data["T_ref"]).reshape(-1).astype(np.float32)
    a = np.array(data["A_ref"]).reshape(-1).astype(np.float32)
    return lam, s11, s21, r, t, a


def resolve_checkpoint_path():
    if CHECKPOINT_MODE == "default":
        return CHECKPOINT_PATH, "default best"

    if CHECKPOINT_MODE == "path":
        if CHECKPOINT_CUSTOM_PATH is None:
            raise ValueError("CHECKPOINT_MODE='path' 时需要设置 CHECKPOINT_CUSTOM_PATH")
        path = Path(CHECKPOINT_CUSTOM_PATH)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"未找到 checkpoint: {path}")
        return path, "custom path"

    if CHECKPOINT_MODE == "history":
        if not BEST_HISTORY_ROOT.exists():
            raise FileNotFoundError(f"未找到历史模型目录: {BEST_HISTORY_ROOT}")

        run_dirs = sorted([p for p in BEST_HISTORY_ROOT.iterdir() if p.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"历史模型目录为空: {BEST_HISTORY_ROOT}")

        if CHECKPOINT_RUN_NAME is None:
            run_dir = run_dirs[-1]
        else:
            run_dir = BEST_HISTORY_ROOT / CHECKPOINT_RUN_NAME
            if not run_dir.exists():
                raise FileNotFoundError(f"未找到指定训练轮次目录: {run_dir}")

        candidates = sorted(run_dir.glob("best_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"目录中没有历史 best 模型: {run_dir}")

        if CHECKPOINT_BEST_INDEX is None:
            ckpt = candidates[-1]
        else:
            idx = int(CHECKPOINT_BEST_INDEX) - 1
            if idx < 0 or idx >= len(candidates):
                raise IndexError(f"CHECKPOINT_BEST_INDEX={CHECKPOINT_BEST_INDEX} 超出范围 1..{len(candidates)}")
            ckpt = candidates[idx]
        return ckpt, f"history run={run_dir.name}"

    raise ValueError(f"未知 CHECKPOINT_MODE: {CHECKPOINT_MODE}")


# ==========================================================
# 2) 模型
# ==========================================================
def resolve_checkpoint_path():
    return resolve_checkpoint_choice(
        MODEL_CHOICE,
        current_best=CURRENT_BEST_PATH,
        current_final=CURRENT_FINAL_PATH,
        history_root=BEST_HISTORY_ROOT,
        custom_path=MODEL_CUSTOM_PATH,
        run_name=MODEL_RUN_NAME,
        best_index=MODEL_BEST_INDEX,
        project_root=PROJECT_ROOT,
    )


class LambdaFourierFeatures(nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        freqs = (2.0 ** torch.arange(n_freq).float()) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam_norm):
        x = lam_norm * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat)
        )

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1), norm="ortho")
            out_ft = torch.zeros(
                x.shape[0], self.weight.shape[1], x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                dtype=torch.cfloat, device=x.device
            )
            mx = min(self.modes_x, x_ft.shape[-3])
            my = min(self.modes_y, x_ft.shape[-2])
            mz = min(self.modes_z, x_ft.shape[-1])
            out_ft[:, :, :mx, :my, :mz] = torch.einsum(
                "bixyz,ioxyz->boxyz", x_ft[:, :, :mx, :my, :mz], self.weight[:, :, :mx, :my, :mz]
            )
            out = torch.fft.irfftn(out_ft, s=x.shape[-3:], norm="ortho")
        return out.to(dtype=x.dtype)


class FNOBlock3d(nn.Module):
    def __init__(self, width, modes_x, modes_y, modes_z):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_x, modes_y, modes_z)
        self.pointwise = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class FNO3dConditionalField(nn.Module):
    def __init__(self, base_in=6, modes_x=10, modes_y=10, modes_z=10, width=32, depth=4, lam_ff=8, head_hidden=128):
        super().__init__()
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)
        total_in = base_in + 2 * lam_ff
        self.input_proj = nn.Sequential(nn.Conv3d(total_in, width, 1), nn.GELU())
        self.blocks = nn.ModuleList([FNOBlock3d(width, modes_x, modes_y, modes_z) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Conv3d(width, head_hidden, 1),
            nn.GELU(),
            nn.Conv3d(head_hidden, 12, 1),
        )
        self.s_head = nn.Sequential(
            nn.Linear(width + 2 * lam_ff, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 4),
        )

    def forward(self, x_static, lam_norm):
        b, _, nx, ny, nz = x_static.shape
        lam_embed = self.lam_embed(lam_norm)
        lam_feat = lam_embed.view(b, -1, 1, 1, 1).expand(b, -1, nx, ny, nz)
        x = torch.cat([x_static, lam_feat], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        field_out = self.head(x)
        latent = x.mean(dim=(-3, -2, -1))
        s_out = self.s_head(torch.cat([latent, lam_embed], dim=-1))
        s_out = torch.cat([s_out[:, :2], torch.zeros_like(s_out[:, 2:4])], dim=-1)
        return field_out, s_out


def split_pred_channels(pred):
    return {
        "Ex": pred[0] + 1j * pred[1],
        "Ey": pred[2] + 1j * pred[3],
        "Ez": pred[4] + 1j * pred[5],
        "Hx": pred[6] + 1j * pred[7],
        "Hy": pred[8] + 1j * pred[9],
        "Hz": pred[10] + 1j * pred[11],
    }


def project_to_passive(predS, eps=1e-12):
    power = predS[:, 0] ** 2 + predS[:, 1] ** 2 + predS[:, 2] ** 2 + predS[:, 3] ** 2
    scale = torch.where(power > 1.0, torch.rsqrt(power + eps), torch.ones_like(power))
    out = predS.clone()
    out[:, 0] = predS[:, 0] * scale
    out[:, 1] = predS[:, 1] * scale
    out[:, 2] = predS[:, 2] * scale
    out[:, 3] = predS[:, 3] * scale
    return out


def field_to_view(arr_complex, view_mode):
    if view_mode == "real":
        return np.real(arr_complex)
    if view_mode == "imag":
        return np.imag(arr_complex)
    return np.abs(arr_complex)


def plot_field_comparison(xv, yv, zv, true_field, pred_field, field_name, lam_val, save_path):
    z_idx = len(zv) // 2 if Z_INDEX is None else int(Z_INDEX)
    y_idx = len(yv) // 2 if Y_INDEX is None else int(Y_INDEX)

    true_xy = field_to_view(true_field[:, :, z_idx], FIELD_VIEW)
    pred_xy = field_to_view(pred_field[:, :, z_idx], FIELD_VIEW)
    err_xy = pred_xy - true_xy

    true_xz = field_to_view(true_field[:, y_idx, :], FIELD_VIEW)
    pred_xz = field_to_view(pred_field[:, y_idx, :], FIELD_VIEW)
    err_xz = pred_xz - true_xz

    vmax_xy = max(np.max(np.abs(true_xy)), np.max(np.abs(pred_xy)), 1e-12)
    vmax_xz = max(np.max(np.abs(true_xz)), np.max(np.abs(pred_xz)), 1e-12)
    emax_xy = max(np.max(np.abs(err_xy)), 1e-12)
    emax_xz = max(np.max(np.abs(err_xz)), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    extent_xy = [xv[0] * 1e6, xv[-1] * 1e6, yv[0] * 1e6, yv[-1] * 1e6]
    extent_xz = [xv[0] * 1e6, xv[-1] * 1e6, zv[0] * 1e9, zv[-1] * 1e9]

    im = axes[0, 0].imshow(true_xy.T, origin="lower", extent=extent_xy, aspect="auto", cmap="jet", vmin=-vmax_xy if FIELD_VIEW != "magnitude" else 0.0, vmax=vmax_xy)
    axes[0, 0].set_title(f"True {field_name} XY @ z={zv[z_idx]*1e9:.1f} nm")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(pred_xy.T, origin="lower", extent=extent_xy, aspect="auto", cmap="jet", vmin=-vmax_xy if FIELD_VIEW != "magnitude" else 0.0, vmax=vmax_xy)
    axes[0, 1].set_title(f"Pred {field_name} XY")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(err_xy.T, origin="lower", extent=extent_xy, aspect="auto", cmap="bwr", vmin=-emax_xy, vmax=emax_xy)
    axes[0, 2].set_title("Error XY")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(true_xz.T, origin="lower", extent=extent_xz, aspect="auto", cmap="jet", vmin=-vmax_xz if FIELD_VIEW != "magnitude" else 0.0, vmax=vmax_xz)
    axes[1, 0].set_title(f"True {field_name} XZ @ y={yv[y_idx]*1e6:.2f} um")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(pred_xz.T, origin="lower", extent=extent_xz, aspect="auto", cmap="jet", vmin=-vmax_xz if FIELD_VIEW != "magnitude" else 0.0, vmax=vmax_xz)
    axes[1, 1].set_title(f"Pred {field_name} XZ")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(err_xz.T, origin="lower", extent=extent_xz, aspect="auto", cmap="bwr", vmin=-emax_xz, vmax=emax_xz)
    axes[1, 2].set_title("Error XZ")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    for ax in axes[0, :]:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
    for ax in axes[1, :]:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (nm)")

    fig.suptitle(f"{field_name} | view={FIELD_VIEW} | lambda={lam_val*1e6:.3f} um", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_sparams(lam, s11_true, s21_true, r_true, t_true, a_true, s11_pred, s21_pred, selected_idx, save_path):
    r_pred = np.abs(s11_pred) ** 2
    t_pred = np.abs(s21_pred) ** 2
    a_pred = np.clip(1.0 - r_pred - t_pred, 0.0, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(lam * 1e6, np.real(s11_true), label="Re(S11) true")
    axes[0, 0].plot(lam * 1e6, np.imag(s11_true), label="Im(S11) true")
    axes[0, 0].plot(lam * 1e6, np.real(s11_pred), "--", label="Re(S11) pred")
    axes[0, 0].plot(lam * 1e6, np.imag(s11_pred), "--", label="Im(S11) pred")
    axes[0, 0].axvline(lam[selected_idx] * 1e6, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("S11")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(lam * 1e6, np.real(s21_true), label="Re(S21) true")
    axes[0, 1].plot(lam * 1e6, np.imag(s21_true), label="Im(S21) true")
    axes[0, 1].plot(lam * 1e6, np.real(s21_pred), "--", label="Re(S21) pred")
    axes[0, 1].plot(lam * 1e6, np.imag(s21_pred), "--", label="Im(S21) pred")
    axes[0, 1].axvline(lam[selected_idx] * 1e6, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("S21")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(lam * 1e6, r_true, label="R true")
    axes[1, 0].plot(lam * 1e6, t_true, label="T true")
    axes[1, 0].plot(lam * 1e6, a_true, label="A true")
    axes[1, 0].plot(lam * 1e6, r_pred, "--", label="R pred")
    axes[1, 0].plot(lam * 1e6, t_pred, "--", label="T pred")
    axes[1, 0].plot(lam * 1e6, a_pred, "--", label="A pred")
    axes[1, 0].axvline(lam[selected_idx] * 1e6, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("R / T / A")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(lam * 1e6, a_true, color="tab:red", label="A true")
    axes[1, 1].plot(lam * 1e6, a_pred, "--", color="tab:blue", label="A pred")
    axes[1, 1].scatter([lam[selected_idx] * 1e6], [a_true[selected_idx]], color="k", zorder=3)
    axes[1, 1].set_title("Absorption")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    for ax in axes.reshape(-1):
        ax.set_xlabel("lambda (um)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    resolved_checkpoint_path, checkpoint_source = resolve_checkpoint_path()
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})

    modes_x = int(cfg.get("MODES_X", 10))
    modes_y = int(cfg.get("MODES_Y", 10))
    modes_z = int(cfg.get("MODES_Z", 10))
    width = int(cfg.get("WIDTH", 32))
    depth = int(cfg.get("DEPTH", 4))
    lam_ff = int(cfg.get("LAM_FF", 8))
    head_hidden = int(cfg.get("HEAD_HIDDEN", 128))
    down_x = int(cfg.get("DOWN_X", 1))
    down_y = int(cfg.get("DOWN_Y", 1))
    down_z = int(cfg.get("DOWN_Z", 1))

    air_eps = complex(1.0, 0.0)
    bottom_zmax = 100e-9
    dielectric_zmax = 400e-9
    top_zmax = 430e-9

    meta = load_mat_auto(META_PATH)
    xv_full = standardize_coord_1d(meta["xv"])
    yv_full = standardize_coord_1d(meta["yv"])
    zv_full = standardize_coord_1d(meta["zv"])
    xv = xv_full[::down_x]
    yv = yv_full[::down_y]
    zv = zv_full[::down_z]
    x_map, y_map, z_map = make_coord_maps(xv, yv, zv)

    patterns_all = None
    if PATTERNS_PATH is not None:
        patterns_all = extract_selected_11x11xN(load_mat_auto(PATTERNS_PATH))

    sample_path = DATA_DIR / f"sample_{SAMPLE_ID:05d}.mat"
    if not sample_path.exists():
        raise FileNotFoundError(f"未找到样本文件：{sample_path}")

    sample_data = load_mat_auto(sample_path)
    if "binary_matrix" in sample_data:
        pattern_11 = standardize_pattern_11x11(sample_data["binary_matrix"])
    else:
        if patterns_all is None:
            raise KeyError("样本内没有 binary_matrix，且未提供 PATTERNS_PATH")
        pattern_11 = patterns_all[:, :, SAMPLE_ID - 1]

    lam, s11, s21, r, t, a = read_sample_sparams(sample_path)
    if LAMBDA_TARGET is not None:
        lam_idx = int(np.argmin(np.abs(lam - float(LAMBDA_TARGET))))
    else:
        lam_idx = int(LAMBDA_INDEX)
    lam_idx = max(0, min(lam_idx, len(lam) - 1))
    lam_val = float(lam[lam_idx])
    metal_eps = np.complex64(au_eps_from_lambda_m(lam_val))
    dielectric_eps = np.complex64(sio2_eps_from_lambda_m(lam_val))
    lam_norm = normalize_interval(lam)[lam_idx]
    omega = 2.0 * math.pi * C0 / max(lam_val, 1e-12)

    eps, metal_mask = build_eps_volume(pattern_11, xv, yv, zv, metal_eps, dielectric_eps, air_eps, bottom_zmax, dielectric_zmax, top_zmax)
    input_static = np.stack(
        [
            metal_mask.astype(np.float32),
            np.real(eps).astype(np.float32),
            np.imag(eps).astype(np.float32),
            x_map,
            y_map,
            z_map,
        ],
        axis=0,
    )

    full_shape = (len(xv_full), len(yv_full), len(zv_full))
    true_fields = {}
    for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        arr = read_field_slice(sample_path, f"{key}_vol", lam_idx, full_shape)
        arr = arr[::down_x, ::down_y, ::down_z]
        true_fields[key] = arr

    target_stack = []
    for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        target_stack.append(np.real(true_fields[key]).astype(np.float32))
        target_stack.append(np.imag(true_fields[key]).astype(np.float32))
    target_stack = np.stack(target_stack, axis=0).astype(np.float32)
    scale = np.sqrt(np.mean(target_stack ** 2, dtype=np.float64) + 1e-12).astype(np.float32)

    model = FNO3dConditionalField(
        base_in=6,
        modes_x=modes_x,
        modes_y=modes_y,
        modes_z=modes_z,
        width=width,
        depth=depth,
        lam_ff=lam_ff,
        head_hidden=head_hidden,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(input_static).unsqueeze(0).to(DEVICE)
        lam_tensor = torch.tensor([[lam_norm]], dtype=torch.float32, device=DEVICE)
        pred_norm, pred_s_one = model(x_tensor, lam_tensor)
        pred_norm = pred_norm[0].detach().cpu().numpy()
        pred_s_one = project_to_passive(pred_s_one).detach().cpu().numpy()[0]
    pred_stack = pred_norm * float(scale)
    pred_fields = split_pred_channels(pred_stack)

    lam_norm_all = normalize_interval(lam).astype(np.float32)
    with torch.no_grad():
        input_curve = []
        for lam_item in lam:
            metal_eps_i = np.complex64(au_eps_from_lambda_m(float(lam_item)))
            dielectric_eps_i = np.complex64(sio2_eps_from_lambda_m(float(lam_item)))
            eps_i, metal_mask_i = build_eps_volume(
                pattern_11,
                xv,
                yv,
                zv,
                metal_eps_i,
                dielectric_eps_i,
                air_eps,
                bottom_zmax,
                dielectric_zmax,
                top_zmax,
            )
            input_curve.append(
                np.stack(
                    [
                        metal_mask_i.astype(np.float32),
                        np.real(eps_i).astype(np.float32),
                        np.imag(eps_i).astype(np.float32),
                        x_map,
                        y_map,
                        z_map,
                    ],
                    axis=0,
                )
            )
        x_curve = torch.from_numpy(np.stack(input_curve, axis=0)).to(DEVICE)
        lam_curve = torch.from_numpy(lam_norm_all[:, None]).to(DEVICE)
        _, pred_s_curve = model(x_curve, lam_curve)
        pred_s_curve = project_to_passive(pred_s_curve).detach().cpu().numpy()

    s11_pred = pred_s_curve[:, 0] + 1j * pred_s_curve[:, 1]
    s21_pred = pred_s_curve[:, 2] + 1j * pred_s_curve[:, 3]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    field_fig_path = SAVE_DIR / f"sample_{SAMPLE_ID:05d}_{FIELD_COMPONENT}_lambda_{lam_idx:03d}.png"
    sparam_fig_path = SAVE_DIR / f"sample_{SAMPLE_ID:05d}_sparams.png"

    plot_field_comparison(xv, yv, zv, true_fields[FIELD_COMPONENT], pred_fields[FIELD_COMPONENT], FIELD_COMPONENT, lam_val, field_fig_path)
    plot_sparams(lam, s11, s21, r, t, a, s11_pred, s21_pred, lam_idx, sparam_fig_path)

    rel_l2 = np.linalg.norm(pred_fields[FIELD_COMPONENT].reshape(-1) - true_fields[FIELD_COMPONENT].reshape(-1)) / (
        np.linalg.norm(true_fields[FIELD_COMPONENT].reshape(-1)) + 1e-12
    )
    s11_rel = np.linalg.norm(s11_pred - s11) / (np.linalg.norm(s11) + 1e-12)
    s21_rel = np.linalg.norm(s21_pred - s21) / (np.linalg.norm(s21) + 1e-12)

    print(f"device = {DEVICE}")
    print(f"checkpoint = {resolved_checkpoint_path}")
    print(f"checkpoint_source = {checkpoint_source}")
    print(f"sample = {SAMPLE_ID}")
    print(f"lambda index = {lam_idx}, lambda = {lam_val*1e6:.4f} um")
    print(f"field component = {FIELD_COMPONENT}, view = {FIELD_VIEW}")
    print(f"{FIELD_COMPONENT} relative L2 error = {rel_l2:.6e}")
    print(f"S11 relative L2 error = {s11_rel:.6e}")
    print(f"S21 relative L2 error = {s21_rel:.6e}")
    print(f"field figure saved to: {field_fig_path}")
    print(f"s-parameter figure saved to: {sparam_fig_path}")
    print("说明：当前脚本会同时画预测场分布和预测/参考 S 参数曲线。")


if __name__ == "__main__":
    main()
