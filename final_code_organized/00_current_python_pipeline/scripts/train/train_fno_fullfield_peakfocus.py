import copy
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except ImportError:  # pragma: no cover
    scipy_find_peaks = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.material_dispersion import au_eps_from_lambda_m, sio2_eps_from_lambda_m
from src.project_paths import (
    BEST_MODEL_HISTORY_ROOT,
    FIELD_DATA_DIR,
    MODELS_CURRENT_DIR,
    SAMPLING_META_PATH,
    TENSORBOARD_RUNS_DIR,
    TRAIN_RUN_OUTPUTS_DIR,
    ensure_standard_dirs,
)

print("TensorBoard：终端运行 tensorboard --logdir logs/tensorboard/runs  打开 http://localhost:6006/")


DATA_DIR = FIELD_DATA_DIR
META_PATH = SAMPLING_META_PATH
SAVE_PATH_FINAL = MODELS_CURRENT_DIR / "fno_fullfield_maxwell_dual_final.pt"
SAVE_PATH_BEST = MODELS_CURRENT_DIR / "fno_fullfield_maxwell_dual_best.pt"
BEST_HISTORY_ROOT = BEST_MODEL_HISTORY_ROOT
RUN_OUTPUTS_ROOT = TRAIN_RUN_OUTPUTS_DIR

SEED = 42
TRAIN_RATIO = 0.85
TRAIN_SAMPLE_LIMIT = 2000
VAL_SAMPLE_LIMIT = 256

EPOCHS = 100
VAL_EVERY = 5
MIN_EPOCHS = 20
PATIENCE = 12
TRAIN_PROGRESS_EVERY = 10

TRAIN_GROUP_SIZE = 8
VAL_GROUP_SIZE = 8
BATCH_SAMPLES = 8
VAL_BATCH_SAMPLES = 8
NUM_WORKERS = 0
PIN_MEMORY = True

DOWN_X = 3
DOWN_Y = 3
DOWN_Z = 3

MODES_X = 8
MODES_Y = 8
MODES_Z = 8
WIDTH = 32
DEPTH = 3
LAM_FF = 6
HEAD_HIDDEN = 128

LR = 2.5e-4
WEIGHT_DECAY = 3.0e-5
GRAD_CLIP = 1.0
USE_AMP = True
AMP_DTYPE = "bfloat16"

HUBER_BETA = 0.02
LAMBDA_FIELD = 0.55
LAMBDA_S11 = 0.10
LAMBDA_A_PEAK = 2.60
LAMBDA_T_ZERO = 2.20
LAMBDA_PASSIVE = 0.05
LAMBDA_PEAK_RANK = 1.20
LAMBDA_PEAK_POS = 2.20
LAMBDA_CURL_E = 0.004
LAMBDA_CURL_H = 0.004
LAMBDA_DIV = 0.0

PHYSICS_START_EPOCH = 18
PHYSICS_WARMUP_EPOCHS = 24
PHYSICS_LOSS_INTERVAL = 6
VAL_WITH_PHYSICS = False

PEAK_SMOOTH_K = 3
PEAK_MIN_HEIGHT = 0.05
PEAK_MIN_PROM = 0.03
PEAK_MIN_DISTANCE = 2
PEAK_EXCLUSION_RADIUS = 2

MAIN_WEIGHT = 14.0
SECONDARY_WEIGHT = 9.0
MAIN_NEIGHBOR_WEIGHT = 4.0
SECONDARY_NEIGHBOR_WEIGHT = 2.5
VALLEY_WEIGHT = 0.08
BACKGROUND_WEIGHT = 0.20
PEAK_SOFTMAX_TEMP = 14.0
T_ZERO_OVERRIDE = True

AIR_EPS = complex(1.0, 0.0)
BOTTOM_METAL_ZMAX = 100e-9
DIELECTRIC_ZMAX = 400e-9
TOP_PATTERN_ZMAX = 430e-9

HDF5_RETRIES = 3
HDF5_RETRY_SLEEP = 0.15
READ_FALLBACK_SAMPLES = 4
MAX_READ_WARNINGS = 20
SEQUENTIAL_BLOCK_SIZE = 64

LOG_ROOT = TENSORBOARD_RUNS_DIR / "fno_fullfield_peakfocus"

C0 = 299792458.0
EPS0 = 8.854187817e-12
MU0 = 4.0e-7 * math.pi
FIELD_KEYS = ("Ex_vol", "Ey_vol", "Ez_vol", "Hx_vol", "Hy_vol", "Hz_vol")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_amp_dtype():
    return torch.bfloat16 if AMP_DTYPE.lower() == "bfloat16" else torch.float16


ensure_standard_dirs()
set_seed(SEED)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def normalize_interval(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return 2.0 * (v - vmin) / (vmax - vmin + 1e-12) - 1.0


def standardize_coord_1d(arr) -> np.ndarray:
    return np.asarray(arr).squeeze().astype(np.float32).reshape(-1)


def standardize_pattern_11x11(arr) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if arr.shape != (11, 11):
        raise ValueError(f"binary_matrix 形状异常：{arr.shape}")
    return (arr != 0).astype(np.float32)


def decode_complex_array(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if hasattr(arr, "dtype") and arr.dtype.fields is not None:
        fields = set(arr.dtype.fields.keys())
        if "real" in fields and "imag" in fields:
            return arr["real"] + 1j * arr["imag"]
    return arr


def load_mat_auto(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    if h5py is not None:
        try:
            if h5py.is_hdf5(str(path)):
                out = {}
                with h5py.File(str(path), "r") as f:
                    for k in f.keys():
                        out[k] = decode_complex_array(f[k][()])
                return out
        except Exception:
            pass
    if loadmat is None:
        raise RuntimeError("需要 scipy.io.loadmat 读取非 v7.3 mat 文件。")
    raw = loadmat(str(path))
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def open_h5_readonly(path: Path):
    if h5py is None:
        raise RuntimeError("需要 h5py 读取 v7.3 / HDF5 样本文件。")
    for kwargs in ({"mode": "r", "locking": False}, {"mode": "r"}):
        try:
            return h5py.File(str(path), **kwargs)
        except TypeError:
            continue
        except OSError:
            continue
    return h5py.File(str(path), "r")


def nearest_resize_2d(pattern_11: np.ndarray, nx: int, ny: int) -> np.ndarray:
    x = torch.from_numpy(pattern_11.astype(np.float32)[None, None, ...])
    y = F.interpolate(x, size=(nx, ny), mode="nearest")
    return y[0, 0].numpy()


def make_coord_maps(xv: np.ndarray, yv: np.ndarray, zv: np.ndarray):
    x_norm = normalize_interval(xv)
    y_norm = normalize_interval(yv)
    z_norm = normalize_interval(zv)
    x_map = np.repeat(x_norm[:, None, None], len(yv), axis=1).repeat(len(zv), axis=2)
    y_map = np.repeat(y_norm[None, :, None], len(xv), axis=0).repeat(len(zv), axis=2)
    z_map = np.repeat(z_norm[None, None, :], len(xv), axis=0).repeat(len(yv), axis=1)
    return x_map.astype(np.float32), y_map.astype(np.float32), z_map.astype(np.float32)


def permute_to_xyz(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == target_shape:
        return arr
    import itertools

    for perm in itertools.permutations(range(arr.ndim)):
        if tuple(arr.shape[p] for p in perm) == tuple(target_shape):
            return np.transpose(arr, perm)
    raise ValueError(f"无法把 shape={arr.shape} 变成目标空间 shape={target_shape}")


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if k <= 1 or len(y) < 3:
        return y.copy()
    pad = k // 2
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def detect_top_two_peaks(a_curve: np.ndarray) -> tuple[int, int]:
    y = moving_average(np.clip(a_curve, 0.0, 1.0), PEAK_SMOOTH_K)
    max_y = float(np.max(y)) if len(y) > 0 else 0.0
    peaks = []
    if max_y > 1e-8 and scipy_find_peaks is not None:
        try:
            found, _ = scipy_find_peaks(
                y,
                height=max(PEAK_MIN_HEIGHT, 0.08 * max_y),
                prominence=max(PEAK_MIN_PROM, 0.05 * max_y),
                distance=PEAK_MIN_DISTANCE,
            )
            peaks = list(np.asarray(found, dtype=np.int64))
        except Exception:
            peaks = []
    if not peaks:
        for i in range(1, len(y) - 1):
            if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= PEAK_MIN_HEIGHT:
                peaks.append(i)
    if not peaks:
        main_idx = int(np.argmax(y))
    else:
        peaks = sorted(peaks, key=lambda idx: float(y[idx]), reverse=True)
        main_idx = int(peaks[0])
    banned = set(range(max(0, main_idx - PEAK_EXCLUSION_RADIUS), min(len(y), main_idx + PEAK_EXCLUSION_RADIUS + 1)))
    second_idx = None
    for idx in peaks[1:]:
        if idx not in banned:
            second_idx = int(idx)
            break
    if second_idx is None:
        for idx in np.argsort(-y):
            if int(idx) not in banned:
                second_idx = int(idx)
                break
    if second_idx is None:
        second_idx = main_idx
    return main_idx, second_idx


def choose_neighbor_index(center: int, n_lambda: int, banned: set[int], rng: np.random.Generator | None) -> int:
    candidates = []
    for delta in (1, -1, 2, -2, 3, -3):
        idx = center + delta
        if 0 <= idx < n_lambda and idx not in banned:
            candidates.append(idx)
    if candidates:
        return int(candidates[0] if rng is None else candidates[int(rng.integers(0, len(candidates)))])
    for idx in range(n_lambda):
        if idx not in banned:
            return int(idx)
    return int(center)


def choose_directional_index(center: int, direction: int, n_lambda: int, banned: set[int]) -> int:
    for step in range(1, 6):
        idx = center + direction * step
        if 0 <= idx < n_lambda and idx not in banned:
            return int(idx)
    for step in range(1, 6):
        idx = center - direction * step
        if 0 <= idx < n_lambda and idx not in banned:
            return int(idx)
    return int(center)


def choose_valley_index(a_curve: np.ndarray, banned: set[int], rng: np.random.Generator | None) -> int:
    remaining = [idx for idx in range(len(a_curve)) if idx not in banned]
    if not remaining:
        return 0
    order = np.argsort(np.asarray(a_curve, dtype=np.float32)[remaining])
    pool = [remaining[int(i)] for i in order[: max(1, min(len(order), 8))]]
    return int(pool[0] if rng is None else pool[int(rng.integers(0, len(pool)))])


def choose_background_index(a_curve: np.ndarray, banned: set[int], rng: np.random.Generator | None) -> int:
    remaining = [idx for idx in range(len(a_curve)) if idx not in banned]
    if not remaining:
        return 0
    values = np.asarray(a_curve, dtype=np.float32)[remaining]
    order = np.argsort(values)
    lo = max(0, int(0.25 * len(order)))
    hi = max(lo + 1, int(0.65 * len(order)))
    pool = [remaining[int(i)] for i in order[lo:hi]]
    if not pool:
        pool = remaining
    return int(pool[0] if rng is None else pool[int(rng.integers(0, len(pool)))])


@dataclass
class SampleRecord:
    path: Path
    sample_id: int
    pattern_11: np.ndarray
    pattern_xy: np.ndarray
    lambda_vec: np.ndarray
    s11_curve: np.ndarray
    a_curve: np.ndarray
    main_idx: int
    secondary_idx: int


@dataclass
class SamplePlan:
    lambda_indices: np.ndarray
    peak_weights: np.ndarray
    physics_mask: np.ndarray


def read_sample_header(sample_path: Path):
    if h5py is not None and h5py.is_hdf5(str(sample_path)):
        with open_h5_readonly(sample_path) as f:
            lam = np.asarray(f["lambda"][()]).reshape(-1).astype(np.float32)
            s11 = decode_complex_array(f["S11_ref"][()]).reshape(-1).astype(np.complex64)
            binary = standardize_pattern_11x11(decode_complex_array(f["binary_matrix"][()]))
            return lam, s11, binary
    data = load_mat_auto(sample_path)
    lam = np.asarray(data["lambda"]).reshape(-1).astype(np.float32)
    s11 = decode_complex_array(data["S11_ref"]).reshape(-1).astype(np.complex64)
    binary = standardize_pattern_11x11(data["binary_matrix"])
    return lam, s11, binary


def read_group_field_slices(sample_path: Path, field_keys, lambda_indices, target_shape):
    requested = [int(x) for x in lambda_indices]
    unique_sorted = sorted(set(requested))
    cache = {field_key: {} for field_key in field_keys}

    if h5py is not None and h5py.is_hdf5(str(sample_path)):
        with open_h5_readonly(sample_path) as f:
            n_lambda = int(np.asarray(f["lambda"][()]).size)
            for field_key in field_keys:
                ds = f[field_key]
                shape = ds.shape
                lambda_axes = [i for i, s in enumerate(shape) if s == n_lambda]
                if not lambda_axes:
                    raise ValueError(f"{field_key} 的 shape={shape} 中找不到波长轴。")
                chosen = lambda_axes[0]
                for lam_axis in lambda_axes:
                    rest = [shape[i] for i in range(len(shape)) if i != lam_axis]
                    if sorted(rest) == sorted(target_shape):
                        chosen = lam_axis
                        break
                for lam_idx in unique_sorted:
                    slc = [slice(None)] * len(shape)
                    slc[chosen] = int(lam_idx)
                    arr = decode_complex_array(ds[tuple(slc)])
                    arr = permute_to_xyz(np.asarray(arr).squeeze(), target_shape)
                    cache[field_key][lam_idx] = arr.astype(np.complex64)
    else:
        data = load_mat_auto(sample_path)
        n_lambda = len(np.asarray(data["lambda"]).reshape(-1))
        for field_key in field_keys:
            arr = np.asarray(decode_complex_array(data[field_key])).squeeze()
            lambda_axes = [i for i, s in enumerate(arr.shape) if s == n_lambda]
            if not lambda_axes:
                raise ValueError(f"{field_key} 未找到波长轴，shape={arr.shape}")
            chosen = lambda_axes[0]
            for lam_idx in unique_sorted:
                one = np.take(arr, int(lam_idx), axis=chosen)
                one = permute_to_xyz(one, target_shape)
                cache[field_key][lam_idx] = one.astype(np.complex64)

    return {field_key: np.stack([cache[field_key][i] for i in requested], axis=0) for field_key in field_keys}


class PeakFocusedGroupedDataset(Dataset):
    def __init__(self, sample_files: list[Path], meta: dict, train: bool):
        self.sample_files = [Path(p) for p in sample_files]
        self.meta = meta
        self.train = bool(train)
        self.dataset_name = "train" if self.train else "val"
        self.read_warning_count = 0

        xv_full = standardize_coord_1d(meta["xv"])
        yv_full = standardize_coord_1d(meta["yv"])
        zv_full = standardize_coord_1d(meta["zv"])
        self.xv = xv_full[::DOWN_X]
        self.yv = yv_full[::DOWN_Y]
        self.zv = zv_full[::DOWN_Z]
        self.full_shape = (
            int(np.asarray(meta["Nx"]).squeeze()),
            int(np.asarray(meta["Ny"]).squeeze()),
            int(np.asarray(meta["Nz"]).squeeze()),
        )
        self.target_shape = (len(self.xv), len(self.yv), len(self.zv))
        self.nx, self.ny, self.nz = self.target_shape
        self.x_map, self.y_map, self.z_map = make_coord_maps(self.xv, self.yv, self.zv)
        self.bottom_mask_z = self.zv <= BOTTOM_METAL_ZMAX
        self.diel_mask_z = (self.zv > BOTTOM_METAL_ZMAX) & (self.zv <= DIELECTRIC_ZMAX)
        self.top_mask_z = (self.zv > DIELECTRIC_ZMAX) & (self.zv <= TOP_PATTERN_ZMAX)
        self.top_indices = np.where(self.top_mask_z)[0]
        self.coord_tensors = {
            "x": torch.from_numpy(self.xv.astype(np.float32)),
            "y": torch.from_numpy(self.yv.astype(np.float32)),
            "z": torch.from_numpy(self.zv.astype(np.float32)),
        }

        self.records: list[SampleRecord] = []
        for path in self.sample_files:
            try:
                sample_id = int(path.stem.split("_")[-1])
                lam, s11_curve, pattern_11 = read_sample_header(path)
                a_curve = np.clip(1.0 - np.abs(s11_curve) ** 2, 0.0, 1.0).astype(np.float32)
                main_idx, secondary_idx = detect_top_two_peaks(a_curve)
                pattern_xy = nearest_resize_2d(pattern_11, self.nx, self.ny)
                self.records.append(
                    SampleRecord(
                        path=path,
                        sample_id=sample_id,
                        pattern_11=pattern_11.astype(np.float32),
                        pattern_xy=pattern_xy.astype(np.float32),
                        lambda_vec=lam.astype(np.float32),
                        s11_curve=s11_curve.astype(np.complex64),
                        a_curve=a_curve,
                        main_idx=main_idx,
                        secondary_idx=secondary_idx,
                    )
                )
            except Exception as exc:
                print(f"[{self.dataset_name}] 跳过异常样本头信息：{path.name} | {exc}")

        if not self.records:
            raise RuntimeError(f"{self.dataset_name} 数据集为空。")

        self.lambda_vec = self.records[0].lambda_vec.copy()
        self.lam_norm = normalize_interval(self.lambda_vec).astype(np.float32)
        self.sample_order = np.arange(len(self.records), dtype=np.int64)
        self.plans: list[SamplePlan] = []
        self.refresh(epoch=0)

    def __len__(self):
        return len(self.sample_order)

    def _build_plan(self, record: SampleRecord, rng: np.random.Generator | None) -> SamplePlan:
        n_lambda = len(record.lambda_vec)
        main_idx = int(record.main_idx)
        secondary_idx = int(record.secondary_idx)
        banned: set[int] = set()

        main_left_idx = choose_directional_index(main_idx, -1, n_lambda, banned | {main_idx})
        banned.update({main_left_idx, main_idx})
        main_right_idx = choose_directional_index(main_idx, +1, n_lambda, banned)
        banned.update({main_right_idx})

        banned.update(set(range(max(0, main_idx - PEAK_EXCLUSION_RADIUS), min(n_lambda, main_idx + PEAK_EXCLUSION_RADIUS + 1))))
        banned.update(set(range(max(0, secondary_idx - PEAK_EXCLUSION_RADIUS), min(n_lambda, secondary_idx + PEAK_EXCLUSION_RADIUS + 1))))

        secondary_left_idx = choose_directional_index(secondary_idx, -1, n_lambda, banned | {secondary_idx})
        banned.update({secondary_left_idx, secondary_idx})
        secondary_right_idx = choose_directional_index(secondary_idx, +1, n_lambda, banned)
        banned.update({secondary_right_idx})

        valley_idx = choose_valley_index(record.a_curve, banned, rng)
        banned.add(valley_idx)
        background_idx = choose_background_index(record.a_curve, banned, rng)
        return SamplePlan(
            lambda_indices=np.array(
                [
                    main_left_idx,
                    main_idx,
                    main_right_idx,
                    secondary_left_idx,
                    secondary_idx,
                    secondary_right_idx,
                    valley_idx,
                    background_idx,
                ],
                dtype=np.int64,
            ),
            peak_weights=np.array(
                [
                    MAIN_NEIGHBOR_WEIGHT,
                    MAIN_WEIGHT,
                    MAIN_NEIGHBOR_WEIGHT,
                    SECONDARY_NEIGHBOR_WEIGHT,
                    SECONDARY_WEIGHT,
                    SECONDARY_NEIGHBOR_WEIGHT,
                    VALLEY_WEIGHT,
                    BACKGROUND_WEIGHT,
                ],
                dtype=np.float32,
            ),
            physics_mask=np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        )

    def refresh(self, epoch: int) -> None:
        if self.train:
            rng = np.random.default_rng(SEED + 10007 * int(epoch))
            blocks = [np.arange(i, min(i + SEQUENTIAL_BLOCK_SIZE, len(self.records)), dtype=np.int64) for i in range(0, len(self.records), SEQUENTIAL_BLOCK_SIZE)]
            block_order = rng.permutation(len(blocks))
            self.sample_order = np.concatenate([blocks[int(i)] for i in block_order], axis=0)
            self.plans = [self._build_plan(self.records[int(i)], rng) for i in self.sample_order]
        else:
            self.sample_order = np.arange(len(self.records), dtype=np.int64)
            self.plans = [self._build_plan(self.records[int(i)], None) for i in self.sample_order]

    def _build_input_static(self, record: SampleRecord, lambda_vals: np.ndarray):
        k = len(lambda_vals)
        x_static = np.empty((k, 6, self.nx, self.ny, self.nz), dtype=np.float32)
        eps_ri = np.empty((k, 2, self.nx, self.ny, self.nz), dtype=np.float32)
        pattern_xy = record.pattern_xy > 0.5
        metal_mask_base = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        metal_mask_base[:, :, self.bottom_mask_z] = 1.0

        for i, lam_val in enumerate(lambda_vals):
            metal_eps = np.complex64(au_eps_from_lambda_m(float(lam_val)))
            diel_eps = np.complex64(sio2_eps_from_lambda_m(float(lam_val)))
            eps = np.full((self.nx, self.ny, self.nz), AIR_EPS, dtype=np.complex64)
            eps[:, :, self.bottom_mask_z] = metal_eps
            eps[:, :, self.diel_mask_z] = diel_eps
            if np.any(self.top_mask_z):
                eps[:, :, self.top_mask_z] = diel_eps
                for zi in self.top_indices:
                    eps[:, :, zi][pattern_xy] = metal_eps

            metal_mask = metal_mask_base.copy()
            for zi in self.top_indices:
                metal_mask[:, :, zi][pattern_xy] = 1.0

            x_static[i] = np.stack(
                [
                    metal_mask,
                    np.real(eps).astype(np.float32),
                    np.imag(eps).astype(np.float32),
                    self.x_map,
                    self.y_map,
                    self.z_map,
                ],
                axis=0,
            )
            eps_ri[i, 0] = np.real(eps).astype(np.float32)
            eps_ri[i, 1] = np.imag(eps).astype(np.float32)

        return x_static, eps_ri

    def _build_item_from_order_index(self, order_index: int):
        record = self.records[int(self.sample_order[order_index])]
        plan = self.plans[order_index]
        lambda_indices = plan.lambda_indices
        lambda_vals = record.lambda_vec[lambda_indices]

        field_dict = None
        last_error = None
        for attempt in range(HDF5_RETRIES):
            try:
                field_dict = read_group_field_slices(record.path, FIELD_KEYS, lambda_indices, self.full_shape)
                break
            except (OSError, PermissionError) as exc:
                last_error = exc
                time.sleep(HDF5_RETRY_SLEEP * (attempt + 1))
        if field_dict is None:
            raise last_error if last_error is not None else RuntimeError(f"读取失败：{record.path}")

        x_static, eps_ri = self._build_input_static(record, lambda_vals)

        target = np.empty((len(lambda_indices), 12, self.nx, self.ny, self.nz), dtype=np.float32)
        scale = np.empty((len(lambda_indices),), dtype=np.float32)
        for li in range(len(lambda_indices)):
            channels = []
            for field_key in FIELD_KEYS:
                arr = field_dict[field_key][li][::DOWN_X, ::DOWN_Y, ::DOWN_Z]
                channels.append(np.real(arr).astype(np.float32))
                channels.append(np.imag(arr).astype(np.float32))
            target_i = np.stack(channels, axis=0).astype(np.float32)
            scale_i = np.float32(np.sqrt(np.mean(target_i ** 2, dtype=np.float64) + 1e-12))
            target[li] = target_i / max(float(scale_i), 1e-6)
            scale[li] = scale_i

        s11_curve = record.s11_curve[lambda_indices]
        s11_target = np.stack([np.real(s11_curve), np.imag(s11_curve)], axis=-1).astype(np.float32)
        a_target = record.a_curve[lambda_indices].astype(np.float32)[:, None]
        lam_norm = self.lam_norm[lambda_indices][:, None]
        omega = (2.0 * math.pi * C0 / np.maximum(lambda_vals.astype(np.float64), 1e-12)).astype(np.float32)

        return {
            "x": torch.from_numpy(x_static),
            "lam_norm": torch.from_numpy(lam_norm.astype(np.float32)),
            "lambda_raw": torch.from_numpy(lambda_vals.astype(np.float32)[:, None]),
            "target": torch.from_numpy(target),
            "eps": torch.from_numpy(eps_ri),
            "scale": torch.from_numpy(scale.astype(np.float32)),
            "s11_target": torch.from_numpy(s11_target),
            "a_target": torch.from_numpy(a_target),
            "peak_weight": torch.from_numpy(plan.peak_weights[:, None].astype(np.float32)),
            "physics_mask": torch.from_numpy(plan.physics_mask.astype(np.float32)),
            "omega": torch.from_numpy(omega.astype(np.float32)),
            "sample_id": torch.tensor(record.sample_id, dtype=torch.long),
            "lambda_indices": torch.from_numpy(lambda_indices.astype(np.int64)),
        }

    def __getitem__(self, idx):
        if len(self.sample_order) == 0:
            raise IndexError("数据集为空。")
        last_error = None
        base_idx = int(idx) % len(self.sample_order)
        for shift in range(READ_FALLBACK_SAMPLES + 1):
            try_idx = (base_idx + shift) % len(self.sample_order)
            try:
                return self._build_item_from_order_index(try_idx)
            except (OSError, PermissionError) as exc:
                last_error = exc
                self.read_warning_count += 1
                if self.read_warning_count <= MAX_READ_WARNINGS:
                    record = self.records[int(self.sample_order[try_idx])]
                    print(f"[{self.dataset_name}] 样本读取失败，换下一个样本：sample={record.sample_id}, file={record.path.name}, shift={shift}, error={exc}")
                time.sleep(HDF5_RETRY_SLEEP)
        raise last_error if last_error is not None else RuntimeError("读取样本失败。")


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
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat))

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1), norm="ortho")
            out_ft = torch.zeros(x.shape[0], self.weight.shape[1], x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
            mx = min(self.modes_x, x_ft.shape[-3])
            my = min(self.modes_y, x_ft.shape[-2])
            mz = min(self.modes_z, x_ft.shape[-1])
            out_ft[:, :, :mx, :my, :mz] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :mx, :my, :mz], self.weight[:, :, :mx, :my, :mz])
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
        self.head = nn.Sequential(nn.Conv3d(width, head_hidden, 1), nn.GELU(), nn.Conv3d(head_hidden, 12, 1))
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


def split_complex_channels(pred):
    return {
        "Ex": torch.complex(pred[:, 0], pred[:, 1]),
        "Ey": torch.complex(pred[:, 2], pred[:, 3]),
        "Ez": torch.complex(pred[:, 4], pred[:, 5]),
        "Hx": torch.complex(pred[:, 6], pred[:, 7]),
        "Hy": torch.complex(pred[:, 8], pred[:, 9]),
        "Hz": torch.complex(pred[:, 10], pred[:, 11]),
    }


def finite_difference(values, coords, axis):
    moved = torch.movedim(values, axis, -1)
    n = moved.shape[-1]
    deriv = torch.zeros_like(moved)
    if n < 2:
        return torch.movedim(deriv, -1, axis)

    coords = coords.to(values.device)
    view_shape = [coords.shape[0]] + [1] * (moved.ndim - 2)
    dx0 = (coords[:, 1] - coords[:, 0]).clamp_min(1e-12).view(*view_shape)
    dxn = (coords[:, -1] - coords[:, -2]).clamp_min(1e-12).view(*view_shape)
    deriv[..., 0] = (moved[..., 1] - moved[..., 0]) / dx0
    deriv[..., -1] = (moved[..., -1] - moved[..., -2]) / dxn

    if n > 2:
        dxm = (coords[:, 1:-1] - coords[:, :-2]).clamp_min(1e-12)
        dxp = (coords[:, 2:] - coords[:, 1:-1]).clamp_min(1e-12)
        c_prev = (-dxp / (dxm * (dxm + dxp))).view(coords.shape[0], *([1] * (moved.ndim - 2)), n - 2)
        c_mid = ((dxp - dxm) / (dxm * dxp)).view(coords.shape[0], *([1] * (moved.ndim - 2)), n - 2)
        c_next = (dxm / (dxp * (dxm + dxp))).view(coords.shape[0], *([1] * (moved.ndim - 2)), n - 2)
        deriv[..., 1:-1] = c_prev * moved[..., :-2] + c_mid * moved[..., 1:-1] + c_next * moved[..., 2:]
    return torch.movedim(deriv, -1, axis)


def curl_3d(vx, vy, vz, coords):
    d_vz_dy = finite_difference(vz, coords["y"], axis=2)
    d_vy_dz = finite_difference(vy, coords["z"], axis=3)
    d_vx_dz = finite_difference(vx, coords["z"], axis=3)
    d_vz_dx = finite_difference(vz, coords["x"], axis=1)
    d_vy_dx = finite_difference(vy, coords["x"], axis=1)
    d_vx_dy = finite_difference(vx, coords["y"], axis=2)
    return d_vz_dy - d_vy_dz, d_vx_dz - d_vz_dx, d_vy_dx - d_vx_dy


def divergence_3d(vx, vy, vz, coords):
    return (
        finite_difference(vx, coords["x"], axis=1)
        + finite_difference(vy, coords["y"], axis=2)
        + finite_difference(vz, coords["z"], axis=3)
    )


def manual_huber_elementwise(pred, target, delta=0.02):
    diff = pred - target
    abs_diff = torch.abs(diff)
    quad = torch.clamp(abs_diff, max=delta)
    lin = abs_diff - quad
    return 0.5 * quad ** 2 + delta * lin


def weighted_mean(loss_tensor, weight_tensor):
    weight = weight_tensor
    while weight.ndim < loss_tensor.ndim:
        weight = weight.unsqueeze(-1)
    return torch.sum(loss_tensor * weight) / (torch.sum(weight) * float(np.prod(loss_tensor.shape[loss_tensor.ndim - (loss_tensor.ndim - weight_tensor.ndim):])) + 1e-12)


def weighted_huber_loss(pred, target, weight, delta=0.02):
    loss = manual_huber_elementwise(pred, target, delta=delta)
    weight_view = weight
    while weight_view.ndim < loss.ndim:
        weight_view = weight_view.unsqueeze(-1)
    return torch.sum(loss * weight_view) / (torch.sum(weight_view.expand_as(loss)) + 1e-12)


def weighted_mse_loss(pred, target, weight):
    loss = (pred - target) ** 2
    weight_view = weight
    while weight_view.ndim < loss.ndim:
        weight_view = weight_view.unsqueeze(-1)
    return torch.sum(loss * weight_view) / (torch.sum(weight_view.expand_as(loss)) + 1e-12)


def rms_abs(x):
    return torch.sqrt(torch.mean(torch.abs(x) ** 2) + 1e-12)


def mean_spacing(coord):
    if coord.shape[-1] < 2:
        return torch.ones(coord.shape[0], device=coord.device, dtype=coord.dtype)
    return torch.mean(torch.diff(coord, dim=-1), dim=-1).clamp_min(1e-12)


def project_to_passive(pred_s, eps=1e-12):
    power = pred_s[:, 0] ** 2 + pred_s[:, 1] ** 2 + pred_s[:, 2] ** 2 + pred_s[:, 3] ** 2
    scale = torch.where(power > 1.0, torch.rsqrt(power + eps), torch.ones_like(power))
    out = pred_s.clone()
    out[:, 0] = pred_s[:, 0] * scale
    out[:, 1] = pred_s[:, 1] * scale
    out[:, 2] = pred_s[:, 2] * scale
    out[:, 3] = pred_s[:, 3] * scale
    return out


def power_excess(pred_s):
    return F.relu(torch.sum(pred_s ** 2, dim=-1) - 1.0)


def s_to_absorption_torch(pred_s):
    return torch.clamp(1.0 - torch.sum(pred_s ** 2, dim=-1), min=0.0, max=1.0)


def maxwell_residual_loss(pred_field, eps_ri, omega, coords):
    field = split_complex_channels(pred_field)
    eps = torch.complex(eps_ri[:, 0], eps_ri[:, 1])
    omega = omega.view(-1, 1, 1, 1).to(pred_field.device)

    curl_e = curl_3d(field["Ex"], field["Ey"], field["Ez"], coords)
    curl_h = curl_3d(field["Hx"], field["Hy"], field["Hz"], coords)

    faraday_terms = [1j * omega * MU0 * h for h in (field["Hx"], field["Hy"], field["Hz"])]
    ampere_terms = [1j * omega * EPS0 * eps * e for e in (field["Ex"], field["Ey"], field["Ez"])]
    curl_e_res = [ce - fh for ce, fh in zip(curl_e, faraday_terms)]
    curl_h_res = [ch + ah for ch, ah in zip(curl_h, ampere_terms)]

    div_d = divergence_3d(eps * field["Ex"], eps * field["Ey"], eps * field["Ez"], coords)
    div_b = divergence_3d(MU0 * field["Hx"], MU0 * field["Hy"], MU0 * field["Hz"], coords)

    rel_curl_e = []
    for res, ce, fh in zip(curl_e_res, curl_e, faraday_terms):
        denom = rms_abs(ce) + rms_abs(fh) + 1e-6
        rel_curl_e.append(torch.mean(torch.abs(res / denom) ** 2))

    rel_curl_h = []
    for res, ch, ah in zip(curl_h_res, curl_h, ampere_terms):
        denom = rms_abs(ch) + rms_abs(ah) + 1e-6
        rel_curl_h.append(torch.mean(torch.abs(res / denom) ** 2))

    dx = mean_spacing(coords["x"]).view(-1, 1, 1, 1)
    dy = mean_spacing(coords["y"]).view(-1, 1, 1, 1)
    dz = mean_spacing(coords["z"]).view(-1, 1, 1, 1)
    inv_len = 1.0 / torch.minimum(torch.minimum(dx, dy), dz)
    d_scale = rms_abs(eps * field["Ex"]) + rms_abs(eps * field["Ey"]) + rms_abs(eps * field["Ez"]) + 1e-6
    b_scale = rms_abs(MU0 * field["Hx"]) + rms_abs(MU0 * field["Hy"]) + rms_abs(MU0 * field["Hz"]) + 1e-6

    loss_curl_e = torch.stack(rel_curl_e).mean()
    loss_curl_h = torch.stack(rel_curl_h).mean()
    loss_div = torch.mean(torch.abs(div_d / (d_scale * inv_len + 1e-6)) ** 2) + torch.mean(torch.abs(div_b / (b_scale * inv_len + 1e-6)) ** 2)
    return loss_curl_e, loss_curl_h, loss_div


def peak_rank_loss(a_pred_grouped):
    main_left = a_pred_grouped[:, 0]
    main = a_pred_grouped[:, 1]
    main_right = a_pred_grouped[:, 2]
    secondary_left = a_pred_grouped[:, 3]
    secondary = a_pred_grouped[:, 4]
    secondary_right = a_pred_grouped[:, 5]
    valley = a_pred_grouped[:, 6]
    background = a_pred_grouped[:, 7]
    loss = (
        F.relu(0.14 - (main - torch.maximum(main_left, main_right)))
        + F.relu(0.10 - (secondary - torch.maximum(secondary_left, secondary_right)))
        + F.relu(0.18 - (main - valley))
        + F.relu(0.10 - (secondary - valley))
        + 0.5 * F.relu(0.08 - (main - background))
        + 0.3 * F.relu(0.03 - (main - secondary))
    )
    return loss.mean()


def local_peak_position_loss(a_pred_grouped, lambda_grouped):
    main_triplet = a_pred_grouped[:, 0:3]
    secondary_triplet = a_pred_grouped[:, 3:6]
    main_lambda = lambda_grouped[:, 0:3]
    secondary_lambda = lambda_grouped[:, 3:6]

    main_prob = torch.softmax(PEAK_SOFTMAX_TEMP * main_triplet, dim=1)
    secondary_prob = torch.softmax(PEAK_SOFTMAX_TEMP * secondary_triplet, dim=1)

    main_center_pred = torch.sum(main_prob * main_lambda, dim=1)
    secondary_center_pred = torch.sum(secondary_prob * secondary_lambda, dim=1)
    main_center_true = lambda_grouped[:, 1]
    secondary_center_true = lambda_grouped[:, 4]

    main_span = (main_lambda[:, -1] - main_lambda[:, 0]).abs().clamp_min(1e-9)
    secondary_span = (secondary_lambda[:, -1] - secondary_lambda[:, 0]).abs().clamp_min(1e-9)

    loss_main = torch.mean(torch.abs(main_center_pred - main_center_true) / main_span)
    loss_secondary = torch.mean(torch.abs(secondary_center_pred - secondary_center_true) / secondary_span)
    return 0.7 * loss_main + 0.3 * loss_secondary


def physics_ramp(epoch):
    if epoch < PHYSICS_START_EPOCH:
        return 0.0
    if PHYSICS_WARMUP_EPOCHS <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - PHYSICS_START_EPOCH + 1) / float(PHYSICS_WARMUP_EPOCHS))))


def build_coords_for_batch(base_coords, batch_size, device):
    return {
        "x": base_coords["x"].to(device).view(1, -1).expand(batch_size, -1),
        "y": base_coords["y"].to(device).view(1, -1).expand(batch_size, -1),
        "z": base_coords["z"].to(device).view(1, -1).expand(batch_size, -1),
    }


def compute_total_loss(
    pred_field,
    pred_s_raw,
    target_field,
    s11_target,
    a_target,
    lambda_raw,
    peak_weight,
    physics_mask,
    scale,
    eps_ri,
    omega,
    coords,
    epoch,
    train_mode,
    apply_physics,
    group_size,
):
    loss_field = weighted_huber_loss(pred_field, target_field, peak_weight, delta=HUBER_BETA)
    pred_s = project_to_passive(pred_s_raw)
    loss_s11 = weighted_huber_loss(pred_s[:, :2], s11_target, peak_weight, delta=HUBER_BETA)
    a_pred = s_to_absorption_torch(pred_s).unsqueeze(-1)
    loss_a_peak = weighted_mse_loss(a_pred, a_target, peak_weight)
    loss_t_zero = weighted_mse_loss(pred_s[:, 2:], torch.zeros_like(pred_s[:, 2:]), peak_weight)
    loss_passive = torch.mean(power_excess(pred_s_raw) ** 2)

    a_grouped = a_pred.reshape(-1, group_size)
    lambda_grouped = lambda_raw.reshape(-1, group_size)
    loss_rank = peak_rank_loss(a_grouped)
    loss_peak_pos = local_peak_position_loss(a_grouped, lambda_grouped)

    if apply_physics and torch.any(physics_mask > 0.5):
        choose = physics_mask > 0.5
        pred_physical = pred_field[choose] * scale[choose].view(-1, 1, 1, 1, 1)
        loss_curl_e, loss_curl_h, loss_div = maxwell_residual_loss(
            pred_physical,
            eps_ri[choose],
            omega[choose],
            build_coords_for_batch(coords, int(torch.count_nonzero(choose).item()), pred_field.device),
        )
    else:
        zero = pred_field.new_zeros(())
        loss_curl_e, loss_curl_h, loss_div = zero, zero, zero

    ramp = physics_ramp(epoch) if train_mode else 1.0
    total = (
        LAMBDA_FIELD * loss_field
        + LAMBDA_S11 * loss_s11
        + LAMBDA_A_PEAK * loss_a_peak
        + LAMBDA_T_ZERO * loss_t_zero
        + LAMBDA_PASSIVE * loss_passive
        + LAMBDA_PEAK_RANK * loss_rank
        + LAMBDA_PEAK_POS * loss_peak_pos
        + ramp * LAMBDA_CURL_E * loss_curl_e
        + ramp * LAMBDA_CURL_H * loss_curl_h
        + ramp * LAMBDA_DIV * loss_div
    )
    stats = {
        "field": float(loss_field.item()),
        "s11": float(loss_s11.item()),
        "a_peak": float(loss_a_peak.item()),
        "t_zero": float(loss_t_zero.item()),
        "passive": float(loss_passive.item()),
        "peak_rank": float(loss_rank.item()),
        "peak_pos": float(loss_peak_pos.item()),
        "curl_e": float(loss_curl_e.item()),
        "curl_h": float(loss_curl_h.item()),
        "div": float(loss_div.item()),
        "physics_ramp": float(ramp),
        "total": float(total.item()),
    }
    return total, stats


def flatten_group_batch(batch, device):
    x = batch["x"].to(device, non_blocking=True)
    lam_norm = batch["lam_norm"].to(device, non_blocking=True)
    lambda_raw = batch["lambda_raw"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    eps_ri = batch["eps"].to(device, non_blocking=True)
    scale = batch["scale"].to(device, non_blocking=True)
    s11_target = batch["s11_target"].to(device, non_blocking=True)
    a_target = batch["a_target"].to(device, non_blocking=True)
    peak_weight = batch["peak_weight"].to(device, non_blocking=True)
    physics_mask = batch["physics_mask"].to(device, non_blocking=True)
    omega = batch["omega"].to(device, non_blocking=True)

    bsz, group_size = x.shape[:2]
    flat = {
        "x": x.reshape(bsz * group_size, *x.shape[2:]),
        "lam_norm": lam_norm.reshape(bsz * group_size, 1),
        "lambda_raw": lambda_raw.reshape(bsz * group_size, 1),
        "target": target.reshape(bsz * group_size, *target.shape[2:]),
        "eps": eps_ri.reshape(bsz * group_size, *eps_ri.shape[2:]),
        "scale": scale.reshape(bsz * group_size),
        "s11_target": s11_target.reshape(bsz * group_size, 2),
        "a_target": a_target.reshape(bsz * group_size, 1),
        "peak_weight": peak_weight.reshape(bsz * group_size, 1),
        "physics_mask": physics_mask.reshape(bsz * group_size),
        "omega": omega.reshape(bsz * group_size),
        "group_size": group_size,
        "batch_samples": bsz,
    }
    return flat


@torch.no_grad()
def evaluate(model, loader, coords_base, device, epoch):
    model.eval()
    total_sum = 0.0
    field_sum = 0.0
    s11_sum = 0.0
    a_peak_sum = 0.0
    t_zero_sum = 0.0
    passive_sum = 0.0
    peak_rank_sum = 0.0
    peak_pos_sum = 0.0
    curl_e_sum = 0.0
    curl_h_sum = 0.0
    div_sum = 0.0
    count = 0

    use_autocast = USE_AMP and device.type == "cuda"
    for batch in loader:
        flat = flatten_group_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=get_amp_dtype() if use_autocast else None):
            pred_field, pred_s_raw = model(flat["x"], flat["lam_norm"])
            loss, stats = compute_total_loss(
                pred_field=pred_field,
                pred_s_raw=pred_s_raw,
                target_field=flat["target"],
                s11_target=flat["s11_target"],
                a_target=flat["a_target"],
                lambda_raw=flat["lambda_raw"],
                peak_weight=flat["peak_weight"],
                physics_mask=flat["physics_mask"],
                scale=flat["scale"],
                eps_ri=flat["eps"],
                omega=flat["omega"],
                coords=coords_base,
                epoch=epoch,
                train_mode=False,
                apply_physics=VAL_WITH_PHYSICS,
                group_size=flat["group_size"],
            )

        bs = flat["batch_samples"]
        total_sum += stats["total"] * bs
        field_sum += stats["field"] * bs
        s11_sum += stats["s11"] * bs
        a_peak_sum += stats["a_peak"] * bs
        t_zero_sum += stats["t_zero"] * bs
        passive_sum += stats["passive"] * bs
        peak_rank_sum += stats["peak_rank"] * bs
        peak_pos_sum += stats["peak_pos"] * bs
        curl_e_sum += stats["curl_e"] * bs
        curl_h_sum += stats["curl_h"] * bs
        div_sum += stats["div"] * bs
        count += bs

    return {
        "val_total": total_sum / max(count, 1),
        "val_field": field_sum / max(count, 1),
        "val_s11": s11_sum / max(count, 1),
        "val_a_peak": a_peak_sum / max(count, 1),
        "val_t_zero": t_zero_sum / max(count, 1),
        "val_passive": passive_sum / max(count, 1),
        "val_peak_rank": peak_rank_sum / max(count, 1),
        "val_peak_pos": peak_pos_sum / max(count, 1),
        "val_curl_e": curl_e_sum / max(count, 1),
        "val_curl_h": curl_h_sum / max(count, 1),
        "val_div": div_sum / max(count, 1),
    }


def build_checkpoint_payload(state_dict, best_epoch_value, best_val_loss_value, lambda_vec):
    return {
        "state_dict": state_dict,
        "config": {
            "MODES_X": MODES_X,
            "MODES_Y": MODES_Y,
            "MODES_Z": MODES_Z,
            "WIDTH": WIDTH,
            "DEPTH": DEPTH,
            "LAM_FF": LAM_FF,
            "HEAD_HIDDEN": HEAD_HIDDEN,
            "DOWN_X": DOWN_X,
            "DOWN_Y": DOWN_Y,
            "DOWN_Z": DOWN_Z,
            "TRAIN_GROUP_SIZE": TRAIN_GROUP_SIZE,
            "VAL_GROUP_SIZE": VAL_GROUP_SIZE,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LAMBDA_FIELD": LAMBDA_FIELD,
            "LAMBDA_S11": LAMBDA_S11,
            "LAMBDA_A_PEAK": LAMBDA_A_PEAK,
            "LAMBDA_T_ZERO": LAMBDA_T_ZERO,
            "LAMBDA_PASSIVE": LAMBDA_PASSIVE,
            "LAMBDA_PEAK_RANK": LAMBDA_PEAK_RANK,
            "LAMBDA_PEAK_POS": LAMBDA_PEAK_POS,
            "LAMBDA_CURL_E": LAMBDA_CURL_E,
            "LAMBDA_CURL_H": LAMBDA_CURL_H,
            "LAMBDA_DIV": LAMBDA_DIV,
            "PHYSICS_START_EPOCH": PHYSICS_START_EPOCH,
            "PHYSICS_WARMUP_EPOCHS": PHYSICS_WARMUP_EPOCHS,
            "PHYSICS_LOSS_INTERVAL": PHYSICS_LOSS_INTERVAL,
            "T_ZERO_OVERRIDE": T_ZERO_OVERRIDE,
            "MAIN_WEIGHT": MAIN_WEIGHT,
            "SECONDARY_WEIGHT": SECONDARY_WEIGHT,
            "MAIN_NEIGHBOR_WEIGHT": MAIN_NEIGHBOR_WEIGHT,
            "SECONDARY_NEIGHBOR_WEIGHT": SECONDARY_NEIGHBOR_WEIGHT,
            "VALLEY_WEIGHT": VALLEY_WEIGHT,
            "BACKGROUND_WEIGHT": BACKGROUND_WEIGHT,
            "METAL_MATERIAL": "Au / Ciesielski 2018 (Au/SiO2)",
            "DIELECTRIC_MATERIAL": "SiO2 / Kischkat 2012",
            "HUBER_BETA": HUBER_BETA,
            "SEED": SEED,
        },
        "best_epoch": best_epoch_value,
        "best_val_loss": best_val_loss_value,
        "lambda_vec": torch.tensor(lambda_vec, dtype=torch.float32),
    }


def main():
    meta = load_mat_auto(META_PATH)
    sample_name_pattern = re.compile(r"^sample_\d+\.mat$")
    all_sample_files = sorted(DATA_DIR.glob("sample_*.mat"))
    ignored_sample_files = [p for p in all_sample_files if not sample_name_pattern.match(p.name)]
    sample_files = [p for p in all_sample_files if sample_name_pattern.match(p.name)]
    if not sample_files:
        raise RuntimeError(f"鍦?{DATA_DIR} 涓病鏈夋壘鍒?sample_*.mat")

    if ignored_sample_files:
        print("跳过非标准样本文件数 =", len(ignored_sample_files))
        print("示例非标准文件 =", ignored_sample_files[0].name)

    n_total = len(sample_files)
    perm = np.random.default_rng(SEED).permutation(n_total)
    n_train = int(TRAIN_RATIO * n_total)
    train_files = sorted([sample_files[i] for i in perm[:n_train]], key=lambda p: p.name)
    val_files = sorted([sample_files[i] for i in perm[n_train:]], key=lambda p: p.name)

    if TRAIN_SAMPLE_LIMIT is not None:
        train_files = train_files[: int(TRAIN_SAMPLE_LIMIT)]
    if VAL_SAMPLE_LIMIT is not None:
        val_files = val_files[: int(VAL_SAMPLE_LIMIT)]
    if not val_files:
        raise RuntimeError("验证集为空，请检查 TRAIN_RATIO / VAL_SAMPLE_LIMIT")

    train_ds = PeakFocusedGroupedDataset(train_files, meta, train=True)
    val_ds = PeakFocusedGroupedDataset(val_files, meta, train=False)

    if TRAIN_GROUP_SIZE != 8 or VAL_GROUP_SIZE != 8:
        raise ValueError("当前峰值聚焦版脚本固定使用 8 个波长点：主峰三点窗口、次峰三点窗口、谷底、背景。")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SAMPLES,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SAMPLES,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )

    print("训练样本文件数 =", len(train_files))
    print("验证样本文件数 =", len(val_files))
    print("训练批次数 =", len(train_loader))
    print("验证批次数 =", len(val_loader))
    print("每个样本波长点 =", TRAIN_GROUP_SIZE)
    print("空间尺寸 =", train_ds.target_shape)
    print("训练读取策略 = 按样本成组顺序读取，块级打乱样本顺序，优先利用机械硬盘顺序吞吐。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device.type)
    if device.type == "cuda":
        print("AMP dtype =", AMP_DTYPE)

    model = FNO3dConditionalField(
        base_in=6,
        modes_x=MODES_X,
        modes_y=MODES_Y,
        modes_z=MODES_Z,
        width=WIDTH,
        depth=DEPTH,
        lam_ff=LAM_FF,
        head_hidden=HEAD_HIDDEN,
    ).to(device)

    has_complex_params = any(torch.is_complex(p) for p in model.parameters())
    scaler_enabled = USE_AMP and device.type == "cuda" and not has_complex_params
    if USE_AMP and device.type == "cuda" and has_complex_params:
        print("检测到模型包含复数参数，自动关闭 GradScaler，保留 autocast。")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=2e-6,
    )

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(LOG_ROOT / run_name))
    run_output_dir = RUN_OUTPUTS_ROOT / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    best_history_dir = BEST_HISTORY_ROOT / run_name
    best_history_dir.mkdir(parents=True, exist_ok=True)
    run_best_path = best_history_dir / "run_best.pt"
    run_final_path = best_history_dir / "run_final.pt"

    coords_base = {
        "x": train_ds.coord_tensors["x"].to(device),
        "y": train_ds.coord_tensors["y"].to(device),
        "z": train_ds.coord_tensors["z"].to(device),
    }

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    best_snapshot_count = 0
    bad_epochs = 0
    global_step = 0
    train_hist = []
    val_hist = []

    use_autocast = USE_AMP and device.type == "cuda"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_ds.refresh(epoch)

        train_total_sum = 0.0
        train_field_sum = 0.0
        train_s11_sum = 0.0
        train_a_peak_sum = 0.0
        train_t_zero_sum = 0.0
        train_passive_sum = 0.0
        train_peak_rank_sum = 0.0
        train_peak_pos_sum = 0.0
        train_curl_e_sum = 0.0
        train_curl_h_sum = 0.0
        train_div_sum = 0.0
        train_count = 0
        nonfinite_batches = 0

        num_train_batches = len(train_loader)
        print(f"Epoch {epoch:03d} started | train_batches={num_train_batches}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx == 1:
                print(f"Epoch {epoch:03d} first batch loaded.")

            flat = flatten_group_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            apply_physics = epoch >= PHYSICS_START_EPOCH and (global_step % PHYSICS_LOSS_INTERVAL == 0)

            with torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=get_amp_dtype() if use_autocast else None):
                pred_field, pred_s_raw = model(flat["x"], flat["lam_norm"])
                loss, stats = compute_total_loss(
                    pred_field=pred_field,
                    pred_s_raw=pred_s_raw,
                    target_field=flat["target"],
                    s11_target=flat["s11_target"],
                    a_target=flat["a_target"],
                    lambda_raw=flat["lambda_raw"],
                    peak_weight=flat["peak_weight"],
                    physics_mask=flat["physics_mask"],
                    scale=flat["scale"],
                    eps_ri=flat["eps"],
                    omega=flat["omega"],
                    coords=coords_base,
                    epoch=epoch,
                    train_mode=True,
                    apply_physics=apply_physics,
                    group_size=flat["group_size"],
                )

            if not torch.isfinite(loss):
                nonfinite_batches += 1
                if nonfinite_batches <= 8:
                    print(f"Epoch {epoch:03d} 检测到非有限 loss，跳过 batch {batch_idx}/{num_train_batches}")
                continue

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            bs = flat["batch_samples"]
            train_total_sum += stats["total"] * bs
            train_field_sum += stats["field"] * bs
            train_s11_sum += stats["s11"] * bs
            train_a_peak_sum += stats["a_peak"] * bs
            train_t_zero_sum += stats["t_zero"] * bs
            train_passive_sum += stats["passive"] * bs
            train_peak_rank_sum += stats["peak_rank"] * bs
            train_peak_pos_sum += stats["peak_pos"] * bs
            train_curl_e_sum += stats["curl_e"] * bs
            train_curl_h_sum += stats["curl_h"] * bs
            train_div_sum += stats["div"] * bs
            train_count += bs
            global_step += 1

            if (batch_idx % TRAIN_PROGRESS_EVERY == 0) or (batch_idx == num_train_batches):
                running_train = train_total_sum / max(train_count, 1)
                print(f"Epoch {epoch:03d} progress {batch_idx}/{num_train_batches} | running_train={running_train:.6e}")

        train_total = train_total_sum / max(train_count, 1)
        train_field = train_field_sum / max(train_count, 1)
        train_s11 = train_s11_sum / max(train_count, 1)
        train_a_peak = train_a_peak_sum / max(train_count, 1)
        train_t_zero = train_t_zero_sum / max(train_count, 1)
        train_passive = train_passive_sum / max(train_count, 1)
        train_peak_rank = train_peak_rank_sum / max(train_count, 1)
        train_peak_pos = train_peak_pos_sum / max(train_count, 1)
        train_curl_e = train_curl_e_sum / max(train_count, 1)
        train_curl_h = train_curl_h_sum / max(train_count, 1)
        train_div = train_div_sum / max(train_count, 1)

        do_val = (epoch % VAL_EVERY == 0) or (epoch == 1)
        if do_val:
            val_metrics = evaluate(model, val_loader, coords_base, device, epoch)
            val_total = val_metrics["val_total"]
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_total)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"[Scheduler] LR reduced: {old_lr:.3e} -> {new_lr:.3e}")
        else:
            val_metrics = {
                "val_total": float("nan"),
                "val_field": float("nan"),
                "val_s11": float("nan"),
                "val_a_peak": float("nan"),
                "val_t_zero": float("nan"),
                "val_passive": float("nan"),
                "val_peak_rank": float("nan"),
                "val_peak_pos": float("nan"),
                "val_curl_e": float("nan"),
                "val_curl_h": float("nan"),
                "val_div": float("nan"),
            }
            val_total = float("nan")

        train_hist.append(train_total)
        val_hist.append(val_total)

        writer.add_scalar("loss/train_total", train_total, epoch)
        writer.add_scalar("loss/train_field", train_field, epoch)
        writer.add_scalar("loss/train_s11", train_s11, epoch)
        writer.add_scalar("loss/train_a_peak", train_a_peak, epoch)
        writer.add_scalar("loss/train_t_zero", train_t_zero, epoch)
        writer.add_scalar("loss/train_passive", train_passive, epoch)
        writer.add_scalar("loss/train_peak_rank", train_peak_rank, epoch)
        writer.add_scalar("loss/train_peak_pos", train_peak_pos, epoch)
        writer.add_scalar("loss/train_curl_e", train_curl_e, epoch)
        writer.add_scalar("loss/train_curl_h", train_curl_h, epoch)
        writer.add_scalar("loss/train_div", train_div, epoch)
        writer.add_scalar("loss/physics_ramp", physics_ramp(epoch), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if do_val:
            writer.add_scalar("loss/val_total", val_total, epoch)
            writer.add_scalar("loss/val_field", val_metrics["val_field"], epoch)
            writer.add_scalar("loss/val_s11", val_metrics["val_s11"], epoch)
            writer.add_scalar("loss/val_a_peak", val_metrics["val_a_peak"], epoch)
            writer.add_scalar("loss/val_t_zero", val_metrics["val_t_zero"], epoch)
            writer.add_scalar("loss/val_passive", val_metrics["val_passive"], epoch)
            writer.add_scalar("loss/val_peak_rank", val_metrics["val_peak_rank"], epoch)
            writer.add_scalar("loss/val_peak_pos", val_metrics["val_peak_pos"], epoch)
            writer.add_scalar("loss/val_curl_e", val_metrics["val_curl_e"], epoch)
            writer.add_scalar("loss/val_curl_h", val_metrics["val_curl_h"], epoch)
            writer.add_scalar("loss/val_div", val_metrics["val_div"], epoch)

        if do_val and val_total < best_score:
            best_score = val_total
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            checkpoint_payload = build_checkpoint_payload(best_state, best_epoch, best_score, train_ds.lambda_vec)
            torch.save(checkpoint_payload, SAVE_PATH_BEST)
            torch.save(checkpoint_payload, run_best_path)
            best_snapshot_count += 1
            numbered_best_path = best_history_dir / f"best_{best_snapshot_count:03d}_epoch_{best_epoch:03d}_val_{best_score:.6e}.pt"
            torch.save(checkpoint_payload, numbered_best_path)
        elif do_val:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | train={train_total:.6e} | val={val_total:.6e} | "
            f"field={train_field:.4e} | s11={train_s11:.4e} | Apeak={train_a_peak:.4e} | "
            f"T0={train_t_zero:.4e} | rank={train_peak_rank:.4e} | pos={train_peak_pos:.4e} | "
            f"curlE={train_curl_e:.4e} | curlH={train_curl_h:.4e} | best_epoch={best_epoch}"
        )

        if do_val and epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    final_payload = build_checkpoint_payload(model.state_dict(), best_epoch, best_score, train_ds.lambda_vec)
    torch.save(final_payload, SAVE_PATH_FINAL)
    torch.save(final_payload, run_final_path)
    writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_score,
        "train_total_last": train_hist[-1] if train_hist else None,
        "val_total_last": val_hist[-1] if val_hist else None,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "batch_samples": BATCH_SAMPLES,
        "downsample": {"x": DOWN_X, "y": DOWN_Y, "z": DOWN_Z},
    }
    with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("训练完成：")
    print(f"  best模型:  {SAVE_PATH_BEST}")
    print(f"  final模型: {SAVE_PATH_FINAL}")
    print(f"  best历史目录: {best_history_dir}")
    print(f"  本轮best:   {run_best_path}")
    print(f"  本轮final:  {run_final_path}")
    print(f"  best_epoch = {best_epoch}, best_val_loss = {best_score:.6e}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
