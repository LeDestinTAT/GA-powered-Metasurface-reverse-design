from __future__ import annotations

import copy
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fullfield_dual_surrogate import (
    MODEL_FAMILY_CURVE_FIELD_V3,
    CurveFieldHybridModelV3,
    load_mat_auto,
    make_coord_maps,
    normalize_interval,
    project_to_passive,
    s_to_absorption_torch,
)
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
with redirect_stdout(io.StringIO()):
    from scripts.train.train_fno_fullfield_peakfocus import (
        HUBER_BETA,
        decode_complex_array,
        detect_top_two_peaks,
        maxwell_residual_loss,
        nearest_resize_2d,
        read_group_field_slices,
        read_sample_header,
        weighted_huber_loss,
        weighted_mse_loss,
    )


print("TensorBoard: run tensorboard --logdir logs/tensorboard/runs and open http://localhost:6006/")


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

EPOCHS = 120
VAL_EVERY = 5
MIN_EPOCHS = 30
PATIENCE = 12
TRAIN_PROGRESS_EVERY = 10

BATCH_SAMPLES = 16
VAL_BATCH_SAMPLES = 16
NUM_WORKERS = 0
PIN_MEMORY = True
SEQUENTIAL_BLOCK_SIZE = 128

DOWN_X = 3
DOWN_Y = 3
DOWN_Z = 3

MODES_X = 10
MODES_Y = 10
MODES_Z = 10
WIDTH = 48
DEPTH = 4
LAM_FF = 6
HEAD_HIDDEN = 192
FIELD_WIDTH = 32
FIELD_DEPTH = 3

LR = 1.6e-4
WEIGHT_DECAY = 5.0e-5
GRAD_CLIP = 1.0
USE_AMP = True
AMP_DTYPE = "bfloat16"

LAMBDA_CURVE_S11 = 0.70
LAMBDA_CURVE_A = 1.80
LAMBDA_MAIN_PEAK_POS = 2.40
LAMBDA_MAIN_PEAK_HEIGHT = 1.20
LAMBDA_SECONDARY_PEAK_HEIGHT = 0.50
LAMBDA_MAIN_PEAK_CLASS = 1.60
LAMBDA_PEAK_RANK = 0.04
LAMBDA_CURVE_TV = 0.015
LAMBDA_FIELD = 0.28
LAMBDA_PASSIVE = 0.03
LAMBDA_CURL_E = 0.004
LAMBDA_CURL_H = 0.004
LAMBDA_DIV = 0.0

FIELD_START_EPOCH = 8
PHYSICS_START_EPOCH = 24
PHYSICS_WARMUP_EPOCHS = 36
PHYSICS_LOSS_INTERVAL = 8
VAL_WITH_PHYSICS = False

FIELD_KEYS = ("Ex_vol", "Ey_vol", "Ez_vol", "Hx_vol", "Hy_vol", "Hz_vol")
FIELD_POINT_WEIGHTS = np.asarray([1.0, 0.75], dtype=np.float32)
FIELD_PHYSICS_MASK = np.asarray([1.0, 1.0], dtype=np.float32)

CURVE_BASE_WEIGHT = 0.08
CURVE_A_BOOST = 0.35
CURVE_MAIN_GAUSS = 4.00
CURVE_SECOND_GAUSS = 2.50
CURVE_MAIN_SIGMA = 2.0
CURVE_SECOND_SIGMA = 2.8
PEAK_WINDOW_RADIUS = 3
PEAK_SOFTMAX_TEMP = 12.0
OUTSIDE_PEAK_RADIUS = 2
PEAK_CLASS_SIGMA = 1.5
PEAK_BIN_EDGES_UM = (5.5, 8.5, 10.5)
SEGMENT_SELF_WEIGHT = 1.90
SEGMENT_ADJ_WEIGHT = 1.30
SEGMENT_FAR_WEIGHT = 0.90

HDF5_RETRIES = 3
HDF5_RETRY_SLEEP = 0.15
READ_FALLBACK_SAMPLES = 4
MAX_READ_WARNINGS = 20

BOTTOM_METAL_ZMAX = 100e-9
DIELECTRIC_ZMAX = 400e-9
TOP_PATTERN_ZMAX = 430e-9
AIR_EPS = complex(1.0, 0.0)

LOG_ROOT = TENSORBOARD_RUNS_DIR / "fno_curvefield_hybrid"
C0 = 299792458.0


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


def standardize_coord_1d(arr) -> np.ndarray:
    return np.asarray(arr).squeeze().astype(np.float32).reshape(-1)


def peak_um_to_bin_id(peak_um: float) -> int:
    if peak_um < PEAK_BIN_EDGES_UM[0]:
        return 0
    if peak_um < PEAK_BIN_EDGES_UM[1]:
        return 1
    if peak_um < PEAK_BIN_EDGES_UM[2]:
        return 2
    return 3


def wavelength_um_to_bin_ids(lambda_um: np.ndarray) -> np.ndarray:
    return np.asarray([peak_um_to_bin_id(float(v)) for v in lambda_um], dtype=np.int64)


def build_curve_weight(lambda_vec: np.ndarray, a_curve: np.ndarray, main_idx: int, secondary_idx: int) -> np.ndarray:
    idx = np.arange(len(a_curve), dtype=np.float32)
    a_norm = np.asarray(a_curve, dtype=np.float32)
    if float(np.max(a_norm)) > 1e-8:
        a_norm = a_norm / float(np.max(a_norm))
    main_gauss = np.exp(-0.5 * ((idx - float(main_idx)) / CURVE_MAIN_SIGMA) ** 2)
    second_gauss = np.exp(-0.5 * ((idx - float(secondary_idx)) / CURVE_SECOND_SIGMA) ** 2)
    weight = CURVE_BASE_WEIGHT + CURVE_A_BOOST * (a_norm ** 2) + CURVE_MAIN_GAUSS * main_gauss + CURVE_SECOND_GAUSS * second_gauss
    lambda_um = np.asarray(lambda_vec, dtype=np.float32) * 1e6
    point_bins = wavelength_um_to_bin_ids(lambda_um)
    main_bin = peak_um_to_bin_id(float(lambda_um[main_idx]))
    segment_multiplier = np.full_like(weight, SEGMENT_FAR_WEIGHT, dtype=np.float32)
    segment_multiplier[point_bins == main_bin] = SEGMENT_SELF_WEIGHT
    if main_bin - 1 >= 0:
        segment_multiplier[point_bins == (main_bin - 1)] = np.maximum(
            segment_multiplier[point_bins == (main_bin - 1)], SEGMENT_ADJ_WEIGHT
        )
    if main_bin + 1 <= 3:
        segment_multiplier[point_bins == (main_bin + 1)] = np.maximum(
            segment_multiplier[point_bins == (main_bin + 1)], SEGMENT_ADJ_WEIGHT
        )
    weight = weight * segment_multiplier
    return (weight / (float(np.mean(weight)) + 1e-8)).astype(np.float32)


def choose_background_index(a_curve: np.ndarray, main_idx: int, secondary_idx: int) -> int:
    n_lambda = len(a_curve)
    banned = set(range(max(0, main_idx - 3), min(n_lambda, main_idx + 4)))
    banned.update(range(max(0, secondary_idx - 3), min(n_lambda, secondary_idx + 4)))
    remaining = [idx for idx in range(n_lambda) if idx not in banned]
    if not remaining:
        return 0
    return int(remaining[int(np.argmin(np.asarray(a_curve, dtype=np.float32)[remaining]))])


def power_excess(pred_s):
    return F.relu(torch.sum(pred_s ** 2, dim=-1) - 1.0)


def gather_1d(values, indices):
    if values.ndim == 3:
        index = indices[:, None, None].expand(-1, 1, values.shape[-1])
        return values.gather(1, index).squeeze(1)
    return values.gather(1, indices[:, None]).squeeze(1)


def gather_windows(values, centers, radius):
    offsets = torch.arange(-radius, radius + 1, device=values.device)
    idx = (centers[:, None] + offsets[None, :]).clamp(0, values.shape[1] - 1)
    if values.ndim == 3:
        index = idx[:, :, None].expand(-1, -1, values.shape[-1])
        return values.gather(1, index), idx
    return values.gather(1, idx), idx


def curve_main_peak_position_loss(a_pred, lambda_raw, main_idx):
    main_win, main_ids = gather_windows(a_pred, main_idx, PEAK_WINDOW_RADIUS)
    main_lambda = lambda_raw.gather(1, main_ids)
    main_prob = torch.softmax(PEAK_SOFTMAX_TEMP * main_win, dim=1)
    main_center_pred = torch.sum(main_prob * main_lambda, dim=1)
    main_center_true = lambda_raw.gather(1, main_idx[:, None]).squeeze(1)
    main_span = (main_lambda[:, -1] - main_lambda[:, 0]).abs().clamp_min(1e-9)
    return torch.mean(torch.abs(main_center_pred - main_center_true) / main_span)


def curve_main_peak_height_loss(a_pred, a_true, main_idx):
    return F.smooth_l1_loss(gather_1d(a_pred, main_idx), gather_1d(a_true, main_idx), beta=0.03)


def curve_secondary_peak_height_loss(a_pred, a_true, secondary_idx):
    return F.smooth_l1_loss(gather_1d(a_pred, secondary_idx), gather_1d(a_true, secondary_idx), beta=0.03)


def main_peak_classification_loss(pred_logits, main_idx):
    if pred_logits is None:
        return None
    n_lambda = pred_logits.shape[1]
    grid = torch.arange(n_lambda, device=pred_logits.device, dtype=torch.float32)[None, :]
    center = main_idx.to(dtype=torch.float32)[:, None]
    target = torch.exp(-0.5 * ((grid - center) / PEAK_CLASS_SIGMA) ** 2)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = F.log_softmax(pred_logits, dim=1)
    normalizer = math.log(float(max(n_lambda, 2)))
    return torch.mean(torch.sum(-target * log_prob, dim=1)) / max(normalizer, 1e-6)


def curve_peak_rank_loss(a_pred, main_idx, secondary_idx, background_idx):
    bsz, n_lambda = a_pred.shape
    main_val = gather_1d(a_pred, main_idx)
    second_val = gather_1d(a_pred, secondary_idx)
    background_val = gather_1d(a_pred, background_idx)
    outside_mask = torch.ones((bsz, n_lambda), dtype=torch.bool, device=a_pred.device)
    offsets = torch.arange(-OUTSIDE_PEAK_RADIUS, OUTSIDE_PEAK_RADIUS + 1, device=a_pred.device)
    outside_mask.scatter_(1, (main_idx[:, None] + offsets[None, :]).clamp(0, n_lambda - 1), False)
    outside_mask.scatter_(1, (secondary_idx[:, None] + offsets[None, :]).clamp(0, n_lambda - 1), False)
    outside = a_pred.masked_fill(~outside_mask, -1e9)
    outside_max = outside.max(dim=1).values
    outside_max = torch.where(torch.isfinite(outside_max), outside_max, background_val)
    loss = (
        F.relu(0.05 - (main_val - second_val))
        + F.relu(0.08 - (main_val - background_val))
        + 0.55 * F.relu(0.04 - (second_val - background_val))
        + 0.80 * F.relu(0.02 - (second_val - outside_max))
        + 0.35 * F.relu(0.04 - (main_val - outside_max))
    )
    return loss.mean()


def curve_tv_loss(a_pred, curve_weight):
    diff = torch.abs(a_pred[:, 1:] - a_pred[:, :-1])
    local_weight = 0.5 * (curve_weight[:, 1:] + curve_weight[:, :-1])
    tv_weight = 1.0 / local_weight.clamp_min(0.2)
    tv_weight = tv_weight / tv_weight.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.mean(diff * tv_weight)


def physics_ramp(epoch):
    if epoch < PHYSICS_START_EPOCH:
        return 0.0
    return float(min(1.0, max(0.0, (epoch - PHYSICS_START_EPOCH + 1) / float(PHYSICS_WARMUP_EPOCHS))))


def build_coords_for_batch(base_coords, batch_size, device):
    return {
        "x": base_coords["x"].to(device).view(1, -1).expand(batch_size, -1),
        "y": base_coords["y"].to(device).view(1, -1).expand(batch_size, -1),
        "z": base_coords["z"].to(device).view(1, -1).expand(batch_size, -1),
    }


@dataclass
class SampleRecord:
    path: Path
    sample_id: int
    pattern_xy: np.ndarray
    lambda_vec: np.ndarray
    s11_curve: np.ndarray
    a_curve: np.ndarray
    curve_weight: np.ndarray
    main_idx: int
    secondary_idx: int
    background_idx: int
    main_peak_um: float
    peak_bin: int


class CurveFieldHybridDataset(Dataset):
    def __init__(self, sample_files: list[Path], meta: dict, train: bool):
        self.train = bool(train)
        self.dataset_name = "train" if self.train else "val"
        self.enable_field = True
        self.read_warning_count = 0
        xv_full = standardize_coord_1d(meta["xv"])
        yv_full = standardize_coord_1d(meta["yv"])
        zv_full = standardize_coord_1d(meta["zv"])
        self.xv = xv_full[::DOWN_X]
        self.yv = yv_full[::DOWN_Y]
        self.zv = zv_full[::DOWN_Z]
        self.full_shape = tuple(int(np.asarray(meta[k]).squeeze()) for k in ("Nx", "Ny", "Nz"))
        self.target_shape = (len(self.xv), len(self.yv), len(self.zv))
        self.nx, self.ny, self.nz = self.target_shape
        self.x_map, self.y_map, self.z_map = make_coord_maps(self.xv, self.yv, self.zv)
        self.bottom_mask_z = self.zv <= BOTTOM_METAL_ZMAX
        self.diel_mask_z = (self.zv > BOTTOM_METAL_ZMAX) & (self.zv <= DIELECTRIC_ZMAX)
        self.top_mask_z = (self.zv > DIELECTRIC_ZMAX) & (self.zv <= TOP_PATTERN_ZMAX)
        self.top_indices = np.where(self.top_mask_z)[0]
        self.coord_tensors = {axis: torch.from_numpy(values.astype(np.float32)) for axis, values in (("x", self.xv), ("y", self.yv), ("z", self.zv))}
        self.records = []
        for path in sample_files:
            try:
                sample_id = int(path.stem.split("_")[-1])
                lam, s11_curve, pattern_11 = read_sample_header(path)
                a_curve = np.clip(1.0 - np.abs(s11_curve) ** 2, 0.0, 1.0).astype(np.float32)
                main_idx, secondary_idx = detect_top_two_peaks(a_curve)
                background_idx = choose_background_index(a_curve, main_idx, secondary_idx)
                main_peak_um = float(lam[main_idx] * 1e6)
                peak_bin = peak_um_to_bin_id(main_peak_um)
                self.records.append(
                    SampleRecord(
                        path=Path(path),
                        sample_id=sample_id,
                        pattern_xy=nearest_resize_2d(pattern_11, self.nx, self.ny).astype(np.float32),
                        lambda_vec=lam.astype(np.float32),
                        s11_curve=s11_curve.astype(np.complex64),
                        a_curve=a_curve,
                        curve_weight=build_curve_weight(lam.astype(np.float32), a_curve, main_idx, secondary_idx),
                        main_idx=main_idx,
                        secondary_idx=secondary_idx,
                        background_idx=background_idx,
                        main_peak_um=main_peak_um,
                        peak_bin=peak_bin,
                    )
                )
            except Exception as exc:
                print(f"[{self.dataset_name}] skip header read failure: {Path(path).name} | {exc}")
        if not self.records:
            raise RuntimeError(f"{self.dataset_name} dataset is empty.")
        self.lambda_vec = self.records[0].lambda_vec.copy()
        self.lam_norm = normalize_interval(self.lambda_vec).astype(np.float32)
        self.bin_to_indices = {bin_id: [] for bin_id in range(4)}
        for idx, record in enumerate(self.records):
            self.bin_to_indices[int(record.peak_bin)].append(idx)
        self.bin_to_indices = {
            bin_id: np.asarray(sorted(indices), dtype=np.int64)
            for bin_id, indices in self.bin_to_indices.items()
            if len(indices) > 0
        }
        bin_summary = {
            f"bin_{bin_id}": int(len(indices))
            for bin_id, indices in sorted(self.bin_to_indices.items())
        }
        print(f"[{self.dataset_name}] peak-bin distribution = {bin_summary}")
        self.sample_order = np.arange(len(self.records), dtype=np.int64)
        self.refresh(epoch=0)

    def __len__(self):
        return len(self.sample_order)

    def set_field_loading(self, enabled: bool) -> None:
        self.enable_field = bool(enabled)

    def refresh(self, epoch: int) -> None:
        if self.train:
            rng = np.random.default_rng(SEED + 10007 * int(epoch))
            active_bins = sorted(self.bin_to_indices.keys())
            if not active_bins:
                self.sample_order = np.arange(len(self.records), dtype=np.int64)
                return

            bin_streams: dict[int, np.ndarray] = {}
            bin_ptrs: dict[int, int] = {}
            block_size = max(8, SEQUENTIAL_BLOCK_SIZE // max(len(active_bins), 1))

            def refill_bin(bin_id: int) -> None:
                indices = self.bin_to_indices[bin_id]
                blocks = [
                    indices[i : min(i + block_size, len(indices))]
                    for i in range(0, len(indices), block_size)
                ]
                order = rng.permutation(len(blocks))
                bin_streams[bin_id] = np.concatenate([blocks[int(i)] for i in order], axis=0)
                bin_ptrs[bin_id] = 0

            for bin_id in active_bins:
                refill_bin(bin_id)

            balanced_order = []
            while len(balanced_order) < len(self.records):
                for bin_id in rng.permutation(active_bins):
                    if len(balanced_order) >= len(self.records):
                        break
                    if bin_ptrs[bin_id] >= len(bin_streams[bin_id]):
                        refill_bin(bin_id)
                    balanced_order.append(int(bin_streams[bin_id][bin_ptrs[bin_id]]))
                    bin_ptrs[bin_id] += 1

            self.sample_order = np.asarray(balanced_order, dtype=np.int64)
        else:
            self.sample_order = np.arange(len(self.records), dtype=np.int64)

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
            eps[:, :, self.top_mask_z] = diel_eps
            for zi in self.top_indices:
                eps[:, :, zi][pattern_xy] = metal_eps
            metal_mask = metal_mask_base.copy()
            for zi in self.top_indices:
                metal_mask[:, :, zi][pattern_xy] = 1.0
            x_static[i] = np.stack([metal_mask, np.real(eps).astype(np.float32), np.imag(eps).astype(np.float32), self.x_map, self.y_map, self.z_map], axis=0)
            eps_ri[i, 0] = np.real(eps).astype(np.float32)
            eps_ri[i, 1] = np.imag(eps).astype(np.float32)
        return x_static, eps_ri

    def _build_item(self, order_index: int):
        record = self.records[int(self.sample_order[order_index])]
        if self.enable_field:
            field_indices = np.asarray([record.main_idx, record.secondary_idx], dtype=np.int64)
            lambda_vals = record.lambda_vec[field_indices]
            field_dict = None
            last_error = None
            for attempt in range(HDF5_RETRIES):
                try:
                    field_dict = read_group_field_slices(record.path, FIELD_KEYS, field_indices, self.full_shape)
                    break
                except (OSError, PermissionError) as exc:
                    last_error = exc
                    time.sleep(HDF5_RETRY_SLEEP * (attempt + 1))
            if field_dict is None:
                raise last_error if last_error is not None else RuntimeError(f"Failed to read fields: {record.path}")
            x_static, eps_ri = self._build_input_static(record, lambda_vals)
            point_count = len(field_indices)
            target = np.empty((point_count, 12, self.nx, self.ny, self.nz), dtype=np.float32)
            scale = np.empty((point_count,), dtype=np.float32)
            for li in range(point_count):
                channels = []
                for field_key in FIELD_KEYS:
                    arr = field_dict[field_key][li][::DOWN_X, ::DOWN_Y, ::DOWN_Z]
                    channels.extend([np.real(arr).astype(np.float32), np.imag(arr).astype(np.float32)])
                target_i = np.stack(channels, axis=0)
                scale_i = np.float32(np.sqrt(np.mean(target_i ** 2, dtype=np.float64) + 1e-12))
                target[li] = target_i / max(float(scale_i), 1e-6)
                scale[li] = scale_i
            field_x = torch.from_numpy(x_static)
            field_target = torch.from_numpy(target)
            field_eps = torch.from_numpy(eps_ri)
            field_scale = torch.from_numpy(scale.astype(np.float32))
            field_lam_norm = torch.from_numpy(self.lam_norm[field_indices][:, None].astype(np.float32))
            field_weight = torch.from_numpy(FIELD_POINT_WEIGHTS[:, None].astype(np.float32))
            physics_mask = torch.from_numpy(FIELD_PHYSICS_MASK.astype(np.float32))
            omega = torch.from_numpy((2.0 * math.pi * C0 / np.maximum(lambda_vals.astype(np.float64), 1e-12)).astype(np.float32))
        else:
            field_x = torch.empty((0, 6, self.nx, self.ny, self.nz), dtype=torch.float32)
            field_target = torch.empty((0, 12, self.nx, self.ny, self.nz), dtype=torch.float32)
            field_eps = torch.empty((0, 2, self.nx, self.ny, self.nz), dtype=torch.float32)
            field_scale = torch.empty((0,), dtype=torch.float32)
            field_lam_norm = torch.empty((0, 1), dtype=torch.float32)
            field_weight = torch.empty((0, 1), dtype=torch.float32)
            physics_mask = torch.empty((0,), dtype=torch.float32)
            omega = torch.empty((0,), dtype=torch.float32)
        return {
            "pattern_xy": torch.from_numpy(record.pattern_xy[None].astype(np.float32)),
            "curve_lam_norm": torch.from_numpy(self.lam_norm[:, None]),
            "curve_lambda_raw": torch.from_numpy(record.lambda_vec[:, None].astype(np.float32)),
            "s11_curve_target": torch.from_numpy(np.stack([np.real(record.s11_curve), np.imag(record.s11_curve)], axis=-1).astype(np.float32)),
            "a_curve_target": torch.from_numpy(record.a_curve[:, None].astype(np.float32)),
            "curve_weight": torch.from_numpy(record.curve_weight[:, None].astype(np.float32)),
            "main_idx": torch.tensor(record.main_idx, dtype=torch.long),
            "secondary_idx": torch.tensor(record.secondary_idx, dtype=torch.long),
            "background_idx": torch.tensor(record.background_idx, dtype=torch.long),
            "field_x": field_x,
            "field_target": field_target,
            "field_eps": field_eps,
            "field_scale": field_scale,
            "field_lam_norm": field_lam_norm,
            "field_weight": field_weight,
            "physics_mask": physics_mask,
            "omega": omega,
        }

    def __getitem__(self, idx):
        last_error = None
        base_idx = int(idx) % len(self.sample_order)
        for shift in range(READ_FALLBACK_SAMPLES + 1):
            try:
                return self._build_item((base_idx + shift) % len(self.sample_order))
            except (OSError, PermissionError) as exc:
                last_error = exc
                self.read_warning_count += 1
                if self.read_warning_count <= MAX_READ_WARNINGS:
                    print(f"[{self.dataset_name}] read failed, fallback to next sample | shift={shift} | error={exc}")
                time.sleep(HDF5_RETRY_SLEEP)
        raise last_error if last_error is not None else RuntimeError("Failed to read sample.")


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def forward_model(model, batch, include_field=True):
    feat_map, latent = model.encode_pattern(batch["pattern_xy"])
    pred_main_peak_logits = None
    if hasattr(model, "decode_curve_and_peak_from_latent"):
        pred_curve_s_raw, pred_main_peak_logits = model.decode_curve_and_peak_from_latent(latent, batch["curve_lam_norm"])
    elif hasattr(model, "predict_peak_properties_from_latent"):
        peak_props = model.predict_peak_properties_from_latent(latent)
        pred_curve_s_raw = model.decode_curve_from_latent(latent, batch["curve_lam_norm"], peak_props=peak_props)
    else:
        pred_curve_s_raw = model.decode_curve_from_latent(latent, batch["curve_lam_norm"])

    pred_field = None
    if include_field and batch["field_x"].shape[1] > 0:
        bsz, field_points = batch["field_x"].shape[:2]
        feat_rep = feat_map.repeat_interleave(field_points, dim=0)
        latent_rep = latent.repeat_interleave(field_points, dim=0)
        field_x_flat = batch["field_x"].reshape(bsz * field_points, *batch["field_x"].shape[2:])
        field_lam_flat = batch["field_lam_norm"].reshape(bsz * field_points, 1)
        pred_field, _ = model.decode_field_from_encoded(feat_rep, latent_rep, field_x_flat, field_lam_flat)
    return pred_curve_s_raw, pred_field, pred_main_peak_logits


def compute_total_loss(pred_curve_s_raw, pred_field, pred_main_peak_logits, batch, coords, epoch, train_mode, apply_physics):
    pred_curve_s = project_to_passive(pred_curve_s_raw)
    pred_curve_a = s_to_absorption_torch(pred_curve_s).unsqueeze(-1)
    a_target = batch["a_curve_target"]
    curve_weight = batch["curve_weight"]

    loss_curve_s11 = weighted_huber_loss(pred_curve_s[..., :2], batch["s11_curve_target"], curve_weight, delta=HUBER_BETA)
    loss_curve_a = weighted_mse_loss(pred_curve_a, a_target, curve_weight)
    loss_main_peak_pos = curve_main_peak_position_loss(
        pred_curve_a.squeeze(-1),
        batch["curve_lambda_raw"].squeeze(-1),
        batch["main_idx"],
    )
    loss_main_peak_height = curve_main_peak_height_loss(
        pred_curve_a.squeeze(-1),
        a_target.squeeze(-1),
        batch["main_idx"],
    )
    loss_secondary_peak_height = curve_secondary_peak_height_loss(
        pred_curve_a.squeeze(-1),
        a_target.squeeze(-1),
        batch["secondary_idx"],
    )
    loss_main_peak_class = main_peak_classification_loss(pred_main_peak_logits, batch["main_idx"])
    if loss_main_peak_class is None:
        loss_main_peak_class = pred_curve_s_raw.new_zeros(())
    loss_peak_rank = curve_peak_rank_loss(
        pred_curve_a.squeeze(-1),
        batch["main_idx"],
        batch["secondary_idx"],
        batch["background_idx"],
    )
    loss_curve_tv = curve_tv_loss(pred_curve_a.squeeze(-1), curve_weight.squeeze(-1))
    loss_passive = torch.mean(power_excess(pred_curve_s_raw) ** 2)

    if pred_field is not None and batch["field_target"].shape[1] > 0:
        field_target = batch["field_target"].reshape(-1, *batch["field_target"].shape[2:])
        field_weight = batch["field_weight"].reshape(-1, 1)
        loss_field = weighted_huber_loss(pred_field, field_target, field_weight, delta=HUBER_BETA)

        physics_mask = batch["physics_mask"].reshape(-1) > 0.5
        if apply_physics and torch.any(physics_mask):
            pred_physical = pred_field[physics_mask] * batch["field_scale"].reshape(-1)[physics_mask].view(-1, 1, 1, 1, 1)
            loss_curl_e, loss_curl_h, loss_div = maxwell_residual_loss(
                pred_physical,
                batch["field_eps"].reshape(-1, *batch["field_eps"].shape[2:])[physics_mask],
                batch["omega"].reshape(-1)[physics_mask],
                build_coords_for_batch(coords, int(torch.count_nonzero(physics_mask).item()), pred_curve_s_raw.device),
            )
        else:
            zero = pred_curve_s_raw.new_zeros(())
            loss_curl_e, loss_curl_h, loss_div = zero, zero, zero
    else:
        zero = pred_curve_s_raw.new_zeros(())
        loss_field, loss_curl_e, loss_curl_h, loss_div = zero, zero, zero, zero

    ramp = physics_ramp(epoch) if train_mode else 1.0
    total = (
        LAMBDA_CURVE_S11 * loss_curve_s11
        + LAMBDA_CURVE_A * loss_curve_a
        + LAMBDA_MAIN_PEAK_POS * loss_main_peak_pos
        + LAMBDA_MAIN_PEAK_HEIGHT * loss_main_peak_height
        + LAMBDA_SECONDARY_PEAK_HEIGHT * loss_secondary_peak_height
        + LAMBDA_MAIN_PEAK_CLASS * loss_main_peak_class
        + LAMBDA_PEAK_RANK * loss_peak_rank
        + LAMBDA_CURVE_TV * loss_curve_tv
        + LAMBDA_FIELD * loss_field
        + LAMBDA_PASSIVE * loss_passive
        + ramp * LAMBDA_CURL_E * loss_curl_e
        + ramp * LAMBDA_CURL_H * loss_curl_h
        + ramp * LAMBDA_DIV * loss_div
    )
    stats = {
        "curve_s11": float(loss_curve_s11.item()),
        "curve_a": float(loss_curve_a.item()),
        "main_peak_pos": float(loss_main_peak_pos.item()),
        "main_peak_height": float(loss_main_peak_height.item()),
        "secondary_peak_height": float(loss_secondary_peak_height.item()),
        "main_peak_class": float(loss_main_peak_class.item()),
        "peak_rank": float(loss_peak_rank.item()),
        "curve_tv": float(loss_curve_tv.item()),
        "field": float(loss_field.item()),
        "passive": float(loss_passive.item()),
        "curl_e": float(loss_curl_e.item()),
        "curl_h": float(loss_curl_h.item()),
        "div": float(loss_div.item()),
        "physics_ramp": float(ramp),
        "total": float(total.item()),
    }
    return total, stats


@torch.no_grad()
def evaluate(model, loader, coords_base, device, epoch):
    model.eval()
    sums = {
        k: 0.0
        for k in (
            "total",
            "curve_s11",
            "curve_a",
            "main_peak_pos",
            "main_peak_height",
            "secondary_peak_height",
            "main_peak_class",
            "peak_rank",
            "curve_tv",
            "field",
            "passive",
            "curl_e",
            "curl_h",
            "div",
        )
    }
    count = 0
    use_autocast = USE_AMP and device.type == "cuda"
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        include_field = batch["field_x"].shape[1] > 0
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=get_amp_dtype() if use_autocast else None):
            pred_curve_s_raw, pred_field, pred_main_peak_logits = forward_model(model, batch, include_field=include_field)
            _, stats = compute_total_loss(
                pred_curve_s_raw=pred_curve_s_raw,
                pred_field=pred_field,
                pred_main_peak_logits=pred_main_peak_logits,
                batch=batch,
                coords=coords_base,
                epoch=epoch,
                train_mode=False,
                apply_physics=VAL_WITH_PHYSICS,
            )
        bs = batch["pattern_xy"].shape[0]
        for key in sums:
            sums[key] += stats[key] * bs
        count += bs
    return {f"val_{key}": value / max(count, 1) for key, value in sums.items()}


def build_checkpoint_payload(state_dict, best_epoch_value, best_val_loss_value, lambda_vec):
    return {
        "state_dict": state_dict,
        "config": {
            "MODEL_FAMILY": MODEL_FAMILY_CURVE_FIELD_V3,
            "MODES_X": MODES_X,
            "MODES_Y": MODES_Y,
            "MODES_Z": MODES_Z,
            "WIDTH": WIDTH,
            "DEPTH": DEPTH,
            "FIELD_WIDTH": FIELD_WIDTH,
            "FIELD_DEPTH": FIELD_DEPTH,
            "LAM_FF": LAM_FF,
            "HEAD_HIDDEN": HEAD_HIDDEN,
            "DOWN_X": DOWN_X,
            "DOWN_Y": DOWN_Y,
            "DOWN_Z": DOWN_Z,
            "BATCH_SAMPLES": BATCH_SAMPLES,
            "VAL_BATCH_SAMPLES": VAL_BATCH_SAMPLES,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LAMBDA_CURVE_S11": LAMBDA_CURVE_S11,
            "LAMBDA_CURVE_A": LAMBDA_CURVE_A,
            "LAMBDA_MAIN_PEAK_POS": LAMBDA_MAIN_PEAK_POS,
            "LAMBDA_MAIN_PEAK_HEIGHT": LAMBDA_MAIN_PEAK_HEIGHT,
            "LAMBDA_SECONDARY_PEAK_HEIGHT": LAMBDA_SECONDARY_PEAK_HEIGHT,
            "LAMBDA_MAIN_PEAK_CLASS": LAMBDA_MAIN_PEAK_CLASS,
            "LAMBDA_PEAK_RANK": LAMBDA_PEAK_RANK,
            "LAMBDA_CURVE_TV": LAMBDA_CURVE_TV,
            "LAMBDA_FIELD": LAMBDA_FIELD,
            "LAMBDA_PASSIVE": LAMBDA_PASSIVE,
            "LAMBDA_CURL_E": LAMBDA_CURL_E,
            "LAMBDA_CURL_H": LAMBDA_CURL_H,
            "LAMBDA_DIV": LAMBDA_DIV,
            "FIELD_START_EPOCH": FIELD_START_EPOCH,
            "PHYSICS_START_EPOCH": PHYSICS_START_EPOCH,
            "PHYSICS_WARMUP_EPOCHS": PHYSICS_WARMUP_EPOCHS,
            "PHYSICS_LOSS_INTERVAL": PHYSICS_LOSS_INTERVAL,
            "FIELD_POINT_COUNT": 2,
            "FIELD_POINT_WEIGHTS": FIELD_POINT_WEIGHTS.tolist(),
            "PEAK_BIN_EDGES_UM": list(PEAK_BIN_EDGES_UM),
            "TRAIN_SAMPLER": "balanced_peak_bins_round_robin",
            "CURVE_SEGMENT_WEIGHTS": {
                "self": SEGMENT_SELF_WEIGHT,
                "adjacent": SEGMENT_ADJ_WEIGHT,
                "far": SEGMENT_FAR_WEIGHT,
            },
            "T_ZERO_OVERRIDE": True,
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
        raise RuntimeError(f"No sample_*.mat files found under {DATA_DIR}")

    if ignored_sample_files:
        print("Ignored non-standard sample files:", len(ignored_sample_files))
        print("Example ignored file:", ignored_sample_files[0].name)

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
        raise RuntimeError("Validation set is empty. Check TRAIN_RATIO / VAL_SAMPLE_LIMIT.")

    train_ds = CurveFieldHybridDataset(train_files, meta, train=True)
    val_ds = CurveFieldHybridDataset(val_files, meta, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SAMPLES, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SAMPLES, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available())

    print("Train sample files =", len(train_files))
    print("Val sample files =", len(val_files))
    print("Train batches =", len(train_loader))
    print("Val batches =", len(val_loader))
    print("Curve length =", len(train_ds.lambda_vec))
    print("Field wavelengths per sample = 2 (main peak / secondary peak)")
    print("Spatial shape =", train_ds.target_shape)
    print("Disk strategy = sequential sample reads with block-level shuffling.")
    print(f"Field supervision starts at epoch {FIELD_START_EPOCH}, physics starts at epoch {PHYSICS_START_EPOCH}.")
    print(f"Best-model tracking starts at epoch {FIELD_START_EPOCH}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device.type)
    if device.type == "cuda":
        print("AMP dtype =", AMP_DTYPE)

    model = CurveFieldHybridModelV3(
        modes_x=MODES_X,
        modes_y=MODES_Y,
        modes_z=MODES_Z,
        width=WIDTH,
        depth=DEPTH,
        lam_ff=LAM_FF,
        head_hidden=HEAD_HIDDEN,
        field_width=FIELD_WIDTH,
        field_depth=FIELD_DEPTH,
    ).to(device)

    has_complex_params = any(torch.is_complex(p) for p in model.parameters())
    scaler_enabled = USE_AMP and device.type == "cuda" and not has_complex_params
    if USE_AMP and device.type == "cuda" and has_complex_params:
        print("Model contains complex parameters, so GradScaler is disabled while autocast remains enabled.")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
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
    coords_base = {axis: tensor.to(device) for axis, tensor in train_ds.coord_tensors.items()}

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    best_snapshot_count = 0
    bad_epochs = 0
    global_step = 0
    train_hist = []
    val_hist = []
    use_autocast = USE_AMP and device.type == "cuda"
    metric_keys = (
        "total",
        "curve_s11",
        "curve_a",
        "main_peak_pos",
        "main_peak_height",
        "secondary_peak_height",
        "main_peak_class",
        "peak_rank",
        "curve_tv",
        "field",
        "passive",
        "curl_e",
        "curl_h",
        "div",
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_ds.refresh(epoch)
        train_ds.set_field_loading(epoch >= FIELD_START_EPOCH)
        val_ds.set_field_loading(epoch >= FIELD_START_EPOCH)
        sums = {k: 0.0 for k in metric_keys}
        count = 0
        nonfinite_batches = 0
        num_train_batches = len(train_loader)
        print(f"Epoch {epoch:03d} started | train_batches={num_train_batches} | field={'on' if epoch >= FIELD_START_EPOCH else 'off'}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx == 1:
                print(f"Epoch {epoch:03d} first batch loaded.")
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            apply_physics = epoch >= PHYSICS_START_EPOCH and (global_step % PHYSICS_LOSS_INTERVAL == 0)
            include_field = epoch >= FIELD_START_EPOCH and batch["field_x"].shape[1] > 0
            with torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=get_amp_dtype() if use_autocast else None):
                pred_curve_s_raw, pred_field, pred_main_peak_logits = forward_model(model, batch, include_field=include_field)
                loss, stats = compute_total_loss(
                    pred_curve_s_raw,
                    pred_field,
                    pred_main_peak_logits,
                    batch,
                    coords_base,
                    epoch,
                    True,
                    apply_physics,
                )

            if not torch.isfinite(loss):
                nonfinite_batches += 1
                if nonfinite_batches <= 8:
                    print(f"Epoch {epoch:03d} found non-finite loss, skip batch {batch_idx}/{num_train_batches}")
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

            bs = batch["pattern_xy"].shape[0]
            for key in metric_keys:
                sums[key] += stats[key] * bs
            count += bs
            global_step += 1
            if (batch_idx % TRAIN_PROGRESS_EVERY == 0) or (batch_idx == num_train_batches):
                print(f"Epoch {epoch:03d} progress {batch_idx}/{num_train_batches} | running_train={sums['total'] / max(count, 1):.6e}")

        train_metrics = {f"train_{key}": sums[key] / max(count, 1) for key in metric_keys}
        train_total = train_metrics["train_total"]

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
            val_metrics = {f"val_{key}": float('nan') for key in metric_keys}
            val_total = float("nan")

        train_hist.append(train_total)
        val_hist.append(val_total)
        for key, value in train_metrics.items():
            writer.add_scalar(f"loss/{key}", value, epoch)
        writer.add_scalar("loss/physics_ramp", physics_ramp(epoch), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if do_val:
            for key, value in val_metrics.items():
                writer.add_scalar(f"loss/{key}", value, epoch)

        if do_val and epoch >= FIELD_START_EPOCH and val_total < best_score:
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
        elif do_val and epoch >= FIELD_START_EPOCH:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | train={train_total:.6e} | val={val_total:.6e} | "
            f"curveS={train_metrics['train_curve_s11']:.4e} | curveA={train_metrics['train_curve_a']:.4e} | "
            f"mainPos={train_metrics['train_main_peak_pos']:.4e} | mainH={train_metrics['train_main_peak_height']:.4e} | "
            f"secondH={train_metrics['train_secondary_peak_height']:.4e} | mainCls={train_metrics['train_main_peak_class']:.4e} | "
            f"rank={train_metrics['train_peak_rank']:.4e} | field={train_metrics['train_field']:.4e} | "
            f"curlE={train_metrics['train_curl_e']:.4e} | curlH={train_metrics['train_curl_h']:.4e} | best_epoch={best_epoch}"
        )

        if do_val and epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    final_payload = build_checkpoint_payload(model.state_dict(), best_epoch, best_score, train_ds.lambda_vec)
    torch.save(final_payload, SAVE_PATH_FINAL)
    torch.save(final_payload, run_final_path)
    writer.close()

    summary = {
        "run_name": run_name,
        "model_family": MODEL_FAMILY_CURVE_FIELD_V3,
        "best_epoch": best_epoch,
        "best_val_loss": best_score,
        "train_total_last": train_hist[-1] if train_hist else None,
        "val_total_last": val_hist[-1] if val_hist else None,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "batch_samples": BATCH_SAMPLES,
        "field_points_per_sample": 2,
        "train_sampler": "balanced_peak_bins_round_robin",
        "peak_bin_edges_um": list(PEAK_BIN_EDGES_UM),
        "downsample": {"x": DOWN_X, "y": DOWN_Y, "z": DOWN_Z},
    }
    with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"  best model:  {SAVE_PATH_BEST}")
    print(f"  final model: {SAVE_PATH_FINAL}")
    print(f"  best history dir: {best_history_dir}")
    print(f"  this run best:    {run_best_path}")
    print(f"  this run final:   {run_final_path}")
    print(f"  best_epoch = {best_epoch}, best_val_loss = {best_score:.6e}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
