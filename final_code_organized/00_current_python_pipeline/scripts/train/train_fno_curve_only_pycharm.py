from __future__ import annotations

import copy
import io
import json
import math
import multiprocessing as mp
import random
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fullfield_dual_surrogate import (
    LambdaFourierFeatures,
    PatternFNO2dEncoder,
    ResidualConv1dBlock,
)
from src.project_paths import (
    BEST_MODEL_HISTORY_ROOT,
    CURVE_DATASET_CACHE_PATH,
    FIELD_DATA_DIR,
    MODELS_CURRENT_DIR,
    TENSORBOARD_RUNS_DIR,
    TRAIN_RUN_OUTPUTS_DIR,
    ensure_standard_dirs,
)

with redirect_stdout(io.StringIO()):
    from scripts.train.train_fno_fullfield_peakfocus import (
        HUBER_BETA,
        detect_top_two_peaks,
        read_sample_header,
        weighted_huber_loss,
        weighted_mse_loss,
    )


print("TensorBoard: run tensorboard --logdir logs/tensorboard/runs and open http://localhost:6006/")


DATA_DIR = FIELD_DATA_DIR
SAVE_PATH_FINAL = MODELS_CURRENT_DIR / "fno_curve_only_final.pt"
SAVE_PATH_BEST = MODELS_CURRENT_DIR / "fno_curve_only_best.pt"
BEST_HISTORY_ROOT = BEST_MODEL_HISTORY_ROOT
RUN_OUTPUTS_ROOT = TRAIN_RUN_OUTPUTS_DIR / "curve_only"

SEED = 42
TRAIN_RATIO = 0.85
TRAIN_SAMPLE_LIMIT = None
VAL_SAMPLE_LIMIT = None

EPOCHS = 160
VAL_EVERY = 5
MIN_EPOCHS = 40
PATIENCE = 16
TRAIN_PROGRESS_EVERY = 10

BATCH_SAMPLES = 64
VAL_BATCH_SAMPLES = 128
NUM_WORKERS = 0
PIN_MEMORY = True
SEQUENTIAL_BLOCK_SIZE = 192

MODES_X = 8
MODES_Y = 8
WIDTH = 48
DEPTH = 4
LAM_FF = 6
HEAD_HIDDEN = 192
CURVE_BLOCKS = 4

LR = 2.0e-4
WEIGHT_DECAY = 3.0e-5
GRAD_CLIP = 1.0
USE_AMP = True
AMP_DTYPE = "bfloat16"

LAMBDA_S11 = 0.70
LAMBDA_A = 1.90
LAMBDA_MAIN_PEAK_POS = 2.20
LAMBDA_MAIN_PEAK_HEIGHT = 1.35
LAMBDA_SECONDARY_PEAK_HEIGHT = 0.55
LAMBDA_MAIN_PEAK_CLASS = 1.40
LAMBDA_PEAK_RANK = 0.04
LAMBDA_CURVE_TV = 0.015
LAMBDA_PASSIVE = 0.03

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

LOG_ROOT = TENSORBOARD_RUNS_DIR / "fno_curve_only"


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
RUN_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def normalize_interval(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return 2.0 * (v - vmin) / (vmax - vmin + 1e-12) - 1.0


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


def nearest_resize_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = arr.shape
    row_idx = np.clip(np.round(np.linspace(0, src_h - 1, out_h)).astype(int), 0, src_h - 1)
    col_idx = np.clip(np.round(np.linspace(0, src_w - 1, out_w)).astype(int), 0, src_w - 1)
    return arr[np.ix_(row_idx, col_idx)]


def project_s11_to_passive(pred_s11, eps=1e-12):
    power = pred_s11[..., 0] ** 2 + pred_s11[..., 1] ** 2
    scale = torch.where(power > 1.0, torch.rsqrt(power + eps), torch.ones_like(power))
    out = pred_s11.clone()
    out[..., 0] = pred_s11[..., 0] * scale
    out[..., 1] = pred_s11[..., 1] * scale
    return out


def s11_to_absorption_torch(pred_s11):
    return torch.clamp(1.0 - torch.sum(pred_s11 ** 2, dim=-1), min=0.0, max=1.0)


def power_excess(pred_s11):
    return F.relu(torch.sum(pred_s11 ** 2, dim=-1) - 1.0)


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


def load_curve_cache(path: Path):
    path = Path(path)
    if not path.is_file():
        return None
    cache = np.load(path, allow_pickle=False)
    return {key: cache[key] for key in cache.files}


class CurveOnlyDataset(Dataset):
    def __init__(self, sample_files: list[Path] | None, train: bool, cache_data: dict | None = None, selected_indices: np.ndarray | None = None):
        self.train = bool(train)
        self.dataset_name = "train" if self.train else "val"
        self.records = []
        if cache_data is not None:
            sel = np.arange(len(cache_data["sample_id"]), dtype=np.int64) if selected_indices is None else np.asarray(selected_indices, dtype=np.int64)
            lambda_vec = np.asarray(cache_data["lambda_vec"], dtype=np.float32).reshape(-1)
            for idx in sel:
                self.records.append(
                    SampleRecord(
                        path=Path(str(cache_data["sample_name"][idx])),
                        sample_id=int(cache_data["sample_id"][idx]),
                        pattern_xy=np.asarray(cache_data["pattern_11"][idx], dtype=np.float32).reshape(11, 11),
                        lambda_vec=lambda_vec.copy(),
                        s11_curve=(np.asarray(cache_data["s11_real"][idx], dtype=np.float32) + 1j * np.asarray(cache_data["s11_imag"][idx], dtype=np.float32)).astype(np.complex64),
                        a_curve=np.asarray(cache_data["absorption"][idx], dtype=np.float32).copy(),
                        curve_weight=np.asarray(cache_data["curve_weight"][idx], dtype=np.float32).copy(),
                        main_idx=int(cache_data["main_idx"][idx]),
                        secondary_idx=int(cache_data["secondary_idx"][idx]),
                        background_idx=int(cache_data["background_idx"][idx]),
                        main_peak_um=float(cache_data["main_peak_um"][idx]),
                        peak_bin=int(cache_data["peak_bin"][idx]),
                    )
                )
        else:
            if sample_files is None:
                raise ValueError("sample_files cannot be None when cache_data is not provided.")
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
                            pattern_xy=nearest_resize_2d(pattern_11.astype(np.float32), 11, 11).astype(np.float32),
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

    def __getitem__(self, idx):
        record = self.records[int(self.sample_order[int(idx)])]
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
        }


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


class CurveOnlySpectrumModel(nn.Module):
    def __init__(
        self,
        modes_x=8,
        modes_y=8,
        width=48,
        depth=4,
        lam_ff=6,
        head_hidden=192,
        curve_blocks=4,
    ):
        super().__init__()
        self.pattern_encoder = PatternFNO2dEncoder(modes_x=modes_x, modes_y=modes_y, width=width, depth=depth)
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)
        self.curve_in = nn.Sequential(
            nn.Linear(width + 2 * lam_ff, head_hidden),
            nn.GELU(),
        )
        self.curve_local_head = nn.Sequential(
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2),
        )
        self.curve_blocks = nn.ModuleList(
            [ResidualConv1dBlock(head_hidden, kernel_size=5) for _ in range(curve_blocks)]
        )
        self.curve_out = nn.Conv1d(head_hidden, 2, kernel_size=1)
        peak_hidden = max(head_hidden // 2, 16)
        self.main_peak_head = nn.Sequential(
            nn.Conv1d(head_hidden, peak_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(peak_hidden, 1, kernel_size=1),
        )

    def forward(self, pattern_xy, lam_norm):
        _, latent = self.pattern_encoder(pattern_xy)
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(-1)
        b, l, _ = lam_norm.shape
        lam_embed = self.lam_embed(lam_norm.reshape(b * l, 1)).reshape(b, l, -1)
        latent_rep = latent.unsqueeze(1).expand(b, l, latent.shape[-1])
        h_local = self.curve_in(torch.cat([latent_rep, lam_embed], dim=-1).reshape(b * l, -1)).reshape(b, l, -1)
        local_pred = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        h_seq = h_local.transpose(1, 2)
        for block in self.curve_blocks:
            h_seq = block(h_seq)
        seq_pred = self.curve_out(h_seq).transpose(1, 2)
        peak_logits = self.main_peak_head(h_seq).squeeze(1)
        return local_pred + seq_pred, peak_logits


def compute_total_loss(pred_s11_raw, pred_main_peak_logits, batch):
    pred_s11 = project_s11_to_passive(pred_s11_raw)
    pred_a = s11_to_absorption_torch(pred_s11).unsqueeze(-1)
    a_target = batch["a_curve_target"]
    curve_weight = batch["curve_weight"]

    loss_s11 = weighted_huber_loss(pred_s11, batch["s11_curve_target"], curve_weight, delta=HUBER_BETA)
    loss_a = weighted_mse_loss(pred_a, a_target, curve_weight)
    loss_main_peak_pos = curve_main_peak_position_loss(
        pred_a.squeeze(-1),
        batch["curve_lambda_raw"].squeeze(-1),
        batch["main_idx"],
    )
    loss_main_peak_height = curve_main_peak_height_loss(
        pred_a.squeeze(-1),
        a_target.squeeze(-1),
        batch["main_idx"],
    )
    loss_secondary_peak_height = curve_secondary_peak_height_loss(
        pred_a.squeeze(-1),
        a_target.squeeze(-1),
        batch["secondary_idx"],
    )
    loss_main_peak_class = main_peak_classification_loss(pred_main_peak_logits, batch["main_idx"])
    if loss_main_peak_class is None:
        loss_main_peak_class = pred_s11_raw.new_zeros(())
    loss_peak_rank = curve_peak_rank_loss(
        pred_a.squeeze(-1),
        batch["main_idx"],
        batch["secondary_idx"],
        batch["background_idx"],
    )
    loss_curve_tv = curve_tv_loss(pred_a.squeeze(-1), curve_weight.squeeze(-1))
    loss_passive = torch.mean(power_excess(pred_s11_raw) ** 2)
    total = (
        LAMBDA_S11 * loss_s11
        + LAMBDA_A * loss_a
        + LAMBDA_MAIN_PEAK_POS * loss_main_peak_pos
        + LAMBDA_MAIN_PEAK_HEIGHT * loss_main_peak_height
        + LAMBDA_SECONDARY_PEAK_HEIGHT * loss_secondary_peak_height
        + LAMBDA_MAIN_PEAK_CLASS * loss_main_peak_class
        + LAMBDA_PEAK_RANK * loss_peak_rank
        + LAMBDA_CURVE_TV * loss_curve_tv
        + LAMBDA_PASSIVE * loss_passive
    )
    stats = {
        "s11": float(loss_s11.item()),
        "a": float(loss_a.item()),
        "main_peak_pos": float(loss_main_peak_pos.item()),
        "main_peak_height": float(loss_main_peak_height.item()),
        "secondary_peak_height": float(loss_secondary_peak_height.item()),
        "main_peak_class": float(loss_main_peak_class.item()),
        "peak_rank": float(loss_peak_rank.item()),
        "curve_tv": float(loss_curve_tv.item()),
        "passive": float(loss_passive.item()),
        "total": float(total.item()),
    }
    return total, stats


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {k: 0.0 for k in ("total", "s11", "a", "main_peak_pos", "main_peak_height", "secondary_peak_height", "main_peak_class", "peak_rank", "curve_tv", "passive")}
    count = 0
    use_autocast = USE_AMP and device.type == "cuda"
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=get_amp_dtype() if use_autocast else None):
            pred_s11_raw, pred_main_peak_logits = model(batch["pattern_xy"], batch["curve_lam_norm"])
            _, stats = compute_total_loss(pred_s11_raw, pred_main_peak_logits, batch)
        bs = batch["pattern_xy"].shape[0]
        for key in sums:
            sums[key] += stats[key] * bs
        count += bs
    return {f"val_{key}": value / max(count, 1) for key, value in sums.items()}


def build_checkpoint_payload(state_dict, best_epoch_value, best_val_loss_value, lambda_vec):
    return {
        "state_dict": state_dict,
        "config": {
            "MODEL_FAMILY": "curve_only_s11_v1",
            "MODES_X": MODES_X,
            "MODES_Y": MODES_Y,
            "WIDTH": WIDTH,
            "DEPTH": DEPTH,
            "LAM_FF": LAM_FF,
            "HEAD_HIDDEN": HEAD_HIDDEN,
            "CURVE_BLOCKS": CURVE_BLOCKS,
            "BATCH_SAMPLES": BATCH_SAMPLES,
            "VAL_BATCH_SAMPLES": VAL_BATCH_SAMPLES,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LAMBDA_S11": LAMBDA_S11,
            "LAMBDA_A": LAMBDA_A,
            "LAMBDA_MAIN_PEAK_POS": LAMBDA_MAIN_PEAK_POS,
            "LAMBDA_MAIN_PEAK_HEIGHT": LAMBDA_MAIN_PEAK_HEIGHT,
            "LAMBDA_SECONDARY_PEAK_HEIGHT": LAMBDA_SECONDARY_PEAK_HEIGHT,
            "LAMBDA_MAIN_PEAK_CLASS": LAMBDA_MAIN_PEAK_CLASS,
            "LAMBDA_PEAK_RANK": LAMBDA_PEAK_RANK,
            "LAMBDA_CURVE_TV": LAMBDA_CURVE_TV,
            "LAMBDA_PASSIVE": LAMBDA_PASSIVE,
            "T_ZERO_OVERRIDE": True,
            "SEED": SEED,
        },
        "best_epoch": best_epoch_value,
        "best_val_loss": best_val_loss_value,
        "lambda_vec": torch.tensor(lambda_vec, dtype=torch.float32),
    }


def main():
    cache_data = load_curve_cache(CURVE_DATASET_CACHE_PATH)
    using_cache = cache_data is not None

    if using_cache:
        n_total = int(len(cache_data["sample_id"]))
        perm = np.random.default_rng(SEED).permutation(n_total)
        n_train = int(TRAIN_RATIO * n_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        if TRAIN_SAMPLE_LIMIT is not None:
            train_idx = train_idx[: int(TRAIN_SAMPLE_LIMIT)]
        if VAL_SAMPLE_LIMIT is not None:
            val_idx = val_idx[: int(VAL_SAMPLE_LIMIT)]
        if len(val_idx) == 0:
            raise RuntimeError("Validation set is empty. Check TRAIN_RATIO / VAL_SAMPLE_LIMIT.")
        train_ds = CurveOnlyDataset(None, train=True, cache_data=cache_data, selected_indices=train_idx)
        val_ds = CurveOnlyDataset(None, train=False, cache_data=cache_data, selected_indices=val_idx)
        train_count = len(train_idx)
        val_count = len(val_idx)
    else:
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
        train_ds = CurveOnlyDataset(train_files, train=True)
        val_ds = CurveOnlyDataset(val_files, train=False)
        train_count = len(train_files)
        val_count = len(val_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SAMPLES, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SAMPLES, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available())

    print("Curve cache source =", CURVE_DATASET_CACHE_PATH if using_cache else "raw sample headers")
    print("Train sample files =", train_count)
    print("Val sample files =", val_count)
    print("Train batches =", len(train_loader))
    print("Val batches =", len(val_loader))
    print("Curve length =", len(train_ds.lambda_vec))
    print("Training mode = curve only (no field data, no Maxwell residual).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device.type)
    if device.type == "cuda":
        print("AMP dtype =", AMP_DTYPE)

    model = CurveOnlySpectrumModel(
        modes_x=MODES_X,
        modes_y=MODES_Y,
        width=WIDTH,
        depth=DEPTH,
        lam_ff=LAM_FF,
        head_hidden=HEAD_HIDDEN,
        curve_blocks=CURVE_BLOCKS,
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

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    best_snapshot_count = 0
    bad_epochs = 0
    train_hist = []
    val_hist = []
    use_autocast = USE_AMP and device.type == "cuda"
    metric_keys = ("total", "s11", "a", "main_peak_pos", "main_peak_height", "secondary_peak_height", "main_peak_class", "peak_rank", "curve_tv", "passive")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_ds.refresh(epoch)
        sums = {k: 0.0 for k in metric_keys}
        count = 0
        nonfinite_batches = 0
        num_train_batches = len(train_loader)
        print(f"Epoch {epoch:03d} started | train_batches={num_train_batches}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx == 1:
                print(f"Epoch {epoch:03d} first batch loaded.")
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=get_amp_dtype() if use_autocast else None):
                pred_s11_raw, pred_main_peak_logits = model(batch["pattern_xy"], batch["curve_lam_norm"])
                loss, stats = compute_total_loss(pred_s11_raw, pred_main_peak_logits, batch)

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
            if (batch_idx % TRAIN_PROGRESS_EVERY == 0) or (batch_idx == num_train_batches):
                print(f"Epoch {epoch:03d} progress {batch_idx}/{num_train_batches} | running_train={sums['total'] / max(count, 1):.6e}")

        train_metrics = {f"train_{key}": sums[key] / max(count, 1) for key in metric_keys}
        train_total = train_metrics["train_total"]

        do_val = (epoch % VAL_EVERY == 0) or (epoch == 1)
        if do_val:
            val_metrics = evaluate(model, val_loader, device)
            val_total = val_metrics["val_total"]
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_total)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"[Scheduler] LR reduced: {old_lr:.3e} -> {new_lr:.3e}")
        else:
            val_metrics = {f"val_{key}": float("nan") for key in metric_keys}
            val_total = float("nan")

        train_hist.append(train_total)
        val_hist.append(val_total)
        for key, value in train_metrics.items():
            writer.add_scalar(f"loss/{key}", value, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if do_val:
            for key, value in val_metrics.items():
                writer.add_scalar(f"loss/{key}", value, epoch)

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
            f"S11={train_metrics['train_s11']:.4e} | A={train_metrics['train_a']:.4e} | "
            f"mainPos={train_metrics['train_main_peak_pos']:.4e} | mainH={train_metrics['train_main_peak_height']:.4e} | "
            f"secondH={train_metrics['train_secondary_peak_height']:.4e} | mainCls={train_metrics['train_main_peak_class']:.4e} | "
            f"rank={train_metrics['train_peak_rank']:.4e} | passive={train_metrics['train_passive']:.4e} | best_epoch={best_epoch}"
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
        "model_family": "curve_only_s11_v1",
        "best_epoch": best_epoch,
        "best_val_loss": best_score,
        "train_total_last": train_hist[-1] if train_hist else None,
        "val_total_last": val_hist[-1] if val_hist else None,
        "train_samples": train_count,
        "val_samples": val_count,
        "batch_samples": BATCH_SAMPLES,
        "train_sampler": "balanced_peak_bins_round_robin",
        "peak_bin_edges_um": list(PEAK_BIN_EDGES_UM),
        "curve_cache_source": str(CURVE_DATASET_CACHE_PATH) if using_cache else None,
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
