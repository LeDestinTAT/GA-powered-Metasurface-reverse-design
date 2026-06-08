from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

from src.material_dispersion import air_eps_from_lambda_m, au_eps_from_lambda_m, sio2_eps_from_lambda_m


MODEL_FAMILY_LEGACY = "legacy_fno3d_conditional"
MODEL_FAMILY_CURVE_FIELD = "curve_field_hybrid_v1"
MODEL_FAMILY_CURVE_FIELD_V2 = "curve_field_hybrid_v2"
MODEL_FAMILY_CURVE_FIELD_V3 = "curve_field_hybrid_v3"
MODEL_FAMILY_TRY2_TRANSFER = "try2_curve_field_transfer_v1"


def load_mat_auto(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    is_hdf5 = False
    if h5py is not None:
        try:
            is_hdf5 = h5py.is_hdf5(str(path))
        except Exception:
            is_hdf5 = False
    else:
        with path.open("rb") as f:
            is_hdf5 = f.read(8) == b"\x89HDF\r\n\x1a\n"

    if is_hdf5:
        if h5py is None:
            raise RuntimeError("需要 h5py 读取 MATLAB v7.3 文件")
        out = {}
        with h5py.File(str(path), "r") as f:
            for key in f.keys():
                out[key] = np.asarray(f[key][()])
        return out

    if loadmat is None:
        raise RuntimeError("需要 scipy.io.loadmat 读取非 v7.3 文件")
    out = loadmat(str(path))
    return {k: v for k, v in out.items() if not k.startswith("__")}


def standardize_coord_1d(arr) -> np.ndarray:
    arr = np.asarray(arr).squeeze().astype(np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def normalize_interval(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return 2.0 * (v - vmin) / (vmax - vmin + 1e-12) - 1.0


def make_coord_maps(xv, yv, zv):
    x_norm = normalize_interval(xv)
    y_norm = normalize_interval(yv)
    z_norm = normalize_interval(zv)
    x_map = np.repeat(x_norm[:, None, None], len(yv), axis=1).repeat(len(zv), axis=2)
    y_map = np.repeat(y_norm[None, :, None], len(xv), axis=0).repeat(len(zv), axis=2)
    z_map = np.repeat(z_norm[None, None, :], len(xv), axis=0).repeat(len(yv), axis=1)
    return x_map.astype(np.float32), y_map.astype(np.float32), z_map.astype(np.float32)


def project_to_passive(pred_s, eps=1e-12):
    power = pred_s[..., 0] ** 2 + pred_s[..., 1] ** 2 + pred_s[..., 2] ** 2 + pred_s[..., 3] ** 2
    scale = torch.where(power > 1.0, torch.rsqrt(power + eps), torch.ones_like(power))
    out = pred_s.clone()
    out[..., 0] = pred_s[..., 0] * scale
    out[..., 1] = pred_s[..., 1] * scale
    out[..., 2] = pred_s[..., 2] * scale
    out[..., 3] = pred_s[..., 3] * scale
    return out


def s_to_absorption_torch(pred_s):
    return torch.clamp(1.0 - torch.sum(pred_s ** 2, dim=-1), min=0.0, max=1.0)


def split_pred_channels(pred):
    return {
        "Ex": pred[0] + 1j * pred[1],
        "Ey": pred[2] + 1j * pred[3],
        "Ez": pred[4] + 1j * pred[5],
        "Hx": pred[6] + 1j * pred[7],
        "Hy": pred[8] + 1j * pred[9],
        "Hz": pred[10] + 1j * pred[11],
    }


def field_to_view(arr_complex, view_mode: str):
    if view_mode == "real":
        return np.real(arr_complex)
    if view_mode == "imag":
        return np.imag(arr_complex)
    return np.abs(arr_complex)


class LambdaFourierFeatures(nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        freqs = (2.0 ** torch.arange(n_freq).float()) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam_norm):
        x = lam_norm * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def choose_gn_groups(width: int) -> int:
    for groups in (8, 4, 2, 1):
        if width % groups == 0:
            return groups
    return 1


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfftn(x_fp32, dim=(-2, -1), norm="ortho")
            out_ft = torch.zeros(
                x.shape[0],
                self.weight.shape[1],
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            mx = min(self.modes_x, x_ft.shape[-2])
            my = min(self.modes_y, x_ft.shape[-1])
            out_ft[:, :, :mx, :my] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :mx, :my],
                self.weight[:, :, :mx, :my],
            )
            out = torch.fft.irfftn(out_ft, s=x.shape[-2:], norm="ortho")
        return out.to(dtype=x.dtype)


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
            out_ft = torch.zeros(
                x.shape[0],
                self.weight.shape[1],
                x.size(-3),
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            mx = min(self.modes_x, x_ft.shape[-3])
            my = min(self.modes_y, x_ft.shape[-2])
            mz = min(self.modes_z, x_ft.shape[-1])
            out_ft[:, :, :mx, :my, :mz] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_ft[:, :, :mx, :my, :mz],
                self.weight[:, :, :mx, :my, :mz],
            )
            out = torch.fft.irfftn(out_ft, s=x.shape[-3:], norm="ortho")
        return out.to(dtype=x.dtype)


class FNOBlock2d(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class FNOBlock3d(nn.Module):
    def __init__(self, width, modes_x, modes_y, modes_z):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_x, modes_y, modes_z)
        self.pointwise = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class ResidualConv1dBlock(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.norm = nn.GroupNorm(choose_gn_groups(channels), channels)

    def forward(self, x):
        return F.gelu(self.norm(x + self.net(x)))


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


class PatternFNO2dEncoder(nn.Module):
    def __init__(self, modes_x=8, modes_y=8, width=32, depth=3):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2d(width, modes_x, modes_y) for _ in range(depth)])
        self.out_norm = nn.InstanceNorm2d(width)

    def forward(self, pattern_xy):
        if pattern_xy.ndim == 3:
            pattern_xy = pattern_xy.unsqueeze(1)
        x = self.in_proj(pattern_xy)
        for block in self.blocks:
            x = block(x)
        feat_map = self.out_norm(x)
        latent = feat_map.mean(dim=(-2, -1))
        return feat_map, latent


class CurveFieldHybridModel(nn.Module):
    def __init__(
        self,
        modes_x=8,
        modes_y=8,
        modes_z=8,
        width=32,
        depth=3,
        lam_ff=6,
        head_hidden=128,
        field_width=24,
        field_depth=2,
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
            [ResidualConv1dBlock(head_hidden, kernel_size=5) for _ in range(3)]
        )
        self.curve_out = nn.Conv1d(head_hidden, 2, kernel_size=1)
        self.field_in = nn.Sequential(nn.Conv3d(6 + width + 2 * lam_ff, field_width, 1), nn.GELU())
        self.field_blocks = nn.ModuleList([FNOBlock3d(field_width, modes_x, modes_y, modes_z) for _ in range(field_depth)])
        self.field_out = nn.Sequential(nn.Conv3d(field_width, head_hidden, 1), nn.GELU(), nn.Conv3d(head_hidden, 12, 1))

    def encode_pattern(self, pattern_xy):
        return self.pattern_encoder(pattern_xy)

    def decode_curve_from_latent(self, latent, lam_norm):
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(-1)
        b, l, _ = lam_norm.shape
        lam_embed = self.lam_embed(lam_norm.reshape(b * l, 1)).reshape(b, l, -1)
        latent_rep = latent.unsqueeze(1).expand(b, l, latent.shape[-1])
        h = torch.cat([latent_rep, lam_embed], dim=-1)
        h_local = self.curve_in(h.reshape(b * l, -1)).reshape(b, l, -1)
        local_pred = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        h_seq = h_local.transpose(1, 2)
        for block in self.curve_blocks:
            h_seq = block(h_seq)
        seq_pred = self.curve_out(h_seq).transpose(1, 2)
        s11 = local_pred + seq_pred
        zeros = torch.zeros(b, l, 2, dtype=s11.dtype, device=s11.device)
        return torch.cat([s11, zeros], dim=-1)

    def decode_curve_local_from_latent(self, latent, lam_norm):
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(1)
        b, l, _ = lam_norm.shape
        lam_embed = self.lam_embed(lam_norm.reshape(b * l, 1)).reshape(b, l, -1)
        latent_rep = latent.unsqueeze(1).expand(b, l, latent.shape[-1])
        h = torch.cat([latent_rep, lam_embed], dim=-1)
        h_local = self.curve_in(h.reshape(b * l, -1)).reshape(b, l, -1)
        s11 = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        zeros = torch.zeros(b, l, 2, dtype=s11.dtype, device=s11.device)
        return torch.cat([s11, zeros], dim=-1)

    def decode_field_from_encoded(self, feat_map, latent, x_static, lam_norm):
        b, _, nx, ny, nz = x_static.shape
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        lam_embed = self.lam_embed(lam_norm)
        feat_3d = feat_map.unsqueeze(-1).expand(-1, feat_map.shape[1], nx, ny, nz)
        lam_feat = lam_embed.view(b, -1, 1, 1, 1).expand(b, -1, nx, ny, nz)
        x = torch.cat([x_static, feat_3d, lam_feat], dim=1)
        x = self.field_in(x)
        for block in self.field_blocks:
            x = block(x)
        field_out = self.field_out(x)
        s_out = self.decode_curve_local_from_latent(latent, lam_norm).squeeze(1)
        return field_out, s_out


class CurveFieldHybridModelV2(nn.Module):
    def __init__(
        self,
        modes_x=8,
        modes_y=8,
        modes_z=8,
        width=32,
        depth=3,
        lam_ff=6,
        head_hidden=128,
        field_width=24,
        field_depth=2,
    ):
        super().__init__()
        self.pattern_encoder = PatternFNO2dEncoder(modes_x=modes_x, modes_y=modes_y, width=width, depth=depth)
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)
        self.peak_head = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
        )
        self.peak_out = nn.Linear(head_hidden, 3)
        self.peak_context = nn.Sequential(
            nn.Linear(3, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
        )
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
            [ResidualConv1dBlock(head_hidden, kernel_size=5) for _ in range(4)]
        )
        self.curve_out = nn.Conv1d(head_hidden, 2, kernel_size=1)
        self.field_in = nn.Sequential(nn.Conv3d(6 + width + 2 * lam_ff, field_width, 1), nn.GELU())
        self.field_blocks = nn.ModuleList([FNOBlock3d(field_width, modes_x, modes_y, modes_z) for _ in range(field_depth)])
        self.field_out = nn.Sequential(nn.Conv3d(field_width, head_hidden, 1), nn.GELU(), nn.Conv3d(head_hidden, 12, 1))

    def encode_pattern(self, pattern_xy):
        return self.pattern_encoder(pattern_xy)

    def predict_peak_properties_from_latent(self, latent):
        raw = self.peak_out(self.peak_head(latent))
        return torch.stack(
            [
                torch.tanh(raw[:, 0]),
                torch.sigmoid(raw[:, 1]),
                torch.sigmoid(raw[:, 2]),
            ],
            dim=-1,
        )

    def _build_curve_tokens(self, latent, lam_norm, peak_props=None):
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(-1)
        b, l, _ = lam_norm.shape
        lam_embed = self.lam_embed(lam_norm.reshape(b * l, 1)).reshape(b, l, -1)
        latent_rep = latent.unsqueeze(1).expand(b, l, latent.shape[-1])
        tokens = self.curve_in(torch.cat([latent_rep, lam_embed], dim=-1).reshape(b * l, -1)).reshape(b, l, -1)
        if peak_props is None:
            peak_props = self.predict_peak_properties_from_latent(latent)
        peak_ctx = self.peak_context(peak_props).unsqueeze(1).expand(b, l, -1)
        return tokens + peak_ctx

    def decode_curve_from_latent(self, latent, lam_norm, peak_props=None):
        h_local = self._build_curve_tokens(latent, lam_norm, peak_props=peak_props)
        b, l, _ = h_local.shape
        local_pred = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        h_seq = h_local.transpose(1, 2)
        for block in self.curve_blocks:
            h_seq = block(h_seq)
        seq_pred = self.curve_out(h_seq).transpose(1, 2)
        s11 = local_pred + seq_pred
        zeros = torch.zeros(b, l, 2, dtype=s11.dtype, device=s11.device)
        return torch.cat([s11, zeros], dim=-1)

    def decode_curve_local_from_latent(self, latent, lam_norm, peak_props=None):
        h_local = self._build_curve_tokens(latent, lam_norm, peak_props=peak_props)
        b, l, _ = h_local.shape
        s11 = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        zeros = torch.zeros(b, l, 2, dtype=s11.dtype, device=s11.device)
        return torch.cat([s11, zeros], dim=-1)

    def decode_field_from_encoded(self, feat_map, latent, x_static, lam_norm, peak_props=None):
        b, _, nx, ny, nz = x_static.shape
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        lam_embed = self.lam_embed(lam_norm)
        feat_3d = feat_map.unsqueeze(-1).expand(-1, feat_map.shape[1], nx, ny, nz)
        lam_feat = lam_embed.view(b, -1, 1, 1, 1).expand(b, -1, nx, ny, nz)
        x = torch.cat([x_static, feat_3d, lam_feat], dim=1)
        x = self.field_in(x)
        for block in self.field_blocks:
            x = block(x)
        field_out = self.field_out(x)
        if peak_props is None:
            peak_props = self.predict_peak_properties_from_latent(latent)
        s_out = self.decode_curve_local_from_latent(latent, lam_norm, peak_props=peak_props).squeeze(1)
        return field_out, s_out


class CurveFieldHybridModelV3(nn.Module):
    def __init__(
        self,
        modes_x=8,
        modes_y=8,
        modes_z=8,
        width=32,
        depth=3,
        lam_ff=6,
        head_hidden=128,
        field_width=24,
        field_depth=2,
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
            [ResidualConv1dBlock(head_hidden, kernel_size=5) for _ in range(4)]
        )
        self.curve_out = nn.Conv1d(head_hidden, 2, kernel_size=1)
        peak_hidden = max(head_hidden // 2, 16)
        self.main_peak_head = nn.Sequential(
            nn.Conv1d(head_hidden, peak_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(peak_hidden, 1, kernel_size=1),
        )
        self.field_in = nn.Sequential(nn.Conv3d(6 + width + 2 * lam_ff, field_width, 1), nn.GELU())
        self.field_blocks = nn.ModuleList([FNOBlock3d(field_width, modes_x, modes_y, modes_z) for _ in range(field_depth)])
        self.field_out = nn.Sequential(nn.Conv3d(field_width, head_hidden, 1), nn.GELU(), nn.Conv3d(head_hidden, 12, 1))

    def encode_pattern(self, pattern_xy):
        return self.pattern_encoder(pattern_xy)

    def _build_curve_hidden(self, latent, lam_norm):
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(-1)
        b, l, _ = lam_norm.shape
        lam_embed = self.lam_embed(lam_norm.reshape(b * l, 1)).reshape(b, l, -1)
        latent_rep = latent.unsqueeze(1).expand(b, l, latent.shape[-1])
        h_local = self.curve_in(torch.cat([latent_rep, lam_embed], dim=-1).reshape(b * l, -1)).reshape(b, l, -1)
        h_seq = h_local.transpose(1, 2)
        for block in self.curve_blocks:
            h_seq = block(h_seq)
        return h_local, h_seq

    def decode_curve_and_peak_from_latent(self, latent, lam_norm):
        h_local, h_seq = self._build_curve_hidden(latent, lam_norm)
        b, l, _ = h_local.shape
        local_pred = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        seq_pred = self.curve_out(h_seq).transpose(1, 2)
        peak_logits = self.main_peak_head(h_seq).squeeze(1)
        s11 = local_pred + seq_pred
        zeros = torch.zeros(b, l, 2, dtype=s11.dtype, device=s11.device)
        return torch.cat([s11, zeros], dim=-1), peak_logits

    def decode_curve_from_latent(self, latent, lam_norm):
        return self.decode_curve_and_peak_from_latent(latent, lam_norm)[0]

    def predict_main_peak_logits_from_latent(self, latent, lam_norm):
        return self.decode_curve_and_peak_from_latent(latent, lam_norm)[1]

    def decode_curve_local_from_latent(self, latent, lam_norm):
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(1)
        b, l, _ = lam_norm.shape
        lam_embed = self.lam_embed(lam_norm.reshape(b * l, 1)).reshape(b, l, -1)
        latent_rep = latent.unsqueeze(1).expand(b, l, latent.shape[-1])
        h = torch.cat([latent_rep, lam_embed], dim=-1)
        h_local = self.curve_in(h.reshape(b * l, -1)).reshape(b, l, -1)
        s11 = self.curve_local_head(h_local.reshape(b * l, -1)).reshape(b, l, 2)
        zeros = torch.zeros(b, l, 2, dtype=s11.dtype, device=s11.device)
        return torch.cat([s11, zeros], dim=-1)

    def decode_field_from_encoded(self, feat_map, latent, x_static, lam_norm):
        b, _, nx, ny, nz = x_static.shape
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        lam_embed = self.lam_embed(lam_norm)
        feat_3d = feat_map.unsqueeze(-1).expand(-1, feat_map.shape[1], nx, ny, nz)
        lam_feat = lam_embed.view(b, -1, 1, 1, 1).expand(b, -1, nx, ny, nz)
        x = torch.cat([x_static, feat_3d, lam_feat], dim=1)
        x = self.field_in(x)
        for block in self.field_blocks:
            x = block(x)
        field_out = self.field_out(x)
        s_out = self.decode_curve_local_from_latent(latent, lam_norm).squeeze(1)
        return field_out, s_out


class Try2SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y))

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfft2(x_fp32, norm="ortho")
            out_ft = torch.zeros(
                x_fp32.shape[0],
                self.out_channels,
                x_fp32.shape[2],
                x_fp32.shape[3] // 2 + 1,
                dtype=torch.complex64,
                device=x.device,
            )
            mx = min(self.modes_x, x_ft.shape[-2])
            my = min(self.modes_y, x_ft.shape[-1])
            weight = torch.complex(self.weight_real[:, :, :mx, :my], self.weight_imag[:, :, :mx, :my])
            out_ft[:, :, :mx, :my] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :mx, :my],
                weight,
            )
            out = torch.fft.irfft2(out_ft, s=(x_fp32.size(-2), x_fp32.size(-1)), norm="ortho")
        return out.to(dtype=x.dtype)


class Try2PatternEncoder(nn.Module):
    def __init__(self, modes_x=6, modes_y=6, width=64, depth=4):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.spectral = nn.ModuleList([Try2SpectralConv2d(width, width, modes_x, modes_y) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.act = nn.GELU()
        self.out_norm = nn.GroupNorm(choose_gn_groups(width), width)

    def forward(self, pattern_xy):
        if pattern_xy.ndim == 3:
            pattern_xy = pattern_xy.unsqueeze(1)
        x = self.in_proj(pattern_xy)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = self.act(spec(x) + pw(x))
        feat_map = self.out_norm(x)
        latent = feat_map.mean(dim=(-2, -1))
        return feat_map, latent


class Try2CurveFieldTransferModel(nn.Module):
    def __init__(
        self,
        modes_x=6,
        modes_y=6,
        modes_z=6,
        width=64,
        depth=4,
        lam_ff=8,
        head_hidden=256,
        field_width=32,
        field_depth=2,
    ):
        super().__init__()
        self.pattern_encoder = Try2PatternEncoder(modes_x=modes_x, modes_y=modes_y, width=width, depth=depth)
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)
        self.head = nn.Sequential(
            nn.Linear(width + 2 * lam_ff, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 4),
        )
        self.field_in = nn.Sequential(nn.Conv3d(6 + width + 2 * lam_ff, field_width, 1), nn.GELU())
        self.field_blocks = nn.ModuleList([FNOBlock3d(field_width, modes_x, modes_y, modes_z) for _ in range(field_depth)])
        self.field_out = nn.Sequential(nn.Conv3d(field_width, head_hidden, 1), nn.GELU(), nn.Conv3d(head_hidden, 12, 1))

    def encode_pattern(self, pattern_xy):
        return self.pattern_encoder(pattern_xy)

    def decode_curve_from_latent(self, latent, lam_norm):
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(-1)
        batch_size, n_lambda, _ = lam_norm.shape
        latent_rep = latent.unsqueeze(1).expand(batch_size, n_lambda, latent.shape[-1])
        lam_embed = self.lam_embed(lam_norm.reshape(batch_size * n_lambda, 1)).reshape(batch_size, n_lambda, -1)
        curve_tokens = torch.cat([latent_rep, lam_embed], dim=-1).reshape(batch_size * n_lambda, -1)
        curve_out = self.head(curve_tokens).reshape(batch_size, n_lambda, 4)
        zeros = torch.zeros(batch_size, n_lambda, 2, dtype=curve_out.dtype, device=curve_out.device)
        return torch.cat([curve_out[..., :2], zeros], dim=-1)

    def decode_curve_local_from_latent(self, latent, lam_norm):
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        if lam_norm.ndim == 2:
            lam_norm = lam_norm.unsqueeze(1)
        return self.decode_curve_from_latent(latent, lam_norm)

    def decode_field_from_encoded(self, feat_map, latent, x_static, lam_norm):
        batch_size, _, nx, ny, nz = x_static.shape
        if lam_norm.ndim == 1:
            lam_norm = lam_norm.unsqueeze(-1)
        lam_embed = self.lam_embed(lam_norm)
        feat_3d = feat_map.unsqueeze(-1).expand(-1, feat_map.shape[1], nx, ny, nz)
        lam_feat = lam_embed.view(batch_size, -1, 1, 1, 1).expand(batch_size, -1, nx, ny, nz)
        x = torch.cat([x_static, feat_3d, lam_feat], dim=1)
        x = self.field_in(x)
        for block in self.field_blocks:
            x = block(x)
        field_out = self.field_out(x)
        s_out = self.decode_curve_local_from_latent(latent, lam_norm).squeeze(1)
        return field_out, s_out


class FullFieldDualSurrogatePredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        meta_path: Path,
        *,
        device: str = "cpu",
        bottom_metal_zmax: float = 100e-9,
        dielectric_zmax: float = 400e-9,
        top_pattern_zmax: float = 430e-9,
        forward_batch_size: int = 64,
        lambda_chunk_size: int = 16,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.meta_path = Path(meta_path)
        self.device = torch.device(device)
        self.bottom_metal_zmax = float(bottom_metal_zmax)
        self.dielectric_zmax = float(dielectric_zmax)
        self.top_pattern_zmax = float(top_pattern_zmax)
        self.forward_batch_size = int(forward_batch_size)
        self.lambda_chunk_size = int(lambda_chunk_size)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        cfg = checkpoint.get("config", {})
        self.config = cfg
        self.model_family = cfg.get("MODEL_FAMILY", MODEL_FAMILY_LEGACY)

        self.down_x = int(cfg.get("DOWN_X", 1))
        self.down_y = int(cfg.get("DOWN_Y", 1))
        self.down_z = int(cfg.get("DOWN_Z", 1))
        self.modes_x = int(cfg.get("MODES_X", 10))
        self.modes_y = int(cfg.get("MODES_Y", 10))
        self.modes_z = int(cfg.get("MODES_Z", 10))
        self.width = int(cfg.get("WIDTH", 32))
        self.depth = int(cfg.get("DEPTH", 4))
        self.lam_ff = int(cfg.get("LAM_FF", 8))
        self.head_hidden = int(cfg.get("HEAD_HIDDEN", 128))
        self.field_width = int(cfg.get("FIELD_WIDTH", 24))
        self.field_depth = int(cfg.get("FIELD_DEPTH", 2))

        lambda_vec = checkpoint["lambda_vec"]
        if isinstance(lambda_vec, torch.Tensor):
            lambda_vec = lambda_vec.detach().cpu().numpy()
        self.lambda_vec = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)
        self.lambda_norm = torch.from_numpy(normalize_interval(self.lambda_vec)).to(self.device)

        meta = load_mat_auto(self.meta_path)
        self.xv = standardize_coord_1d(meta["xv"])[:: self.down_x]
        self.yv = standardize_coord_1d(meta["yv"])[:: self.down_y]
        self.zv = standardize_coord_1d(meta["zv"])[:: self.down_z]
        self.nx = len(self.xv)
        self.ny = len(self.yv)
        self.nz = len(self.zv)

        x_map, y_map, z_map = make_coord_maps(self.xv, self.yv, self.zv)
        self.coord_maps = torch.from_numpy(np.stack([x_map, y_map, z_map], axis=0)).to(self.device)

        self.bottom_mask_z = torch.from_numpy((self.zv <= self.bottom_metal_zmax).astype(np.float32)).to(self.device)
        self.dielectric_mask_z = torch.from_numpy(
            ((self.zv > self.bottom_metal_zmax) & (self.zv <= self.dielectric_zmax)).astype(np.float32)
        ).to(self.device)
        self.top_mask_z = torch.from_numpy(
            ((self.zv > self.dielectric_zmax) & (self.zv <= self.top_pattern_zmax)).astype(np.float32)
        ).to(self.device)

        metal_eps = np.asarray(au_eps_from_lambda_m(self.lambda_vec), dtype=np.complex64)
        dielectric_eps = np.asarray(sio2_eps_from_lambda_m(self.lambda_vec), dtype=np.complex64)
        air_eps = np.asarray(air_eps_from_lambda_m(self.lambda_vec), dtype=np.complex64)

        self.metal_real = torch.from_numpy(np.real(metal_eps).astype(np.float32)).to(self.device)
        self.metal_imag = torch.from_numpy(np.imag(metal_eps).astype(np.float32)).to(self.device)
        self.diel_real = torch.from_numpy(np.real(dielectric_eps).astype(np.float32)).to(self.device)
        self.diel_imag = torch.from_numpy(np.imag(dielectric_eps).astype(np.float32)).to(self.device)
        self.air_real = torch.from_numpy(np.real(air_eps).astype(np.float32)).to(self.device)
        self.air_imag = torch.from_numpy(np.imag(air_eps).astype(np.float32)).to(self.device)

        self.eps_real_lz = self._build_eps_lz(self.metal_real, self.diel_real, self.air_real)
        self.eps_imag_lz = self._build_eps_lz(self.metal_imag, self.diel_imag, self.air_imag)
        self.bottom_metal_mask_lz = self.bottom_mask_z[None, :].expand(len(self.lambda_vec), -1)

        if self.model_family == MODEL_FAMILY_CURVE_FIELD:
            self.model = CurveFieldHybridModel(
                modes_x=self.modes_x,
                modes_y=self.modes_y,
                modes_z=self.modes_z,
                width=self.width,
                depth=self.depth,
                lam_ff=self.lam_ff,
                head_hidden=self.head_hidden,
                field_width=self.field_width,
                field_depth=self.field_depth,
            ).to(self.device)
        elif self.model_family == MODEL_FAMILY_CURVE_FIELD_V2:
            self.model = CurveFieldHybridModelV2(
                modes_x=self.modes_x,
                modes_y=self.modes_y,
                modes_z=self.modes_z,
                width=self.width,
                depth=self.depth,
                lam_ff=self.lam_ff,
                head_hidden=self.head_hidden,
                field_width=self.field_width,
                field_depth=self.field_depth,
            ).to(self.device)
        elif self.model_family == MODEL_FAMILY_CURVE_FIELD_V3:
            self.model = CurveFieldHybridModelV3(
                modes_x=self.modes_x,
                modes_y=self.modes_y,
                modes_z=self.modes_z,
                width=self.width,
                depth=self.depth,
                lam_ff=self.lam_ff,
                head_hidden=self.head_hidden,
                field_width=self.field_width,
                field_depth=self.field_depth,
            ).to(self.device)
        elif self.model_family == MODEL_FAMILY_TRY2_TRANSFER:
            self.model = Try2CurveFieldTransferModel(
                modes_x=self.modes_x,
                modes_y=self.modes_y,
                modes_z=self.modes_z,
                width=self.width,
                depth=self.depth,
                lam_ff=self.lam_ff,
                head_hidden=self.head_hidden,
                field_width=self.field_width,
                field_depth=self.field_depth,
            ).to(self.device)
        else:
            self.model = FNO3dConditionalField(
                base_in=6,
                modes_x=self.modes_x,
                modes_y=self.modes_y,
                modes_z=self.modes_z,
                width=self.width,
                depth=self.depth,
                lam_ff=self.lam_ff,
                head_hidden=self.head_hidden,
            ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def _build_eps_lz(self, metal, dielectric, air):
        out = air[:, None].expand(-1, self.nz).clone()
        out[:, self.bottom_mask_z.bool()] = metal[:, None].expand(-1, int(self.bottom_mask_z.sum().item()))
        diel_top = (self.dielectric_mask_z + self.top_mask_z).bool()
        out[:, diel_top] = dielectric[:, None].expand(-1, int(diel_top.sum().item()))
        return out.contiguous()

    def _resize_patterns(self, patterns_11):
        if isinstance(patterns_11, np.ndarray):
            pattern_tensor = torch.from_numpy(patterns_11.astype(np.float32))
        else:
            pattern_tensor = patterns_11.detach().float().cpu()
        if pattern_tensor.ndim == 2:
            pattern_tensor = pattern_tensor.unsqueeze(0)
        pattern_tensor = pattern_tensor[:, None, :, :]
        pattern_xy = F.interpolate(pattern_tensor, size=(self.nx, self.ny), mode="nearest")[:, 0]
        return pattern_xy.to(self.device)

    def _build_input_chunk(self, pattern_xy, lambda_indices):
        b = pattern_xy.shape[0]
        idx = torch.as_tensor(lambda_indices, dtype=torch.long, device=self.device)
        l = idx.numel()

        pattern_xy = (pattern_xy > 0.5).float()
        top_pattern = pattern_xy[:, None, :, :, None] * self.top_mask_z[None, None, None, None, :]
        top_pattern = top_pattern.expand(b, l, self.nx, self.ny, self.nz)

        base_real = self.eps_real_lz[idx][None, :, None, None, :].expand(b, l, self.nx, self.ny, self.nz).clone()
        base_imag = self.eps_imag_lz[idx][None, :, None, None, :].expand(b, l, self.nx, self.ny, self.nz).clone()
        delta_real = (self.metal_real[idx] - self.diel_real[idx])[None, :, None, None, None]
        delta_imag = (self.metal_imag[idx] - self.diel_imag[idx])[None, :, None, None, None]
        eps_real = base_real + top_pattern * delta_real
        eps_imag = base_imag + top_pattern * delta_imag

        metal_mask = self.bottom_metal_mask_lz[idx][None, :, None, None, :].expand(b, l, self.nx, self.ny, self.nz).clone()
        metal_mask = torch.clamp(metal_mask + top_pattern, 0.0, 1.0)
        coord_maps = self.coord_maps[None, None, :, :, :, :].expand(b, l, -1, -1, -1, -1)
        x_static = torch.cat(
            [metal_mask.unsqueeze(2), eps_real.unsqueeze(2), eps_imag.unsqueeze(2), coord_maps],
            dim=2,
        )
        lam_norm = self.lambda_norm[idx][None, :, None].expand(b, l, 1)

        return x_static.reshape(b * l, 6, self.nx, self.ny, self.nz), lam_norm.reshape(b * l, 1), b, l

    @torch.no_grad()
    def predict_spectrum(self, patterns_11):
        pattern_xy = self._resize_patterns(patterns_11)
        b = pattern_xy.shape[0]

        if self.model_family == MODEL_FAMILY_CURVE_FIELD:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            lam_all = self.lambda_norm[None, :, None].expand(b, -1, 1)
            pred_s = project_to_passive(self.model.decode_curve_from_latent(latent, lam_all))
        elif self.model_family == MODEL_FAMILY_CURVE_FIELD_V2:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            peak_props = self.model.predict_peak_properties_from_latent(latent)
            lam_all = self.lambda_norm[None, :, None].expand(b, -1, 1)
            pred_s = project_to_passive(self.model.decode_curve_from_latent(latent, lam_all, peak_props=peak_props))
        elif self.model_family == MODEL_FAMILY_CURVE_FIELD_V3:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            lam_all = self.lambda_norm[None, :, None].expand(b, -1, 1)
            pred_s = project_to_passive(self.model.decode_curve_from_latent(latent, lam_all))
        elif self.model_family == MODEL_FAMILY_TRY2_TRANSFER:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            lam_all = self.lambda_norm[None, :, None].expand(b, -1, 1)
            pred_s = project_to_passive(self.model.decode_curve_from_latent(latent, lam_all))
        else:
            k = len(self.lambda_vec)
            pred_s = torch.empty((b, k, 4), dtype=torch.float32, device=self.device)
            for start in range(0, k, self.lambda_chunk_size):
                stop = min(k, start + self.lambda_chunk_size)
                indices = list(range(start, stop))
                x_chunk, lam_chunk, _, l = self._build_input_chunk(pattern_xy, indices)
                s_parts = []
                for fb_start in range(0, x_chunk.shape[0], self.forward_batch_size):
                    fb_stop = min(x_chunk.shape[0], fb_start + self.forward_batch_size)
                    _, s_out = self.model(x_chunk[fb_start:fb_stop], lam_chunk[fb_start:fb_stop])
                    s_parts.append(project_to_passive(s_out))
                s_block = torch.cat(s_parts, dim=0).reshape(b, l, 4)
                pred_s[:, start:stop, :] = s_block

        absorption = s_to_absorption_torch(pred_s)
        return absorption.detach().cpu().numpy().astype(np.float32), pred_s.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_field_at_lambda(self, patterns_11, *, lambda_index: int | None = None, lambda_value: float | None = None):
        pattern_xy = self._resize_patterns(patterns_11)
        b = pattern_xy.shape[0]

        if lambda_value is not None:
            lam_idx = int(np.argmin(np.abs(self.lambda_vec - float(lambda_value))))
        elif lambda_index is not None:
            lam_idx = int(lambda_index)
        else:
            raise ValueError("lambda_index 和 lambda_value 至少提供一个")
        lam_idx = max(0, min(lam_idx, len(self.lambda_vec) - 1))

        x_chunk, lam_chunk, _, _ = self._build_input_chunk(pattern_xy, [lam_idx])

        if self.model_family == MODEL_FAMILY_CURVE_FIELD:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            feat_map_rep = feat_map
            latent_rep = latent
            field_out, s_out = self.model.decode_field_from_encoded(feat_map_rep, latent_rep, x_chunk, lam_chunk)
        elif self.model_family == MODEL_FAMILY_CURVE_FIELD_V2:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            peak_props = self.model.predict_peak_properties_from_latent(latent)
            field_out, s_out = self.model.decode_field_from_encoded(
                feat_map,
                latent,
                x_chunk,
                lam_chunk,
                peak_props=peak_props,
            )
        elif self.model_family == MODEL_FAMILY_CURVE_FIELD_V3:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            field_out, s_out = self.model.decode_field_from_encoded(feat_map, latent, x_chunk, lam_chunk)
        elif self.model_family == MODEL_FAMILY_TRY2_TRANSFER:
            feat_map, latent = self.model.encode_pattern(pattern_xy)
            field_out, s_out = self.model.decode_field_from_encoded(feat_map, latent, x_chunk, lam_chunk)
        else:
            field_out, s_out = self.model(x_chunk, lam_chunk)

        field_out = field_out.reshape(b, 12, self.nx, self.ny, self.nz)
        s_out = project_to_passive(s_out).reshape(b, 4)

        field_np = field_out.detach().cpu().numpy().astype(np.float32)
        s_np = s_out.detach().cpu().numpy().astype(np.float32)
        split_fields = [split_pred_channels(field_np[i]) for i in range(b)]

        return {
            "lambda_index": lam_idx,
            "lambda_m": float(self.lambda_vec[lam_idx]),
            "fields": split_fields,
            "sparams": s_np,
        }
