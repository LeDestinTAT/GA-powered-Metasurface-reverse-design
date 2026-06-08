import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.project_paths import TRAIN_RUN_OUTPUTS_DIR

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

EPS0 = 8.854187817e-12
MU0 = 4.0e-7 * math.pi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a 3D FNO surrogate with Maxwell residual losses.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--patterns-mat", type=Path, default=None)
    p.add_argument("--sampling-meta", type=Path, default=None)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=Path, default=TRAIN_RUN_OUTPUTS_DIR / "fno_maxwell")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--modes-x", type=int, default=12)
    p.add_argument("--modes-y", type=int, default=12)
    p.add_argument("--modes-z", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--in-channels", type=int, default=7)
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--predict-h", action="store_true")
    p.add_argument("--time-convention", choices=["exp_minus_iwt", "exp_plus_iwt"], default="exp_minus_iwt")
    p.add_argument("--downsample-x", type=int, default=1)
    p.add_argument("--downsample-y", type=int, default=1)
    p.add_argument("--downsample-z", type=int, default=1)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--metal-eps-real", type=float, default=-200.0)
    p.add_argument("--metal-eps-imag", type=float, default=80.0)
    p.add_argument("--dielectric-eps-real", type=float, default=2.25)
    p.add_argument("--dielectric-eps-imag", type=float, default=0.0)
    p.add_argument("--air-eps-real", type=float, default=1.0)
    p.add_argument("--air-eps-imag", type=float, default=0.0)
    p.add_argument("--bottom-metal-zmax", type=float, default=100e-9)
    p.add_argument("--dielectric-zmax", type=float, default=400e-9)
    p.add_argument("--top-pattern-zmax", type=float, default=430e-9)
    p.add_argument("--period-x", type=float, default=2.8e-6)
    p.add_argument("--period-y", type=float, default=2.8e-6)
    p.add_argument("--field-weight", type=float, default=1.0)
    p.add_argument("--curl-e-weight", type=float, default=0.1)
    p.add_argument("--curl-h-weight", type=float, default=0.1)
    p.add_argument("--div-weight", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--supervise-components", type=str, default="Ex,Ey,Ez,Hx,Hy,Hz")
    p.add_argument("--target-scale-mode", choices=["none", "per_sample_rms"], default="per_sample_rms")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def component_aliases() -> Dict[str, Sequence[str]]:
    return {
        "Ex": ("Ex", "ex", "E_x"),
        "Ey": ("Ey", "ey", "E_y"),
        "Ez": ("Ez", "ez", "E_z"),
        "Hx": ("Hx", "hx", "H_x"),
        "Hy": ("Hy", "hy", "H_y"),
        "Hz": ("Hz", "hz", "H_z"),
        "x": ("x", "X", "x_grid", "xx"),
        "y": ("y", "Y", "y_grid", "yy"),
        "z": ("z", "Z", "z_grid", "zz"),
        "freq": ("freq", "frequency", "f", "freq_hz"),
        "omega": ("omega", "w", "angular_frequency"),
        "pattern": ("pattern", "selected", "structure", "binary_pattern", "mask11", "top_pattern"),
        "s11": ("S11", "s11"),
        "s21": ("S21", "s21"),
        "R": ("R", "r", "reflectance"),
        "T": ("T", "t", "transmittance"),
        "A": ("A", "a", "absorbance"),
    }


def str_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_sample_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse sample index from {path.name}")
    return int(match.group(1))


def matlab_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_ and obj.size == 1:
            return matlab_to_numpy(obj.item())
        return np.asarray(obj)
    return obj


def h5_to_numpy(obj):
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, np.ndarray) and data.dtype.fields and "real" in data.dtype.fields and "imag" in data.dtype.fields:
            return data["real"] + 1j * data["imag"]
        return np.asarray(data)
    if isinstance(obj, h5py.Group):
        keys = set(obj.keys())
        if {"real", "imag"}.issubset(keys):
            return np.asarray(obj["real"][()]) + 1j * np.asarray(obj["imag"][()])
        return {k: h5_to_numpy(v) for k, v in obj.items()}
    return obj


def load_mat_dict(path: Path) -> Dict[str, object]:
    if h5py is not None:
        try:
            with h5py.File(path, "r") as handle:
                return {k: h5_to_numpy(v) for k, v in handle.items()}
        except OSError:
            pass
    if loadmat is None:
        raise RuntimeError("scipy is required to load non-v7.3 MATLAB files.")
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: matlab_to_numpy(v) for k, v in data.items() if not k.startswith("__")}


def find_first_key(data: Dict[str, object], aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if alias in data:
            return alias
    lowered = {k.lower(): k for k in data.keys()}
    for alias in aliases:
        key = lowered.get(alias.lower())
        if key is not None:
            return key
    return None


def squeeze_to_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim > 1:
        shape = [dim for dim in arr.shape if dim > 1]
        if len(shape) == 1:
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"Expected 1D array, got {arr.shape}")
    return arr.astype(np.float64)


def ensure_complex_channels(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    real = np.real(arr).astype(np.float32)
    imag = np.imag(arr).astype(np.float32) if np.iscomplexobj(arr) else np.zeros_like(real)
    return np.stack([real, imag], axis=0)


def nearest_resize_2d(pattern: np.ndarray, nx: int, ny: int) -> np.ndarray:
    tensor = torch.from_numpy(pattern.astype(np.float32)[None, None, ...])
    out = F.interpolate(tensor, size=(nx, ny), mode="nearest")
    return out[0, 0].numpy()


def normalize_coord_map(coord: np.ndarray, axis: int, nx: int, ny: int, nz: int) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float32)
    denom = max(float(coord.max() - coord.min()), 1e-12)
    coord = (coord - float(coord.min())) / denom
    if axis == 0:
        return np.repeat(coord[:, None, None], ny, axis=1).repeat(nz, axis=2)
    if axis == 1:
        return np.repeat(coord[None, :, None], nx, axis=0).repeat(nz, axis=2)
    return np.repeat(coord[None, None, :], nx, axis=0).repeat(ny, axis=1)


class FieldDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        args: argparse.Namespace,
        predict_h: bool,
        supervise_components: Sequence[str],
    ) -> None:
        self.files = [Path(f) for f in files]
        self.args = args
        self.predict_h = predict_h
        self.supervise_components = [c for c in supervise_components if predict_h or not c.startswith("H")]
        self.alias = component_aliases()
        self.pattern_bank = self._load_pattern_bank(args.patterns_mat) if args.patterns_mat else None
        meta_path = args.sampling_meta or (args.data_dir / "sampling_meta.mat")
        self.meta = self._load_optional_meta(meta_path)

    def _load_pattern_bank(self, path: Path) -> np.ndarray:
        data = load_mat_dict(path)
        key = find_first_key(data, self.alias["pattern"])
        if key is None:
            raise KeyError(f"Pattern variable not found in {path}")
        patterns = np.asarray(data[key])
        if patterns.ndim == 2:
            patterns = patterns[:, :, None]
        return patterns.astype(np.float32)

    def _load_optional_meta(self, path: Path) -> Optional[Dict[str, object]]:
        if path is None or not path.exists():
            return None
        try:
            return load_mat_dict(path)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        file_path = self.files[index]
        sample = load_mat_dict(file_path)
        sample_id = parse_sample_index(file_path)

        x0 = self._get_coord(sample, "x")
        y0 = self._get_coord(sample, "y")
        z0 = self._get_coord(sample, "z")
        z_mask = self._crop_z_mask(z0)

        x = x0[:: self.args.downsample_x]
        y = y0[:: self.args.downsample_y]
        z = z0[z_mask][:: self.args.downsample_z]

        fields = []
        for comp in self.supervise_components:
            key = find_first_key(sample, self.alias[comp])
            if key is None:
                raise KeyError(f"Missing {comp} in {file_path}")
            arr = np.asarray(sample[key])
            arr = self._reshape_field(arr, len(x0), len(y0), len(z0))
            arr = arr[:, :, z_mask]
            arr = arr[:: self.args.downsample_x, :: self.args.downsample_y, :: self.args.downsample_z]
            fields.append(ensure_complex_channels(arr))

        pattern = self._get_pattern(sample, sample_id)
        eps = self._build_eps(pattern, x, y, z)
        omega = self._get_omega(sample)
        labels = self._collect_reference_labels(sample)

        targets = np.concatenate(fields, axis=0).astype(np.float32)
        scale = self._compute_scale(targets)
        targets = targets / scale
        inputs = self._build_inputs(pattern, x, y, z, eps, omega)
        eps_channels = np.stack([np.real(eps), np.imag(eps)], axis=0).astype(np.float32)

        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "eps": torch.from_numpy(eps_channels),
            "coords": {
                "x": torch.from_numpy(x.astype(np.float32)),
                "y": torch.from_numpy(y.astype(np.float32)),
                "z": torch.from_numpy(z.astype(np.float32)),
            },
            "omega": torch.tensor(float(omega), dtype=torch.float32),
            "scale": torch.tensor(float(scale), dtype=torch.float32),
            "labels": {k: torch.as_tensor(v, dtype=torch.float32) for k, v in labels.items()},
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
        }

    def _get_coord(self, sample: Dict[str, object], coord_name: str) -> np.ndarray:
        key = find_first_key(sample, self.alias[coord_name])
        if key is None and self.meta is not None:
            key = find_first_key(self.meta, self.alias[coord_name])
            if key is not None:
                return squeeze_to_1d(np.asarray(self.meta[key]))
        if key is None:
            raise KeyError(f"Missing coordinate {coord_name}")
        return squeeze_to_1d(np.asarray(sample[key]))

    def _crop_z_mask(self, z: np.ndarray) -> np.ndarray:
        mask = np.ones_like(z, dtype=bool)
        if self.args.z_min is not None:
            mask &= z >= self.args.z_min
        if self.args.z_max is not None:
            mask &= z <= self.args.z_max
        if not np.any(mask):
            raise ValueError("z crop removed all samples")
        return mask

    def _reshape_field(self, arr: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        arr = np.squeeze(np.asarray(arr))
        total = nx * ny * nz
        if arr.shape == (nx, ny, nz):
            return arr
        if arr.shape == (ny, nx, nz):
            return np.transpose(arr, (1, 0, 2))
        if arr.size == total:
            return arr.reshape(nx, ny, nz)
        raise ValueError(f"Cannot reshape field {arr.shape} -> {(nx, ny, nz)}")

    def _get_pattern(self, sample: Dict[str, object], sample_id: int) -> np.ndarray:
        key = find_first_key(sample, self.alias["pattern"])
        if key is not None:
            pattern = np.squeeze(np.asarray(sample[key]))
        else:
            if self.pattern_bank is None:
                raise KeyError("Pattern not found in sample and --patterns-mat is missing.")
            pattern = self.pattern_bank[:, :, sample_id - 1]
        if pattern.ndim != 2:
            raise ValueError(f"Expected 2D pattern, got {pattern.shape}")
        return pattern.astype(np.float32)

    def _get_omega(self, sample: Dict[str, object]) -> float:
        omega_key = find_first_key(sample, self.alias["omega"])
        if omega_key is not None:
            return float(np.asarray(sample[omega_key]).squeeze())
        freq_key = find_first_key(sample, self.alias["freq"])
        if freq_key is not None:
            return 2.0 * math.pi * float(np.asarray(sample[freq_key]).squeeze())
        if self.meta is not None:
            for name in ("omega", "freq"):
                key = find_first_key(self.meta, self.alias[name])
                if key is not None:
                    value = float(np.asarray(self.meta[key]).squeeze())
                    return value if name == "omega" else 2.0 * math.pi * value
        raise KeyError("Missing omega/frequency in sample and metadata.")

    def _collect_reference_labels(self, sample: Dict[str, object]) -> Dict[str, np.ndarray]:
        labels: Dict[str, np.ndarray] = {}
        for name in ("s11", "s21", "R", "T", "A"):
            key = find_first_key(sample, self.alias[name])
            if key is None:
                continue
            value = np.asarray(sample[key]).squeeze()
            if np.iscomplexobj(value):
                labels[name] = np.array([np.real(value), np.imag(value)], dtype=np.float32)
            else:
                labels[name] = np.array(value, dtype=np.float32, ndmin=1)
        return labels

    def _build_eps(self, pattern11: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        nx, ny, nz = len(x), len(y), len(z)
        pattern_xy = nearest_resize_2d(pattern11, nx, ny)
        eps_air = complex(self.args.air_eps_real, self.args.air_eps_imag)
        eps_die = complex(self.args.dielectric_eps_real, self.args.dielectric_eps_imag)
        eps_metal = complex(self.args.metal_eps_real, self.args.metal_eps_imag)
        eps = np.full((nx, ny, nz), eps_air, dtype=np.complex64)

        bottom = z <= self.args.bottom_metal_zmax
        dielectric = (z > self.args.bottom_metal_zmax) & (z <= self.args.dielectric_zmax)
        top = (z > self.args.dielectric_zmax) & (z <= self.args.top_pattern_zmax)

        if np.any(bottom):
            eps[:, :, bottom] = eps_metal
        if np.any(dielectric):
            eps[:, :, dielectric] = eps_die
        if np.any(top):
            eps[:, :, top] = eps_die
            metal_xy = pattern_xy > 0.5
            for zi in np.where(top)[0]:
                eps[:, :, zi][metal_xy] = eps_metal
        return eps

    def _build_inputs(
        self, pattern11: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: np.ndarray, omega: float
    ) -> np.ndarray:
        nx, ny, nz = len(x), len(y), len(z)
        pattern_xy = nearest_resize_2d(pattern11, nx, ny)
        pattern_3d = np.repeat(pattern_xy[:, :, None], nz, axis=2)
        omega_map = np.full((nx, ny, nz), omega / (2.0 * math.pi * 1e12), dtype=np.float32)
        channels = [
            pattern_3d.astype(np.float32),
            np.real(eps).astype(np.float32),
            np.imag(eps).astype(np.float32),
            normalize_coord_map(x, 0, nx, ny, nz),
            normalize_coord_map(y, 1, nx, ny, nz),
            normalize_coord_map(z, 2, nx, ny, nz),
            omega_map,
        ]
        return np.stack(channels[: self.args.in_channels], axis=0)

    def _compute_scale(self, targets: np.ndarray) -> np.float32:
        if self.args.target_scale_mode == "none":
            return np.float32(1.0)
        rms = np.sqrt(np.mean(np.square(targets), dtype=np.float64) + 1e-12)
        return np.float32(max(rms, 1e-6))


def collate_samples(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "inputs": torch.stack([b["inputs"] for b in batch], dim=0),
        "targets": torch.stack([b["targets"] for b in batch], dim=0),
        "eps": torch.stack([b["eps"] for b in batch], dim=0),
        "coords": {k: torch.stack([b["coords"][k] for b in batch], dim=0) for k in ("x", "y", "z")},
        "omega": torch.stack([b["omega"] for b in batch], dim=0),
        "scale": torch.stack([b["scale"] for b in batch], dim=0),
        "labels": [b["labels"] for b in batch],
        "sample_ids": torch.stack([b["sample_id"] for b in batch], dim=0),
    }


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int) -> None:
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1))
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
            out = torch.fft.irfftn(out_ft, s=x.shape[-3:])
        return out.to(dtype=x.dtype)


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int, modes_z: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_x, modes_y, modes_z)
        self.pointwise = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class FNO3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, width: int, hidden_channels: int, depth: int,
        modes_x: int, modes_y: int, modes_z: int
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Conv3d(in_channels, width, 1), nn.GELU())
        self.blocks = nn.ModuleList([FNOBlock3d(width, modes_x, modes_y, modes_z) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv3d(width, hidden_channels, 1), nn.GELU(), nn.Conv3d(hidden_channels, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def split_components(tensor: torch.Tensor, supervise_components: Sequence[str]) -> Dict[str, torch.Tensor]:
    chunks = torch.chunk(tensor, len(supervise_components), dim=1)
    return {name: chunk for name, chunk in zip(supervise_components, chunks)}


def channels_to_complex(two_channel_tensor: torch.Tensor) -> torch.Tensor:
    return torch.complex(two_channel_tensor[:, 0], two_channel_tensor[:, 1])


def finite_difference(values: torch.Tensor, coords: torch.Tensor, axis: int) -> torch.Tensor:
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


def curl_3d(vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor, coords: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_vz_dy = finite_difference(vz, coords["y"], axis=2)
    d_vy_dz = finite_difference(vy, coords["z"], axis=3)
    d_vx_dz = finite_difference(vx, coords["z"], axis=3)
    d_vz_dx = finite_difference(vz, coords["x"], axis=1)
    d_vy_dx = finite_difference(vy, coords["x"], axis=1)
    d_vx_dy = finite_difference(vx, coords["y"], axis=2)
    return d_vz_dy - d_vy_dz, d_vx_dz - d_vz_dx, d_vy_dx - d_vx_dy


def divergence_3d(vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor, coords: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        finite_difference(vx, coords["x"], axis=1)
        + finite_difference(vy, coords["y"], axis=2)
        + finite_difference(vz, coords["z"], axis=3)
    )


def make_complex_field_dict(prediction: torch.Tensor, supervise_components: Sequence[str], predict_h: bool) -> Dict[str, torch.Tensor]:
    parts = split_components(prediction, supervise_components)
    out = {name: channels_to_complex(value) for name, value in parts.items()}
    if not predict_h:
        zeros = torch.zeros_like(out["Ex"])
        out["Hx"] = zeros
        out["Hy"] = zeros
        out["Hz"] = zeros
    return out


def maxwell_residuals(
    pred: torch.Tensor,
    eps_channels: torch.Tensor,
    omega: torch.Tensor,
    coords: Dict[str, torch.Tensor],
    supervise_components: Sequence[str],
    predict_h: bool,
    time_convention: str,
) -> Dict[str, torch.Tensor]:
    field = make_complex_field_dict(pred, supervise_components, predict_h)
    eps = torch.complex(eps_channels[:, 0], eps_channels[:, 1])
    curl_e = curl_3d(field["Ex"], field["Ey"], field["Ez"], coords)
    curl_h = curl_3d(field["Hx"], field["Hy"], field["Hz"], coords)
    omega_c = omega.view(-1, 1, 1, 1).to(pred.device)

    if time_convention == "exp_minus_iwt":
        curl_e_res = [ce - 1j * omega_c * MU0 * h for ce, h in zip(curl_e, (field["Hx"], field["Hy"], field["Hz"]))]
        curl_h_res = [ch + 1j * omega_c * EPS0 * eps * e for ch, e in zip(curl_h, (field["Ex"], field["Ey"], field["Ez"]))]
    else:
        curl_e_res = [ce + 1j * omega_c * MU0 * h for ce, h in zip(curl_e, (field["Hx"], field["Hy"], field["Hz"]))]
        curl_h_res = [ch - 1j * omega_c * EPS0 * eps * e for ch, e in zip(curl_h, (field["Ex"], field["Ey"], field["Ez"]))]

    div_d = divergence_3d(eps * field["Ex"], eps * field["Ey"], eps * field["Ez"], coords)
    div_b = divergence_3d(MU0 * field["Hx"], MU0 * field["Hy"], MU0 * field["Hz"], coords)
    return {
        "curl_e": torch.stack([torch.abs(v) ** 2 for v in curl_e_res], dim=1).mean(),
        "curl_h": torch.stack([torch.abs(v) ** 2 for v in curl_h_res], dim=1).mean(),
        "div": (torch.abs(div_d) ** 2).mean() + (torch.abs(div_b) ** 2).mean(),
    }


def relative_field_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) / (torch.mean(target.pow(2)) + 1e-8)


def smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    return targets if smoothing <= 0 else (1.0 - smoothing) * targets


def build_dataloaders(args: argparse.Namespace, supervise_components: Sequence[str], predict_h: bool) -> Tuple[DataLoader, DataLoader]:
    files = sorted(args.data_dir.glob("sample_*.mat"))
    if not files:
        raise FileNotFoundError(f"No sample_*.mat files found in {args.data_dir}")
    if args.max_samples is not None:
        files = files[: args.max_samples]
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    files = [files[i] for i in order]
    n_train = int(len(files) * args.train_frac)
    n_train = max(1, min(n_train, len(files) - 1))
    train_files = files[:n_train]
    val_files = files[n_train:] if n_train < len(files) else files[-1:]
    train_ds = FieldDataset(train_files, args, predict_h, supervise_components)
    val_ds = FieldDataset(val_files, args, predict_h, supervise_components)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_samples, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_samples, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "inputs": batch["inputs"].to(device),
        "targets": batch["targets"].to(device),
        "eps": batch["eps"].to(device),
        "coords": {k: v.to(device) for k, v in batch["coords"].items()},
        "omega": batch["omega"].to(device),
        "scale": batch["scale"].to(device),
        "labels": batch["labels"],
        "sample_ids": batch["sample_ids"].to(device),
    }


def compute_losses(pred: torch.Tensor, batch: Dict[str, object], args: argparse.Namespace, supervise_components: Sequence[str], predict_h: bool) -> Dict[str, torch.Tensor]:
    target = smooth_targets(batch["targets"], args.label_smoothing)
    field_loss = relative_field_loss(pred, target)
    pred_physical = pred * batch["scale"].view(-1, 1, 1, 1, 1)
    residuals = maxwell_residuals(pred_physical, batch["eps"], batch["omega"], batch["coords"], supervise_components, predict_h, args.time_convention)
    total = args.field_weight * field_loss + args.curl_e_weight * residuals["curl_e"] + args.curl_h_weight * residuals["curl_h"] + args.div_weight * residuals["div"]
    return {"total": total, "field": field_loss.detach(), "curl_e": residuals["curl_e"].detach(), "curl_h": residuals["curl_h"].detach(), "div": residuals["div"].detach()}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler, device: torch.device, args: argparse.Namespace, supervise_components: Sequence[str], predict_h: bool, epoch: int) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "field": 0.0, "curl_e": 0.0, "curl_h": 0.0, "div": 0.0}
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        autocast = args.amp and device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=autocast):
            pred = model(batch["inputs"])
            losses = compute_losses(pred, batch, args, supervise_components, predict_h)
            loss = losses["total"]
        if scaler is not None and scaler.is_enabled() and autocast:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["field"] += float(losses["field"].cpu())
        totals["curl_e"] += float(losses["curl_e"].cpu())
        totals["curl_h"] += float(losses["curl_h"].cpu())
        totals["div"] += float(losses["div"].cpu())
        if step % args.log_every == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={totals['loss']/step:.6f} field={totals['field']/step:.6f} curlE={totals['curl_e']/step:.6f} curlH={totals['curl_h']/step:.6f} div={totals['div']/step:.6f}")
    count = max(1, len(loader))
    return {k: v / count for k, v in totals.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace, supervise_components: Sequence[str], predict_h: bool) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "field": 0.0, "curl_e": 0.0, "curl_h": 0.0, "div": 0.0}
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch["inputs"])
        losses = compute_losses(pred, batch, args, supervise_components, predict_h)
        totals["loss"] += float(losses["total"].cpu())
        totals["field"] += float(losses["field"].cpu())
        totals["curl_e"] += float(losses["curl_e"].cpu())
        totals["curl_h"] += float(losses["curl_h"].cpu())
        totals["div"] += float(losses["div"].cpu())
    count = max(1, len(loader))
    return {k: v / count for k, v in totals.items()}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, best_val: float, args: argparse.Namespace, supervise_components: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "args": vars(args),
        "supervise_components": list(supervise_components),
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload["epoch"]) + 1, float(payload["best_val"]), payload["supervise_components"]


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.save_dir = args.save_dir.resolve()
    if args.patterns_mat is not None:
        args.patterns_mat = args.patterns_mat.resolve()
    if args.sampling_meta is not None:
        args.sampling_meta = args.sampling_meta.resolve()

    set_seed(args.seed)
    supervise_components = [c for c in str_list(args.supervise_components) if args.predict_h or not c.startswith("H")]
    if not supervise_components:
        raise ValueError("No supervised components selected.")

    train_loader, val_loader = build_dataloaders(args, supervise_components, args.predict_h)
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch["inputs"].shape[1]
    out_channels = 2 * len(supervise_components)
    args.in_channels = in_channels

    device = torch.device(args.device)
    model = FNO3d(in_channels, out_channels, args.width, args.hidden_channels, args.depth, args.modes_x, args.modes_y, args.modes_z).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    has_complex_params = any(p.is_complex() for p in model.parameters())
    scaler_enabled = args.amp and device.type == "cuda" and not has_complex_params
    if args.amp and device.type == "cuda" and has_complex_params:
        print("Detected complex parameters in the FNO; disabling GradScaler and keeping autocast only.")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None:
        start_epoch, best_val, supervise_components = load_checkpoint(args.resume, model, optimizer, scheduler)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, supervise_components, args.predict_h, epoch)
        val_metrics = evaluate(model, val_loader, device, args, supervise_components, args.predict_h)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"[epoch {epoch}] train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} val_field={val_metrics['field']:.6f} val_curlE={val_metrics['curl_e']:.6f} val_curlH={val_metrics['curl_h']:.6f} val_div={val_metrics['div']:.6f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, scheduler, epoch, best_val, args, supervise_components)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, scheduler, epoch, best_val, args, supervise_components)
        with (args.save_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    print(f"Training finished. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
