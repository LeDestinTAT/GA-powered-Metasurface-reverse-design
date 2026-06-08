# ==========================================================
# predict_spectrum_param_fourier2d_dwconv_curvestyle.py
# 11x11 -> full spectrum prediction (S11,S21 complex + Absorption)
# Architecture: Fourier2D_LambdaConditional_SParams (encoder + lam_embed + head)
#
# ✅ Updated to match your optimized training code:
#   - Encoder pointwise branch = DWConv3×3 + 1×1 (Depthwise Separable Conv)  [方案2]
#   - Optional: GroupNorm / BatchNorm auto-detected from checkpoint
#   - STFT curve-loss only affects training, inference unchanged
#
# ✅ Auto infer hyper-params from state_dict (no manual model params)
# ✅ Lambda normalization: [-1,1] (same as training)
# ✅ FFT forced fp32 inside FourierOperator2d to avoid 11x11 fp16 cuFFT issues
# ==========================================================

import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib.pyplot as plt

# ------------------------------
# 0) 参数区：只改这里
# ------------------------------
CFG = {
    # 建议指向你“优化训练版”保存的 best：
    # best_fourier2d_dwconv_stft_curve.pt  或你自己的 best 名称
    "ckpt_path": r"C:\Users\90740\Desktop\final\best_fourier2d_dwconv_stft_curve.pt",

    # pattern 输入方式（二选一）
    "pattern_npy": None,
    "pattern_array": np.array([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ], dtype=np.float32),

    # lambda 输入
    "lambda_mode": "linspace",  # "linspace" or "array"
    "lambda_linspace": {"start": 3.0, "stop": 12.0, "num": 100},
    "lambda_array": None,

    # device / amp
    "device": "cuda",
    "use_amp": True,

    # 后处理
    "clamp_mag": True,
    "clamp_eps": 1e-6,

    # 输出
    "save_csv": False,
    "csv_path": r"pred_fourier2d_dwconv.csv",

    # 绘图
    "plot": {
        "lam_label": "Wavelength",
        "lam_unit": "(μm)",
        "title": "Predicted Spectrum (Fourier2D + DWConv, encoder+head)",
        "show": True,
        "save_path": None,
    },
}

# ------------------------------
# 1) 绘图/保存
# ------------------------------
def plot_spectrum(lam, s11, s21, A, lam_label="Wavelength", lam_unit="",
                  title="Predicted Spectrum", show=True, save_path=None):
    lam = lam.reshape(-1)
    R = np.abs(s11) ** 2
    T = np.abs(s21) ** 2

    plt.figure(figsize=(7, 4.5))
    plt.plot(lam, R, label=r"$|S_{11}|^2$ (Reflection)", linewidth=2)
    plt.plot(lam, T, label=r"$|S_{21}|^2$ (Transmission)", linewidth=2)
    plt.plot(lam, A, label=r"$A$ (Absorption)", linewidth=2)

    plt.xlabel(f"{lam_label} {lam_unit}".strip())
    plt.ylabel("Response")
    plt.title(title)

    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved figure] {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def save_csv(path, lam, s11, s21, A):
    data = np.stack([lam, s11.real, s11.imag, s21.real, s21.imag, A], axis=1)
    header = "lambda,s11_real,s11_imag,s21_real,s21_imag,absorption"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"[Saved] {path}")

# ------------------------------
# 2) 与训练一致：lambda 归一化到 [-1,1]
# ------------------------------
def normalize_lambda_pm1(lam: np.ndarray, lam_min: float, lam_max: float) -> np.ndarray:
    lam = lam.astype(np.float32)
    denom = (lam_max - lam_min) if (lam_max - lam_min) != 0 else 1.0
    return 2.0 * (lam - lam_min) / denom - 1.0

# ------------------------------
# 3) 模型定义（与训练一致：encoder + lam_embed + head）
#    ✅ 支持 DWConv pointwise（方案2） + 可选 GN/BN
# ------------------------------
class FourierOperator2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def forward(self, x):
        # FFT 强制 fp32：避免 cuFFT fp16 对 11x11 的限制
        orig_dtype = x.dtype
        if x.is_cuda:
            with autocast("cuda", enabled=False):
                x32 = x.float()
                x_ft = torch.fft.rfft2(x32, norm="ortho")
                out_ft = torch.zeros(
                    x_ft.shape[0], self.out_channels, x_ft.shape[2], x_ft.shape[3],
                    dtype=torch.complex64, device=x.device
                )
                weight = torch.complex(self.weight_real.float(), self.weight_imag.float()).to(dtype=torch.complex64)
                m1 = min(self.modes1, x_ft.shape[2])
                m2 = min(self.modes2, x_ft.shape[3])
                out_ft[:, :, :m1, :m2] = torch.einsum(
                    "bixy,ioxy->boxy",
                    x_ft[:, :, :m1, :m2],
                    weight[:, :, :m1, :m2]
                )
                y32 = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
            return y32.to(dtype=orig_dtype)
        else:
            x_ft = torch.fft.rfft2(x.float(), norm="ortho")
            out_ft = torch.zeros(
                x_ft.shape[0], self.out_channels, x_ft.shape[2], x_ft.shape[3],
                dtype=torch.complex64, device=x.device
            )
            weight = torch.complex(self.weight_real.float(), self.weight_imag.float()).to(dtype=torch.complex64)
            m1 = min(self.modes1, x_ft.shape[2])
            m2 = min(self.modes2, x_ft.shape[3])
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :m1, :m2],
                weight[:, :, :m1, :m2]
            )
            y32 = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
            return y32.to(dtype=orig_dtype)

class LambdaFourierFeatures(nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq) * np.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam_norm):
        x = lam_norm * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class DWConvBlock(nn.Module):
    """Depthwise 3×3 + Pointwise 1×1"""
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))

def _pick_gn_groups(width: int, preferred: int) -> int:
    g = int(preferred)
    if g <= 0:
        g = 1
    if width % g == 0:
        return g
    for gg in [32, 16, 8, 4, 2, 1]:
        if gg <= width and width % gg == 0:
            return gg
    return 1

class Fourier2DEncoder(nn.Module):
    """
    输出: (B,width) ——最后 mean(H,W)
    支持：
      - pointwise: DWConvBlock 或 Conv1x1
      - out_norm: BatchNorm2d 或 GroupNorm
    """
    def __init__(self, modes=6, width=192, depth=4,
                 use_dwconv=True,
                 norm_type="bn", gn_groups=8):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.spectral = nn.ModuleList([FourierOperator2d(width, width, modes, modes) for _ in range(depth)])

        if use_dwconv:
            self.pointwise = nn.ModuleList([DWConvBlock(width) for _ in range(depth)])
        else:
            self.pointwise = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])

        self.act = nn.GELU()

        if norm_type.lower() == "gn":
            g = _pick_gn_groups(width, gn_groups)
            self.out_norm = nn.GroupNorm(num_groups=g, num_channels=width)
        else:
            self.out_norm = nn.BatchNorm2d(width)

    def forward(self, x):
        x = self.in_proj(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = self.act(spec(x) + pw(x))
        x = self.out_norm(x)
        return x.mean(dim=(-2, -1))

class Fourier2D_LambdaConditional_SParams(nn.Module):
    def __init__(self, modes=6, width=192, depth=4, lam_ff=8, head_hidden=256,
                 use_dwconv=True, norm_type="bn", gn_groups=8):
        super().__init__()
        self.encoder = Fourier2DEncoder(
            modes=modes, width=width, depth=depth,
            use_dwconv=use_dwconv, norm_type=norm_type, gn_groups=gn_groups
        )
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)
        head_in = width + 2 * lam_ff
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 4),
        )

    def forward(self, x, lam_norm):
        z = self.encoder(x)
        le = self.lam_embed(lam_norm)
        return self.head(torch.cat([z, le], dim=-1))

# ------------------------------
# 4) 读取 checkpoint + 自动推断模型超参（不需要手动写）
# ------------------------------
def load_checkpoint(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    # 纯 state_dict
    if isinstance(ckpt, dict) and ("encoder.in_proj.weight" in ckpt):
        return {"state_dict": ckpt, "config": None}
    # 包装 dict（你的训练保存就是这种）
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return {"state_dict": ckpt["state_dict"], "config": ckpt.get("config", None)}
    raise ValueError(f"Unrecognized checkpoint format at {path}")

def infer_model_cfg_from_state_dict(sd: dict, config: dict | None) -> dict:
    # width
    width = int(sd["encoder.in_proj.weight"].shape[0])

    # depth: encoder.spectral.{i}.weight_real
    depth = len([k for k in sd.keys() if re.match(r"encoder\.spectral\.\d+\.weight_real$", k)])
    if depth <= 0:
        raise KeyError("state_dict中未找到 encoder.spectral.*.weight_real（确认 ckpt 是否为 encoder+head 版本）")

    # modes
    modes = int(sd["encoder.spectral.0.weight_real"].shape[-1])

    # lam_ff
    lam_ff = int(sd["lam_embed.freqs"].numel())

    # head_hidden
    head_hidden = int(sd["head.0.weight"].shape[0])

    # ✅ DWConv 检测：是否存在 encoder.pointwise.0.dw.weight
    use_dwconv = any(re.match(r"encoder\.pointwise\.0\.dw\.weight$", k) for k in sd.keys())

    # ✅ Norm 检测：BatchNorm 会有 running_mean；GroupNorm 没有
    is_bn = any(re.match(r"encoder\.out_norm\.running_mean$", k) for k in sd.keys())
    norm_type = "bn" if is_bn else "gn"

    # gn_groups：优先从 ckpt["config"] 拿（你的训练保存里有）
    gn_groups = 8
    if isinstance(config, dict):
        # 你保存的 config 是 {**CFG["model"], **CFG["train"], ...} 合并后的
        if "gn_groups" in config:
            try:
                gn_groups = int(config["gn_groups"])
            except Exception:
                gn_groups = 8

    return dict(
        modes=modes, width=width, depth=depth,
        lam_ff=lam_ff, head_hidden=head_hidden,
        use_dwconv=use_dwconv, norm_type=norm_type, gn_groups=gn_groups
    )

@torch.no_grad()
def predict_full_S_and_A_like_training(model, pattern_11x11, lambda_vec, device,
                                       clamp_mag=True, clamp_eps=1e-6, amp_on=False):
    """
    完全按训练代码的推理逻辑：
      z = encoder(x) once
      z repeat over all lambda
      le = lam_embed(lam_n)
      out = head([z, le])
    """
    model.eval()
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()

    lam_min, lam_max = float(lambda_vec.min()), float(lambda_vec.max())
    lam_n = normalize_lambda_pm1(lambda_vec, lam_min, lam_max).astype(np.float32)  # (M,)

    x = torch.from_numpy(pattern_11x11.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,11,11)
    lam_n_t = torch.from_numpy(lam_n).unsqueeze(1).to(device)                                     # (M,1)

    with autocast("cuda", enabled=amp_on and x.is_cuda):
        z = model.encoder(x)                     # (1,width)
        z = z.repeat(lam_n_t.size(0), 1)         # (M,width)
        le = model.lam_embed(lam_n_t)            # (M,2*lam_ff)
        out = model.head(torch.cat([z, le], dim=-1))  # (M,4)

    out = out.detach().cpu().numpy().astype(np.float32)

    s11 = out[:, 0] + 1j * out[:, 1]
    s21 = out[:, 2] + 1j * out[:, 3]

    if clamp_mag:
        s11 = s11 * np.minimum(1.0, 1.0 / (np.abs(s11) + clamp_eps))
        s21 = s21 * np.minimum(1.0, 1.0 / (np.abs(s21) + clamp_eps))

    A = (1.0 - np.abs(s11) ** 2 - np.abs(s21) ** 2).astype(np.float32)
    return s11.astype(np.complex64), s21.astype(np.complex64), A

# ------------------------------
# 5) 主预测入口（保持你原脚本接口/输出）
# ------------------------------
@torch.no_grad()
def predict_from_cfg(cfg: dict):
    # device
    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = "cpu"
    amp_on = bool(cfg["use_amp"]) and str(device).startswith("cuda")

    # pattern
    pat = np.load(cfg["pattern_npy"]) if cfg["pattern_npy"] is not None else cfg["pattern_array"]
    pat = np.asarray(pat, dtype=np.float32)
    assert pat.shape == (11, 11), f"pattern must be (11,11), got {pat.shape}"

    # lambda
    if cfg["lambda_mode"] == "linspace":
        p = cfg["lambda_linspace"]
        lam = np.linspace(float(p["start"]), float(p["stop"]), int(p["num"]), dtype=np.float32)
    elif cfg["lambda_mode"] == "array":
        lam = np.asarray(cfg["lambda_array"], dtype=np.float32).reshape(-1)
    else:
        raise ValueError("lambda_mode must be 'linspace' or 'array'")

    # load ckpt
    pack = load_checkpoint(cfg["ckpt_path"], map_location="cpu")
    sd = pack["state_dict"]
    ckpt_config = pack.get("config", None)

    # infer model cfg from sd + (optional) ckpt config
    mc = infer_model_cfg_from_state_dict(sd, ckpt_config)

    print("[Model inferred]", mc)

    model = Fourier2D_LambdaConditional_SParams(**mc).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # predict
    s11, s21, A = predict_full_S_and_A_like_training(
        model, pat, lam, device,
        clamp_mag=bool(cfg["clamp_mag"]),
        clamp_eps=float(cfg["clamp_eps"]),
        amp_on=amp_on,
    )
    return lam, s11, s21, A

if __name__ == "__main__":
    lam, s11, s21, A = predict_from_cfg(CFG)

    plot_cfg = CFG.get("plot", {})
    plot_spectrum(
        lam, s11, s21, A,
        lam_label=plot_cfg.get("lam_label", "Wavelength"),
        lam_unit=plot_cfg.get("lam_unit", ""),
        title=plot_cfg.get("title", "Predicted Spectrum"),
        show=bool(plot_cfg.get("show", True)),
        save_path=plot_cfg.get("save_path", None),
    )

    print("[Done] Predicted:", len(lam), "points")
    print("A range:", float(A.min()), float(A.max()))
    if CFG["save_csv"]:
        save_csv(CFG["csv_path"], lam, s11, s21, A)
