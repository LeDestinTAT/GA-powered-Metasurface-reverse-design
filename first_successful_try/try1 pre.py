# ==========================================================
# Predict full spectrum from a given 11x11 matrix
# 对应当前训练代码：
# - FNO encoder + lambda Fourier features + MLP head
# - checkpoint 中自动读取 config 和 lambda_vec
# - 兼容 PyTorch 2.6+ 的 torch.load(weights_only=False)
# ==========================================================

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ==========================================================
# 0) 这里改你的参数
# ==========================================================
CKPT_PATH = r"C:\Users\90740\Desktop\final\fno_peak_curve_best.pt"

pattern_array = np.array([
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,1,1,0],
    [0,1,1,0,0,0,0,0,1,1,0],
    [0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0],
    [0,1,1,0,0,0,0,0,1,1,0],
    [0,1,1,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)

PROJECT_PASSIVE = True
SAVE_NPY = False


# ==========================================================
# 1) 与训练代码一致的工具函数
# ==========================================================
def normalize_lambda(lam, lam_min, lam_max):
    return 2.0 * (lam - lam_min) / (lam_max - lam_min + 1e-12) - 1.0

def choose_gn_groups(width):
    for g in [8, 4, 2, 1]:
        if width % g == 0:
            return g
    return 1


# ==========================================================
# 2) 与训练代码一致的模型定义
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
        # z: (B, width)
        # lam_norm: (B, K, 1)
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
# 3) 加载 checkpoint
# ==========================================================
def load_model_from_ckpt(ckpt_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # PyTorch 2.6+ 兼容
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # 兼容旧版 PyTorch
        ckpt = torch.load(ckpt_path, map_location=device)

    if "config" not in ckpt:
        raise KeyError("checkpoint 中未找到 'config'")
    if "state_dict" not in ckpt:
        raise KeyError("checkpoint 中未找到 'state_dict'")
    if "lambda_vec" not in ckpt:
        raise KeyError("checkpoint 中未找到 'lambda_vec'")

    config = ckpt["config"]
    lambda_vec = np.array(ckpt["lambda_vec"]).astype(np.float32).reshape(-1)

    model = FNO_LambdaConditional_SParams(
        modes=config["MODES"],
        width=config["WIDTH"],
        depth=config["DEPTH"],
        lam_ff=config["LAM_FF"],
        head_hidden=config["HEAD_HIDDEN"]
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, lambda_vec, config, device


# ==========================================================
# 4) 被动性投影（可选）
#    保证 |S11|^2 + |S21|^2 <= 1
# ==========================================================
def project_to_passive(s11, s21, eps=1e-12):
    power = np.abs(s11) ** 2 + np.abs(s21) ** 2
    scale = np.ones_like(power, dtype=np.float32)

    mask = power > 1.0
    scale[mask] = 1.0 / np.sqrt(power[mask] + eps)

    s11_new = s11 * scale
    s21_new = s21 * scale
    return s11_new.astype(np.complex64), s21_new.astype(np.complex64)


# ==========================================================
# 5) 核心预测函数
# ==========================================================
@torch.no_grad()
def predict_spectrum_from_pattern(pattern_11x11, ckpt_path, project_passive=True, device=None):
    """
    输入：
        pattern_11x11: shape=(11,11)
        ckpt_path: 模型 .pt 路径
        project_passive: 是否做被动性投影
    输出：
        result: dict
            lambda_vec
            S11
            S21
            A
            raw_output
            config
    """
    model, lambda_vec, config, device = load_model_from_ckpt(ckpt_path, device=device)

    pattern = np.array(pattern_11x11, dtype=np.float32)
    if pattern.shape != (11, 11):
        raise ValueError(f"输入矩阵必须是 (11,11)，当前 shape={pattern.shape}")

    # 与训练一致：非零 -> 1，零 -> 0
    pattern = (pattern != 0).astype(np.float32)

    lam_min = float(lambda_vec.min())
    lam_max = float(lambda_vec.max())
    lam_n = normalize_lambda(lambda_vec, lam_min, lam_max).astype(np.float32)

    x = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0).to(device)       # (1,1,11,11)
    lam_n_t = torch.from_numpy(lam_n).unsqueeze(0).unsqueeze(-1).to(device)  # (1,M,1)

    pred = model.forward_curve(x, lam_n_t)   # (1,M,4)
    pred = pred.squeeze(0).cpu().numpy().astype(np.float32)

    s11 = pred[:, 0] + 1j * pred[:, 1]
    s21 = pred[:, 2] + 1j * pred[:, 3]

    if project_passive:
        s11, s21 = project_to_passive(s11, s21)

    A = 1.0 - np.abs(s11) ** 2 - np.abs(s21) ** 2
    A = A.astype(np.float32)

    return {
        "lambda_vec": lambda_vec,
        "S11": s11.astype(np.complex64),
        "S21": s21.astype(np.complex64),
        "A": A,
        "raw_output": pred,
        "config": config,
    }


# ==========================================================
# 6) 画图函数
# ==========================================================
def plot_predicted_curves(result, title_prefix="Prediction"):
    lam = result["lambda_vec"]
    s11 = result["S11"]
    s21 = result["S21"]
    A = result["A"]

    R = np.abs(s11) ** 2
    T = np.abs(s21) ** 2

    plt.figure(figsize=(7, 4))
    plt.plot(lam, A, label="A")
    plt.plot(lam, R, label="R=|S11|^2")
    plt.plot(lam, T, label="T=|S21|^2")
    plt.xlabel("lambda")
    plt.ylabel("value")
    plt.title(f"{title_prefix}: A / R / T")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(lam, np.real(s11), label="Re(S11)")
    plt.plot(lam, np.imag(s11), label="Im(S11)")
    plt.plot(lam, np.real(s21), label="Re(S21)")
    plt.plot(lam, np.imag(s21), label="Im(S21)")
    plt.xlabel("lambda")
    plt.ylabel("value")
    plt.title(f"{title_prefix}: complex S-parameters")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 7) 主程序示例
# ==========================================================
if __name__ == "__main__":
    result = predict_spectrum_from_pattern(
        pattern_11x11=pattern_array,
        ckpt_path=CKPT_PATH,
        project_passive=PROJECT_PASSIVE,
        device=None
    )

    lam = result["lambda_vec"]
    A = result["A"]
    s11 = result["S11"]
    s21 = result["S21"]

    print("lambda_vec shape =", lam.shape)
    print("A shape =", A.shape)
    print("S11 shape =", s11.shape)
    print("S21 shape =", s21.shape)
    print("A min/max =", float(A.min()), float(A.max()))

    plot_predicted_curves(result, title_prefix="Given 11x11 pattern")

    if SAVE_NPY:
        np.save("pred_lambda.npy", lam)
        np.save("pred_A.npy", A)
        np.save("pred_S11.npy", s11)
        np.save("pred_S21.npy", s21)
        print("已保存 pred_lambda.npy / pred_A.npy / pred_S11.npy / pred_S21.npy")