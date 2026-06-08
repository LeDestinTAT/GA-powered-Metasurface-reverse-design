import os
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt

# =========================
# 1) 你需要修改的路径
# =========================
CKPT_PATH = r"C:\Users\90740\Desktop\final\fno_sparams_curve.pt"
# 或者用 best 模型：
CKPT_PATH = r"C:\Users\90740\Desktop\final\best_fno_sparams_curve.pt"

SPARAMS_MAT_PATH = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) 输入：一个指定 11×11 二值矩阵
# =========================
binary_matrix = np.array([
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,1,1,0],
    [0,1,1,1,0,0,0,1,1,1,0],
    [0,0,1,1,1,0,1,1,1,0,0],
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,0,1,1,1,0,0],
    [0,1,1,1,0,0,0,1,1,1,0],
    [0,1,1,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
assert binary_matrix.shape == (11, 11), f"输入必须是(11,11)，现在是 {binary_matrix.shape}"

# =========================
# 3) .mat 自动读取（v7.3 / 非v7.3）
# =========================
def load_mat_auto(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    if os.path.getsize(path) < 1024:
        raise OSError(f"文件过小/损坏/路径指错：{path}")

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

def normalize_lambda(lam, lam_min, lam_max):
    # 映射到 [-1, 1]
    return 2.0 * (lam - lam_min) / (lam_max - lam_min) - 1.0

# =========================
# 4) 模型结构（与训练一致）
# =========================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2))

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
        freqs = 2.0 ** torch.arange(n_freq) * np.pi
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
        self.out_norm = nn.BatchNorm2d(width)

    def forward(self, x):
        x = self.in_proj(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = self.act(spec(x) + pw(x))
        x = self.out_norm(x)
        return x.mean(dim=(-2, -1))  # (B,width)

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
            nn.Linear(head_hidden, 4)  # [ReS11, ImS11, ReS21, ImS21]
        )

    def forward(self, x, lam_norm):
        z = self.encoder(x)
        le = self.lam_embed(lam_norm)
        return self.head(torch.cat([z, le], dim=-1))

def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError("ckpt格式不对：需要包含 state_dict 和 config")

    state = ckpt["state_dict"]
    cfg = ckpt.get("config", {})

    # 你训练保存config是混合字典，所以这里优先读小写键名
    modes = int(cfg.get("modes", cfg.get("MODES", 6)))
    width = int(cfg.get("width", cfg.get("WIDTH", 64)))
    depth = int(cfg.get("depth", cfg.get("DEPTH", 4)))
    lam_ff = int(cfg.get("lam_ff", cfg.get("LAM_FF", 8)))
    head_hidden = int(cfg.get("head_hidden", cfg.get("HEAD_HIDDEN", 256)))

    model = FNO_LambdaConditional_SParams(
        modes=modes, width=width, depth=depth, lam_ff=lam_ff, head_hidden=head_hidden
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[模型加载成功] modes={modes}, width={width}, depth={depth}, lam_ff={lam_ff}, head_hidden={head_hidden}")
    return model

# =========================
# 5) 预测：给一个结构，输出整条 S(λ) 与 A(λ)
# =========================
@torch.no_grad()
def predict_absorption_curve(model, binary_11x11, lambda_vec, device, clamp_mag=True, clamp_eps=1e-6):
    """
    返回：
      S11_pred: (M,) complex
      S21_pred: (M,) complex
      A_pred:   (M,) float
    """
    model.eval()
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()
    lam_min, lam_max = float(lambda_vec.min()), float(lambda_vec.max())

    x = torch.from_numpy(binary_11x11.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,11,11)
    lam = torch.from_numpy(lambda_vec).unsqueeze(1)  # (M,1)
    lam_n = normalize_lambda(lam, lam_min, lam_max).to(device)

    # 先encoder一次，再批量预测所有λ点
    z = model.encoder(x)                 # (1,width)
    z = z.repeat(lam_n.size(0), 1)       # (M,width)
    le = model.lam_embed(lam_n)          # (M,2*ff)
    out = model.head(torch.cat([z, le], dim=-1))  # (M,4)

    out = out.detach().cpu().numpy().astype(np.float32)
    S11 = out[:, 0] + 1j * out[:, 1]
    S21 = out[:, 2] + 1j * out[:, 3]

    # 可选：把 |S|>1 的点投影回去（让A更“物理”/更稳）
    if clamp_mag:
        mag11 = np.abs(S11)
        mag21 = np.abs(S21)
        S11 = S11 * np.minimum(1.0, 1.0 / (mag11 + clamp_eps))
        S21 = S21 * np.minimum(1.0, 1.0 / (mag21 + clamp_eps))

    A = 1.0 - (np.abs(S11) ** 2) - (np.abs(S21) ** 2)
    return S11.astype(np.complex64), S21.astype(np.complex64), A.astype(np.float32)

# =========================
# 6) 主流程：读lambda → 加载模型 → 预测 → 画图
# =========================
sp = load_mat_auto(SPARAMS_MAT_PATH)
if "lambda_vec" not in sp:
    raise KeyError(f"找不到 lambda_vec，mat里变量有：{list(sp.keys())}")
lambda_vec = np.array(sp["lambda_vec"]).squeeze().astype(np.float32)

model = load_model_from_ckpt(CKPT_PATH, DEVICE)

S11_pred, S21_pred, A_pred = predict_absorption_curve(
    model, binary_matrix, lambda_vec, DEVICE,
    clamp_mag=True   # 建议先 True，更稳、更物理
)

# --- 打印主峰信息（多峰你也可以自己找多个峰） ---
peak_idx = int(np.argmax(A_pred))
print(f"Amax={float(A_pred[peak_idx]):.4f} @ lambda={float(lambda_vec[peak_idx]):.4f}")

# --- 画吸收谱 ---
plt.figure(figsize=(7, 4))
plt.plot(lambda_vec, A_pred, label="Pred A(λ)")
plt.xlabel("lambda")
plt.ylabel("Absorption A")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 可选：画 |S11|, |S21| 看看是否超界 ---
plt.figure(figsize=(7, 4))
plt.plot(lambda_vec, np.abs(S11_pred), label="|S11|")
plt.plot(lambda_vec, np.abs(S21_pred), label="|S21|")
plt.axhline(1.0, linestyle="--")
plt.xlabel("lambda")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
