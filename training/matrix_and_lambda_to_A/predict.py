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
SPARAMS_MAT_PATH = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) 只预测一个指定 11×11 矩阵
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
# 3) .mat 自动读取（v7.3 / 非v7.3）——修正版
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
        # 非v7.3 需要 scipy
        from scipy.io import loadmat
        out = loadmat(path)
        return {k: v for k, v in out.items() if not k.startswith("__")}

def normalize_lambda(lam, lam_min, lam_max):
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
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B,C,H,W//2+1)
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, x_ft.shape[2], x_ft.shape[3],
                             dtype=torch.complex64, device=x.device)
        weight = torch.complex(self.weight_real, self.weight_imag)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            weight
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x

class LambdaFourierFeatures(nn.Module):
    def __init__(self, n_freq=16):
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
    def __init__(self, modes=6, width=64, depth=4, lam_ff=16, head_hidden=256):
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

    def forward(self, x, lam_norm):
        z = self.encoder(x)
        le = self.lam_embed(lam_norm)
        return self.head(torch.cat([z, le], dim=-1))

# =========================
# 5) 从 ckpt 读取 config 并构造模型（关键修正：默认值要与训练一致）
# =========================
def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        cfg = ckpt.get("config", {})
    else:
        state = ckpt
        cfg = {}

    # ⚠️默认值要与你训练脚本一致：LAM_FF 训练时是16，就默认16
    modes = int(cfg.get("MODES", 6))
    width = int(cfg.get("WIDTH", 64))
    depth = int(cfg.get("DEPTH", 4))
    lam_ff = int(cfg.get("LAM_FF", 16))
    head_hidden = int(cfg.get("HEAD_HIDDEN", 256))

    model = FNO_LambdaConditional_SParams(
        modes=modes, width=width, depth=depth, lam_ff=lam_ff, head_hidden=head_hidden
    ).to(device)

    # 严格加载（如果结构不匹配，会直接报错提醒你）
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[模型加载成功] modes={modes}, width={width}, depth={depth}, lam_ff={lam_ff}, head_hidden={head_hidden}")
    return model

# =========================
# 6) 预测函数（增加可选|S|裁剪，显示更物理更稳定）
# =========================
@torch.no_grad()
def predict_one_matrix(model, binary_matrix_11x11, lambda_vec, device,
                       clamp_mag=True, clamp_eps=1e-6, smooth_A_k=0):
    """
    返回：
      S11_pred: (M,) complex64
      S21_pred: (M,) complex64
      A_pred:   (M,) float32
    """
    model.eval()
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()
    lam_min, lam_max = float(lambda_vec.min()), float(lambda_vec.max())

    x = torch.from_numpy(binary_matrix_11x11.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,11,11)
    lam = torch.from_numpy(lambda_vec).unsqueeze(1)  # (M,1)
    lam_n = normalize_lambda(lam, lam_min, lam_max).to(device)  # (M,1) torch

    # encoder一次 + head批量
    z = model.encoder(x)                 # (1,width)
    z = z.repeat(lam_n.size(0), 1)       # (M,width)
    le = model.lam_embed(lam_n)          # (M,2*ff)
    out = model.head(torch.cat([z, le], dim=-1))  # (M,4)

    out = out.detach().cpu().numpy().astype(np.float32)
    S11 = out[:, 0] + 1j * out[:, 1]
    S21 = out[:, 2] + 1j * out[:, 3]

    # 可选：限制 |S|<=1，避免 A 出界导致“看起来很毛”
    if clamp_mag:
        mag11 = np.abs(S11)
        mag21 = np.abs(S21)
        scale11 = np.minimum(1.0, 1.0 / (mag11 + clamp_eps))
        scale21 = np.minimum(1.0, 1.0 / (mag21 + clamp_eps))
        S11 = S11 * scale11
        S21 = S21 * scale21

    A = 1.0 - np.abs(S11)**2 - np.abs(S21)**2
    A = A.astype(np.float32)

    # 可选：仅用于显示的轻微平滑
    if smooth_A_k and smooth_A_k > 1:
        k = int(smooth_A_k)
        pad = k // 2
        A_pad = np.pad(A, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float32) / k
        A = np.convolve(A_pad, kernel, mode="valid").astype(np.float32)

    return S11.astype(np.complex64), S21.astype(np.complex64), A

# =========================
# 7) 主流程：读lambda → 加载模型 → 预测 → 画图
# =========================
sp = load_mat_auto(SPARAMS_MAT_PATH)
if "lambda_vec" not in sp:
    raise KeyError(f"找不到 lambda_vec，mat里变量有：{list(sp.keys())}")

lambda_vec = np.array(sp["lambda_vec"]).squeeze().astype(np.float32)

model = load_model_from_ckpt(CKPT_PATH, DEVICE)

S11_pred, S21_pred, A_pred = predict_one_matrix(
    model, binary_matrix, lambda_vec, DEVICE,
    clamp_mag=True,     # 推荐开
    smooth_A_k=0        # 如果你只想看更平滑的图，可以设5或7
)

peak_idx = int(np.argmax(A_pred))
print(f"吸收峰值 Amax={float(A_pred[peak_idx]):.4f} @ lambda={float(lambda_vec[peak_idx]):.6f}")

plt.figure(figsize=(7,4))
plt.plot(lambda_vec, A_pred, label="Pred A(λ)")
plt.xlabel("lambda")
plt.ylabel("Absorption A")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 画 |S11|, |S21|
plt.figure(figsize=(7,4))
plt.plot(lambda_vec, np.abs(S11_pred), label="|S11|")
plt.plot(lambda_vec, np.abs(S21_pred), label="|S21|")

plt.xlabel("lambda")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
