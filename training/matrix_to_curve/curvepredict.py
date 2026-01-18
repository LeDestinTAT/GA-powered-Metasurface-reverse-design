import os
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt

CKPT_PATH = r"C:\Users\90740\Desktop\final\curve_direct_A.pt"
SPARAMS_MAT_PATH = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_mat_auto(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
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

# 读 lambda_vec，确定 M
sp = load_mat_auto(SPARAMS_MAT_PATH)
if "lambda_vec" not in sp:
    raise KeyError(f"找不到 lambda_vec，mat里变量有：{list(sp.keys())}")
lambda_vec = np.array(sp["lambda_vec"]).squeeze().astype(np.float32)
M = int(lambda_vec.shape[0])

# ---------- 模型 ----------
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
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, x_ft.shape[2], x_ft.shape[3],
                             dtype=torch.complex64, device=x.device)
        weight = torch.complex(self.weight_real, self.weight_imag)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            weight
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")

class FNOEncoder(nn.Module):
    def __init__(self, modes=6, width=96, depth=5):
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

class CurveRegressor(nn.Module):
    def __init__(self, M, modes=6, width=96, depth=5, head_hidden=512):
        super().__init__()
        self.encoder = FNOEncoder(modes=modes, width=width, depth=depth)
        self.head = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, M)
        )

    def forward(self, x):
        return self.head(self.encoder(x))

# ---------- 加载 ckpt（兼容不同保存格式） ----------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

model = CurveRegressor(
    M=M,
    modes=int(cfg.get("MODES", 6)),
    width=int(cfg.get("WIDTH", 96)),
    depth=int(cfg.get("DEPTH", 5)),
    head_hidden=int(cfg.get("HEAD_HIDDEN", 512)),
).to(DEVICE)

state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
model.load_state_dict(state, strict=True)
model.eval()

# ---------- 预测 ----------
x = torch.from_numpy(binary_matrix).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,11,11)
with torch.no_grad():
    predA = model(x).detach().cpu().numpy().squeeze()

# ---------- 画图 ----------
plt.figure(figsize=(7,4))
plt.plot(lambda_vec, predA, label="Pred A(λ)")
plt.xlabel("lambda")
plt.ylabel("A")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
