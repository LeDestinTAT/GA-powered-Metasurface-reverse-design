import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

print("TensorBoard：终端运行  tensorboard --logdir runs  打开 http://localhost:6006/")

# ==========================================================
# 0) 配置（你主要改这些）
# ==========================================================
patterns_path = r"C:\Users\90740\Desktop\final\training_patterns_11x11.mat"
sparams_path  = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"

EPOCHS = 1000
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

# 只用有效样本前 N_USE 条（前800写800；用全部写 None）
N_USE = 800
EPS_VALID = 1e-12

# 模型结构（回归整条曲线）
MODES = 6
WIDTH = 96          # 曲线回归可以适当大一点
DEPTH = 5
HEAD_HIDDEN = 512

LR = 3e-4
WEIGHT_DECAY = 1e-4

# 曲线平滑正则（强烈建议开）
USE_SMOOTH = True
SMOOTH_W1 = 0.00     # 一阶差分
SMOOTH_W2 = 0.20     # 二阶差分，压“全带毛”很有效

# （可选）峰附近误差加权：让峰位/峰高更准
USE_PEAK_WEIGHT = True
PEAK_ALPHA = 2.0     # 1~5
PEAK_P = 2.0         # 1~3

# 可视化
LOG_ROOT = "runs/curve_direct"
PLOT_EVERY = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 1) 读取 .mat（v7.3 / 非v7.3）
# ==========================================================
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

def to_numpy_bool(mat_11x11xN):
    x = np.array(mat_11x11xN)
    x = (x != 0).astype(np.float32)
    return x

def load_complex_struct(arr):
    if hasattr(arr, "dtype") and arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    return arr.astype(np.complex64)

# ==========================================================
# 2) Dataset：一个结构 -> 一整条 A(λ)
# ==========================================================
class CurveDataset(Dataset):
    def __init__(self, patterns_11x11xN, A_by_sample):
        """
        patterns_11x11xN: (11,11,N)
        A_by_sample:      (N,M) float32
        """
        self.x = patterns_11x11xN
        self.A = A_by_sample.astype(np.float32)
        self.N = self.A.shape[0]
        self.M = self.A.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.x[:, :, i]  # (11,11)
        y = self.A[i, :]     # (M,)
        x = torch.from_numpy(x).unsqueeze(0)    # (1,11,11)
        y = torch.from_numpy(y)                 # (M,)
        return x, y

# ==========================================================
# 3) FNO Encoder（处理11x11） + 曲线Head（输出M点）
# ==========================================================
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
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x

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
        z = x.mean(dim=(-2, -1))  # (B,width)
        return z

class CurveRegressor(nn.Module):
    def __init__(self, M, modes=6, width=64, depth=4, head_hidden=256):
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
        z = self.encoder(x)   # (B,width)
        y = self.head(z)      # (B,M)
        return y

# ==========================================================
# 4) 曲线损失：整条曲线MSE + 平滑 +（可选）峰权重
# ==========================================================
def curve_loss(pred, true, use_smooth=True, w1=0.0, w2=0.2,
               use_peak_weight=True, alpha=2.0, p=2.0):
    """
    pred,true: (B,M)
    """
    if use_peak_weight:
        # 峰加权：A越大，权重越大（让峰位/峰高更准）
        w = 1.0 + alpha * (true.clamp(0,1) ** p)  # (B,M)
        mse = (w * (pred - true)**2).mean()
    else:
        mse = ((pred - true)**2).mean()

    loss = mse

    if use_smooth and pred.shape[1] >= 3:
        if w1 > 0:
            d1 = pred[:, 1:] - pred[:, :-1]
            loss = loss + w1 * (d1**2).mean()
        if w2 > 0:
            d2 = pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2]
            loss = loss + w2 * (d2**2).mean()

    return loss

# ==========================================================
# 5) 读数据：从 S11/S21 计算 A(λ) 作为曲线标签
# ==========================================================
patterns = load_mat_auto(patterns_path)
sp = load_mat_auto(sparams_path)

selected = to_numpy_bool(patterns["selected"])                 # (11,11,N_total)
lambda_vec = np.array(sp["lambda_vec"]).squeeze().astype(np.float32)  # (M,)

S11 = load_complex_struct(np.array(sp["S11_all"]))
S21 = load_complex_struct(np.array(sp["S21_all"]))

# 统一为 (N_total,M)
if S11.shape[0] == lambda_vec.shape[0]:
    S11 = S11.T
    S21 = S21.T

# 对齐 N
N_total = min(selected.shape[2], S11.shape[0], S21.shape[0])
selected = selected[:, :, :N_total]
S11 = S11[:N_total, :]
S21 = S21[:N_total, :]

valid = (np.any(np.abs(S11) > EPS_VALID, axis=1) |
         np.any(np.abs(S21) > EPS_VALID, axis=1))
idx_valid = np.where(valid)[0]
print("总样本数 =", N_total, "| 有效样本数 =", len(idx_valid))

if N_USE is not None:
    idx_valid = idx_valid[:N_USE]
    print("使用有效样本前", len(idx_valid), "条")

selected = selected[:, :, idx_valid]
S11 = S11[idx_valid, :]
S21 = S21[idx_valid, :]

A = (1.0 - np.abs(S11)**2 - np.abs(S21)**2).astype(np.float32)  # (N,M)

N = selected.shape[2]
M = A.shape[1]
print("最终训练样本 N =", N, "| 曲线长度 M =", M)

# ==========================================================
# 6) 划分 train/val
# ==========================================================
np.random.seed(0)
idx = np.random.permutation(N)
n_train = int(0.8 * N)
train_idx, val_idx = idx[:n_train], idx[n_train:]

x_train = selected[:, :, train_idx]
y_train = A[train_idx, :]
x_val   = selected[:, :, val_idx]
y_val   = A[val_idx, :]

train_ds = CurveDataset(x_train, y_train)
val_ds   = CurveDataset(x_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================================
# 7) 模型/优化器
# ==========================================================
model = CurveRegressor(M=M, modes=MODES, width=WIDTH, depth=DEPTH, head_hidden=HEAD_HIDDEN).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def eval_val(loader):
    model.eval()
    s, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = curve_loss(pred, y, USE_SMOOTH, SMOOTH_W1, SMOOTH_W2,
                          USE_PEAK_WEIGHT, PEAK_ALPHA, PEAK_P)
        bs = x.size(0)
        s += float(loss.item()) * bs
        n += bs
    return s / max(n,1)

# ==========================================================
# 8) 训练 + TensorBoard（精简）
# ==========================================================
run_name = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"{LOG_ROOT}/{run_name}")

VIS_SID = 0
train_hist, val_hist = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    s, n = 0.0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = curve_loss(pred, y, USE_SMOOTH, SMOOTH_W1, SMOOTH_W2,
                          USE_PEAK_WEIGHT, PEAK_ALPHA, PEAK_P)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        s += float(loss.item()) * bs
        n += bs

    train_loss = s / max(n,1)
    val_loss = eval_val(val_loader)

    train_hist.append(train_loss)
    val_hist.append(val_loss)

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    if epoch % PLOT_EVERY == 0 or epoch == 1:
        model.eval()
        x0 = torch.from_numpy(x_val[:, :, VIS_SID]).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,11,11)
        pred0 = model(x0).detach().cpu().numpy().squeeze()
        true0 = y_val[VIS_SID, :]

        fig = plt.figure(figsize=(7,4))
        plt.plot(lambda_vec, true0, label="true A")
        plt.plot(lambda_vec, pred0, "--", label="pred A")
        plt.xlabel("lambda")
        plt.ylabel("A")
        plt.title(f"Epoch {epoch}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        writer.add_figure("viz/val0_A", fig, epoch)
        plt.close(fig)

    print(f"Epoch {epoch:03d} | train={train_loss:.6e} | val={val_loss:.6e}")

SAVE_PATH = r"C:\Users\90740\Desktop\final\curve_direct_A.pt"
torch.save({
    "state_dict": model.state_dict(),
    "config": {
        "M": M, "MODES": MODES, "WIDTH": WIDTH, "DEPTH": DEPTH, "HEAD_HIDDEN": HEAD_HIDDEN,
        "USE_SMOOTH": USE_SMOOTH, "SMOOTH_W1": SMOOTH_W1, "SMOOTH_W2": SMOOTH_W2,
        "USE_PEAK_WEIGHT": USE_PEAK_WEIGHT, "PEAK_ALPHA": PEAK_ALPHA, "PEAK_P": PEAK_P
    }
}, SAVE_PATH)

writer.close()

plt.figure(figsize=(7,4))
plt.plot(train_hist, label="train")
plt.plot(val_hist, label="val")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("curve_direct_loss.png", dpi=200)
plt.show()

print(f"训练完成：已保存模型 {SAVE_PATH} 和 curve_direct_loss.png")
