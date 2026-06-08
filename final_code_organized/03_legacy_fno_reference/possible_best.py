# ==========================================================
# 曲线段训练版（S参数复数监督）：
# - 每个样本 = 一个11x11结构 + K个λ点（同一结构内）
# - 主损失：K点整体S误差（向量loss）
# - 曲线一致性：只在同结构K点内部做二阶差分平滑（抑制“全带毛”）
# - 可选：偏峰采样（只影响“同结构内选哪些λ点”）
# - TensorBoard 精简：只记录 train/val 总loss + lr + 每隔PLOT_EVERY画一张 A(λ) 图
# ==========================================================

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
# 0) 配置（你最常改的参数）
# ==========================================================
patterns_path = r"C:\Users\90740\Desktop\final\training_patterns_11x11.mat"
sparams_path  = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"

EPOCHS = 300
BATCH_SIZE = 64           # 曲线段训练每个样本含K点，显存更吃紧；建议先用64/32
VAL_BATCH_SIZE = 64
K_LAM_PER_SAMPLE = 32     # 每个结构每次抽K个λ点（曲线段长度）建议16~64

# 只用有效样本前 N_USE 条（前800写800；用全部写 None）
N_USE = 800

# 每隔多少 epoch 做一次可视化（写入一张图）
PLOT_EVERY = 10

# 判定有效样本（S参数非全0）的阈值
EPS_VALID = 1e-12

# ——偏峰采样（只在“同一结构内选哪些λ点”）
PEAK_SAMPLING = True
GAMMA_PEAK_SAMPLING = 1.5  # 1.2~2.0，越大越偏峰；太大可能导致非峰段毛

# ——曲线一致性（只在同结构内K点做差分）
USE_SMOOTHNESS = True
SMOOTH_W2 = 0.10           # 二阶差分权重：0.05~0.2（先0.1）
SMOOTH_W1 = 0.00           # 一阶差分一般不必；可设0

# ——峰加权辅助损失（仍然是同结构K点内计算）
USE_PEAK_AUX_LOSS = True
ALPHA_PEAK_WEIGHT = 5.0    # 2~10
P_PEAK_WEIGHT = 2.0        # 1~3
LAMBDA_AUX = 0.5           # 0.1~2（辅助项占比）

# ——幅值约束（防止 |S|>1 导致A乱跳/毛）
USE_MAG_PENALTY = True
MAG_W = 0.05               # 0.01~0.1

# ——模型结构
MODES = 6
WIDTH = 64
DEPTH = 4
LAM_FF = 8                 # 强烈建议从8开始；毛刺明显时别用16
HEAD_HIDDEN = 256

# ——优化器
LR = 3e-4                  # 你原先1e-3较激进；建议先3e-4更稳
WEIGHT_DECAY = 1e-4

# ——TensorBoard 日志目录（自动分run）
LOG_ROOT = "runs/fno_sparams_curve"

# ==========================================================
# 1) 自动兼容读取 .mat（v7.3 / 非v7.3）
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
    """
    兼容 MATLAB 把复数存为结构体:
      dtype=[('real','<f8'),('imag','<f8')]
    """
    if hasattr(arr, "dtype") and arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    return arr.astype(np.complex64)

def normalize_lambda(lam, lam_min, lam_max):
    return 2.0 * (lam - lam_min) / (lam_max - lam_min) - 1.0

# ==========================================================
# 2) Dataset：每个样本返回“同一结构的K个λ点”（曲线段）
# ==========================================================
class CurveSegmentSParamsDataset(Dataset):
    """
    每个样本 = 一个结构 + K个λ点（同一结构内）
    返回：
      x:     (1,11,11)
      lam_n: (K,1)
      yS:    (K,4) = [ReS11, ImS11, ReS21, ImS21]
      a_true:(K,1) 仅用于加权/可视化，不是主监督
    """
    def __init__(self, patterns_11x11xN, lambda_vec, S11_by_sample, S21_by_sample,
                 k_lam_per_sample=32, peak_sampling=True, gamma=1.5):
        self.x = patterns_11x11xN            # (11,11,N)
        self.lam_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()  # (M,)
        self.S11 = S11_by_sample             # (N,M) complex
        self.S21 = S21_by_sample             # (N,M) complex
        self.N, self.M = self.S11.shape
        self.K = int(k_lam_per_sample)

        self.lam_min = float(self.lam_vec.min())
        self.lam_max = float(self.lam_vec.max())

        # 真值吸收，仅用于采样与权重（不作为主监督）
        R = np.abs(self.S11)**2
        T = np.abs(self.S21)**2
        self.A = (1.0 - R - T).astype(np.float32)  # (N,M)

        self.peak_sampling = peak_sampling
        self.gamma = float(gamma)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.x[:, :, i].astype(np.float32)  # (11,11)

        # 选择该结构的K个λ点
        if self.peak_sampling:
            eps = 1e-6
            a = self.A[i, :]
            prob = (np.clip(a, 0, 1) + eps) ** self.gamma
            prob = prob / prob.sum()
            idx = np.random.choice(self.M, size=self.K, replace=False, p=prob)
        else:
            idx = np.random.choice(self.M, size=self.K, replace=False)

        idx = np.sort(idx)  # ✅关键：排序后才有“曲线一致性”的含义

        lam = self.lam_vec[idx]  # (K,)
        lam_n = normalize_lambda(lam, self.lam_min, self.lam_max).astype(np.float32)  # (K,)

        s11 = self.S11[i, idx]
        s21 = self.S21[i, idx]

        yS = np.stack([np.real(s11), np.imag(s11), np.real(s21), np.imag(s21)], axis=-1).astype(np.float32)  # (K,4)
        a_true = self.A[i, idx].astype(np.float32).reshape(-1, 1)  # (K,1)

        # torch
        x = torch.from_numpy(x).unsqueeze(0)            # (1,11,11)
        lam_n = torch.from_numpy(lam_n).unsqueeze(1)    # (K,1)
        yS = torch.from_numpy(yS)                       # (K,4)
        a_true = torch.from_numpy(a_true)               # (K,1)
        return x, lam_n, yS, a_true

# ==========================================================
# 3) 模型：FNO Encoder + λ Fourier Features + MLP Head（输出4维）
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
    def __init__(self, n_freq=8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq) * np.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam_norm):  # (B,1)
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
            nn.Linear(head_hidden, 4)
        )

    def forward(self, x, lam_norm):
        z = self.encoder(x)
        le = self.lam_embed(lam_norm)
        return self.head(torch.cat([z, le], dim=-1))

# ==========================================================
# 4) 推理：给定结构预测整条 S(λ) 并转成 A(λ)（用于可视化）
# ==========================================================
@torch.no_grad()
def predict_full_S_and_A(model, pattern_11x11, lambda_vec, device, clamp_mag=True, clamp_eps=1e-6):
    model.eval()
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()
    lam_min, lam_max = float(lambda_vec.min()), float(lambda_vec.max())

    x = torch.from_numpy(pattern_11x11.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,11,11)
    lam = torch.from_numpy(lambda_vec).unsqueeze(1)  # (M,1)
    lam_n = normalize_lambda(lam, lam_min, lam_max).to(device)

    # encoder一次 + head批量
    z = model.encoder(x)                 # (1,width)
    z = z.repeat(lam_n.size(0), 1)       # (M,width)
    le = model.lam_embed(lam_n)          # (M,2*ff)
    out = model.head(torch.cat([z, le], dim=-1))  # (M,4)

    out = out.detach().cpu().numpy().astype(np.float32)
    s11 = out[:, 0] + 1j * out[:, 1]
    s21 = out[:, 2] + 1j * out[:, 3]

    if clamp_mag:
        mag11 = np.abs(s11)
        mag21 = np.abs(s21)
        s11 = s11 * np.minimum(1.0, 1.0 / (mag11 + clamp_eps))
        s21 = s21 * np.minimum(1.0, 1.0 / (mag21 + clamp_eps))

    A = (1.0 - np.abs(s11)**2 - np.abs(s21)**2).astype(np.float32)
    return s11.astype(np.complex64), s21.astype(np.complex64), A

def make_A_figure(lambda_vec, true_A, pred_A, title="A(λ)"):
    fig = plt.figure(figsize=(6.5,4.2))
    plt.plot(lambda_vec, true_A, label="true A")
    plt.plot(lambda_vec, pred_A, "--", label="pred A")
    plt.xlabel("lambda")
    plt.ylabel("A")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig

# ==========================================================
# 5) 读取数据 + 过滤无效样本 + 可选截断前N_USE有效样本
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

# 过滤无效样本（全0）
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

N = selected.shape[2]
M = lambda_vec.shape[0]
print("最终训练样本 N =", N, "| 谱点 M =", M)

# ==========================================================
# 6) 划分 train/val + DataLoader（曲线段Dataset）
# ==========================================================
np.random.seed(0)
idx = np.random.permutation(N)
n_train = int(0.8 * N)
train_idx, val_idx = idx[:n_train], idx[n_train:]

x_train = selected[:, :, train_idx]
x_val   = selected[:, :, val_idx]

S11_train = S11[train_idx, :]
S21_train = S21[train_idx, :]
S11_val   = S11[val_idx, :]
S21_val   = S21[val_idx, :]

train_ds = CurveSegmentSParamsDataset(
    x_train, lambda_vec, S11_train, S21_train,
    k_lam_per_sample=K_LAM_PER_SAMPLE,
    peak_sampling=PEAK_SAMPLING,
    gamma=GAMMA_PEAK_SAMPLING
)
val_ds = CurveSegmentSParamsDataset(
    x_val, lambda_vec, S11_val, S21_val,
    k_lam_per_sample=K_LAM_PER_SAMPLE,
    peak_sampling=False  # 验证集建议均匀采样
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================================
# 7) 模型/优化器/评估函数
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FNO_LambdaConditional_SParams(
    modes=MODES, width=WIDTH, depth=DEPTH, lam_ff=LAM_FF, head_hidden=HEAD_HIDDEN
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def eval_val_total(loader):
    model.eval()
    total_sum, n = 0.0, 0

    with torch.no_grad():
        for x, lam_n, yS, a_true in loader:
            x = x.to(device)          # (B,1,11,11)
            lam_n = lam_n.to(device)  # (B,K,1)
            yS = yS.to(device)        # (B,K,4)
            a_true = a_true.to(device)

            B, K, _ = lam_n.shape
            xk = x.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, 1, 11, 11)
            lamk = lam_n.view(B*K, 1)

            predS = model(xk, lamk).view(B, K, 4)

            S_loss = ((predS - yS) ** 2).mean()
            loss = S_loss

            if USE_SMOOTHNESS and K >= 3:
                if SMOOTH_W1 > 0:
                    d1 = predS[:, 1:, :] - predS[:, :-1, :]
                    loss = loss + SMOOTH_W1 * (d1**2).mean()
                if SMOOTH_W2 > 0:
                    d2 = predS[:, 2:, :] - 2*predS[:, 1:-1, :] + predS[:, :-2, :]
                    loss = loss + SMOOTH_W2 * (d2**2).mean()

            if USE_PEAK_AUX_LOSS:
                ReS11, ImS11, ReS21, ImS21 = predS[...,0], predS[...,1], predS[...,2], predS[...,3]  # (B,K)
                A_pred = 1.0 - (ReS11**2 + ImS11**2) - (ReS21**2 + ImS21**2)
                w = 1.0 + ALPHA_PEAK_WEIGHT * (a_true.clamp(0, 1) ** P_PEAK_WEIGHT)  # (B,K,1)
                A_aux_loss = (w.squeeze(-1) * (A_pred - a_true.squeeze(-1))**2).mean()
                loss = loss + LAMBDA_AUX * A_aux_loss

            if USE_MAG_PENALTY:
                ReS11, ImS11, ReS21, ImS21 = predS[...,0], predS[...,1], predS[...,2], predS[...,3]
                mag11 = torch.sqrt(ReS11**2 + ImS11**2 + 1e-12)
                mag21 = torch.sqrt(ReS21**2 + ImS21**2 + 1e-12)
                mag_penalty = torch.relu(mag11 - 1.0).mean() + torch.relu(mag21 - 1.0).mean()
                loss = loss + MAG_W * mag_penalty

            bs = x.size(0)
            total_sum += float(loss.item()) * bs
            n += bs

    return total_sum / max(n, 1)

# ==========================================================
# 8) 训练 + TensorBoard（精简）
# ==========================================================
run_name = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"{LOG_ROOT}/{run_name}")

train_hist, val_hist = [], []

if x_val.shape[2] == 0:
    raise RuntimeError("验证集为空：请检查 N_USE 或 训练/验证划分比例")
VIS_SID = 0  # val集中的第0个样本

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_sum, n_count = 0.0, 0

    for x, lam_n, yS, a_true in train_loader:
        x = x.to(device)
        lam_n = lam_n.to(device)
        yS = yS.to(device)
        a_true = a_true.to(device)

        B, K, _ = lam_n.shape

        optimizer.zero_grad()

        # 同结构K点批量forward
        xk = x.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, 1, 11, 11)
        lamk = lam_n.view(B*K, 1)
        predS = model(xk, lamk).view(B, K, 4)

        # 1) 主损失：同结构曲线段整体误差
        S_loss = ((predS - yS) ** 2).mean()
        loss = S_loss

        # 2) 曲线一致性：只在同结构K点内部做差分（排序已在Dataset完成）
        if USE_SMOOTHNESS and K >= 3:
            if SMOOTH_W1 > 0:
                d1 = predS[:, 1:, :] - predS[:, :-1, :]
                loss = loss + SMOOTH_W1 * (d1**2).mean()
            if SMOOTH_W2 > 0:
                d2 = predS[:, 2:, :] - 2*predS[:, 1:-1, :] + predS[:, :-2, :]
                loss = loss + SMOOTH_W2 * (d2**2).mean()

        # 3) 峰加权辅助损失：仍是同结构K点内计算
        if USE_PEAK_AUX_LOSS:
            ReS11, ImS11, ReS21, ImS21 = predS[...,0], predS[...,1], predS[...,2], predS[...,3]
            A_pred = 1.0 - (ReS11**2 + ImS11**2) - (ReS21**2 + ImS21**2)  # (B,K)
            w = 1.0 + ALPHA_PEAK_WEIGHT * (a_true.clamp(0, 1) ** P_PEAK_WEIGHT)  # (B,K,1)
            A_aux_loss = (w.squeeze(-1) * (A_pred - a_true.squeeze(-1))**2).mean()
            loss = loss + LAMBDA_AUX * A_aux_loss

        # 4) 幅值约束：抑制|S|>1导致的不物理毛刺
        if USE_MAG_PENALTY:
            ReS11, ImS11, ReS21, ImS21 = predS[...,0], predS[...,1], predS[...,2], predS[...,3]
            mag11 = torch.sqrt(ReS11**2 + ImS11**2 + 1e-12)
            mag21 = torch.sqrt(ReS21**2 + ImS21**2 + 1e-12)
            mag_penalty = torch.relu(mag11 - 1.0).mean() + torch.relu(mag21 - 1.0).mean()
            loss = loss + MAG_W * mag_penalty

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_sum += float(loss.item()) * bs
        n_count += bs

    train_total = total_sum / max(n_count, 1)
    val_total = eval_val_total(val_loader)

    train_hist.append(train_total)
    val_hist.append(val_total)

    # TensorBoard 精简记录：只记录 train/val 总loss + lr
    writer.add_scalar("loss/train", train_total, epoch)
    writer.add_scalar("loss/val", val_total, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    # 每隔若干epoch：画一次 A(λ)（只写一张）
    if (epoch % PLOT_EVERY == 0) or (epoch == 1):
        # 真值 A(λ)
        s11_true = S11_val[VIS_SID, :]
        s21_true = S21_val[VIS_SID, :]
        A_true = (1.0 - np.abs(s11_true)**2 - np.abs(s21_true)**2).astype(np.float32)

        # 预测 A(λ)
        _, _, A_pred_full = predict_full_S_and_A(model, x_val[:, :, VIS_SID], lambda_vec, device, clamp_mag=True)

        fig = make_A_figure(lambda_vec, A_true, A_pred_full, title=f"Epoch {epoch} | A(λ)")
        writer.add_figure("viz/A_val0", fig, epoch)
        plt.close(fig)

    print(f"Epoch {epoch:03d} | train_total={train_total:.6e} | val_total={val_total:.6e}")

# 保存模型（含config，方便预测脚本自动匹配结构）
SAVE_PATH = r"C:\Users\90740\Desktop\final\fno_sparams_curve.pt"
torch.save({
    "state_dict": model.state_dict(),
    "config": {
        "MODES": MODES, "WIDTH": WIDTH, "DEPTH": DEPTH, "LAM_FF": LAM_FF, "HEAD_HIDDEN": HEAD_HIDDEN,
        "K_LAM_PER_SAMPLE": K_LAM_PER_SAMPLE,
        "PEAK_SAMPLING": PEAK_SAMPLING, "GAMMA_PEAK_SAMPLING": GAMMA_PEAK_SAMPLING,
        "USE_SMOOTHNESS": USE_SMOOTHNESS, "SMOOTH_W2": SMOOTH_W2, "SMOOTH_W1": SMOOTH_W1,
        "USE_PEAK_AUX_LOSS": USE_PEAK_AUX_LOSS, "ALPHA_PEAK_WEIGHT": ALPHA_PEAK_WEIGHT,
        "P_PEAK_WEIGHT": P_PEAK_WEIGHT, "LAMBDA_AUX": LAMBDA_AUX,
        "USE_MAG_PENALTY": USE_MAG_PENALTY, "MAG_W": MAG_W,
        "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY
    }
}, SAVE_PATH)

writer.close()

# 训练结束画loss曲线
plt.figure(figsize=(7,4))
plt.plot(train_hist, label="train_total")
plt.plot(val_hist, label="val_total")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_curve_training.png", dpi=200)
plt.show()

print(f"训练完成：已保存模型 {SAVE_PATH} 和 loss_curve_curve_training.png")
