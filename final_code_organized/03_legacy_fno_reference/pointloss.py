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
# 0) 配置
# ==========================================================
patterns_path = r"C:\Users\90740\Desktop\final\training_patterns_11x11.mat"
sparams_path  = r"C:\Users\90740\Desktop\final\Sparams_dataset.mat"

EPOCHS = 1000
BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
K_LAM_PER_SAMPLE = 24

# 只用有效样本前 N_USE 条（前800写800；用全部写 None）
N_USE = 800

# 每隔多少 epoch 做一次可视化
PLOT_EVERY = 5

# 判定有效样本（S参数非全0）的阈值
EPS_VALID = 1e-12

# ——偏峰采样强度：越大越偏向吸收峰附近
PEAK_SAMPLING = True
GAMMA_PEAK_SAMPLING = 1.5  # 1~4

# ——峰加权辅助损失：让峰附近误差更贵
USE_PEAK_AUX_LOSS = True
ALPHA_PEAK_WEIGHT = 5.0    # 2~10
P_PEAK_WEIGHT = 2.0        # 1~3
LAMBDA_AUX = 0.5           # 0.1~2（辅助项占比）

# ——模型结构
MODES = 5
WIDTH = 96
DEPTH = 5
LAM_FF = 4
HEAD_HIDDEN = 256

# ——优化器
LR = 1e-4
WEIGHT_DECAY = 1e-4

# ——TensorBoard 日志目录（自动分run）
LOG_ROOT = "runs/fno_sparams"


# ==========================================================
# 1) 自动兼容读取 .mat（v7.3 / 非v7.3）
# ==========================================================
def load_mat_auto(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    if os.path.getsize(path) < 1024:
        raise OSError(f"文件过小，可能损坏或路径指错：{path}")

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
        out = {k: v for k, v in out.items() if not k.startswith("__")}
        return out

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
# 2) Dataset：λ 条件化 +（可选）偏峰采样
# ==========================================================
class LambdaConditionalSParamsDataset(Dataset):
    """
    返回：
      x: (1,11,11)
      lam_n: (1,)
      yS: (4,) = [ReS11, ImS11, ReS21, ImS21]
      a_true: (1,) = A(λ) 仅用于加权/指标
    """
    def __init__(self, patterns_11x11xN, lambda_vec, S11_by_sample, S21_by_sample,
                 k_lam_per_sample=8, peak_sampling=True, gamma=2.0):
        self.x = patterns_11x11xN            # (11,11,N)
        self.lam_vec = lambda_vec            # (M,)
        self.S11 = S11_by_sample             # (N,M) complex
        self.S21 = S21_by_sample             # (N,M) complex
        self.N, self.M = self.S11.shape
        self.k = k_lam_per_sample

        self.lam_min = float(lambda_vec.min())
        self.lam_max = float(lambda_vec.max())

        # 真值吸收：用于偏峰采样和加权（不作为主监督）
        R = np.abs(self.S11)**2
        T = np.abs(self.S21)**2
        self.A = (1.0 - R - T).astype(np.float32)  # (N,M)

        self.peak_sampling = peak_sampling
        self.gamma = gamma

        self.sample_pairs = []
        self.reshuffle_lambda()

    def reshuffle_lambda(self):
        self.sample_pairs = []
        eps = 1e-6

        for i in range(self.N):
            if self.peak_sampling:
                a = self.A[i, :]  # (M,)
                prob = (np.clip(a, 0, 1) + eps) ** self.gamma
                prob = prob / prob.sum()
                idx = np.random.choice(self.M, size=self.k, replace=False, p=prob)
            else:
                idx = np.random.choice(self.M, size=self.k, replace=False)

            for j in idx:
                self.sample_pairs.append((i, j))

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, t):
        i, j = self.sample_pairs[t]
        x = self.x[:, :, i]  # (11,11)

        lam = float(self.lam_vec[j])
        lam_n = normalize_lambda(lam, self.lam_min, self.lam_max)

        s11 = self.S11[i, j]
        s21 = self.S21[i, j]

        yS = np.array([np.real(s11), np.imag(s11), np.real(s21), np.imag(s21)], dtype=np.float32)
        a_true = np.float32(self.A[i, j])

        x = torch.from_numpy(x).unsqueeze(0)                       # (1,11,11)
        lam_n = torch.tensor([lam_n], dtype=torch.float32)         # (1,)
        yS = torch.from_numpy(yS)                                  # (4,)
        a_true = torch.tensor([a_true], dtype=torch.float32)       # (1,)
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
    def __init__(self, n_freq=16):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq) * np.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam_norm):  # (B,1)
        x = lam_norm * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class FNOEncoder(nn.Module):
    def __init__(self, modes=6, width=WIDTH, depth=4):
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

class FNO_LambdaConditional_SParams(nn.Module):
    def __init__(self, modes=6, width=WIDTH, depth=DEPTH, lam_ff=LAM_FF, head_hidden=128):
        super().__init__()
        self.encoder = FNOEncoder(modes=modes, width=width, depth=depth)
        self.lam_embed = LambdaFourierFeatures(n_freq=lam_ff)

        head_in = width + 2*lam_ff
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 4)  # 输出 4 维：Re/Im of S11/S21
        )

    def forward(self, x, lam_norm):
        z = self.encoder(x)
        le = self.lam_embed(lam_norm)
        out = self.head(torch.cat([z, le], dim=-1))
        return out


# ==========================================================
# 4) 推理：给定结构预测整条 S(λ) 并转成 A(λ)
# ==========================================================
@torch.no_grad()
def predict_spectrum_S_and_A(model, pattern_11x11, lambda_vec, device):
    """
    返回：
      S11_pred: (M,) complex
      S21_pred: (M,) complex
      A_pred:   (M,) float
    """
    model.eval()
    lam_min, lam_max = float(lambda_vec.min()), float(lambda_vec.max())

    x = torch.from_numpy(pattern_11x11.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,11,11)
    lam = torch.from_numpy(lambda_vec.astype(np.float32)).unsqueeze(1)  # (M,1)
    lam_n = normalize_lambda(lam, lam_min, lam_max).to(device)

    # 先算 z，再批量 head
    z = model.encoder(x)                 # (1,width)
    z = z.repeat(lam_n.size(0), 1)       # (M,width)
    le = model.lam_embed(lam_n)          # (M,2*ff)
    out = model.head(torch.cat([z, le], dim=-1))  # (M,4)

    out = out.detach().cpu().numpy()
    s11 = out[:, 0] + 1j * out[:, 1]
    s21 = out[:, 2] + 1j * out[:, 3]
    A = 1.0 - (np.abs(s11)**2) - (np.abs(s21)**2)
    return s11.astype(np.complex64), s21.astype(np.complex64), A.astype(np.float32)

def make_A_figure(lambda_vec, true_A, pred_A, title="Absorption Spectrum"):
    fig = plt.figure(figsize=(6,4))
    plt.plot(lambda_vec, true_A, label="true A")
    plt.plot(lambda_vec, pred_A, "--", label="pred A")
    plt.xlabel("lambda")
    plt.ylabel("A")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig

def peak_pos(y, lambda_vec):
    return float(lambda_vec[int(np.argmax(y))])

# ==========================================================
# ====【新增：多峰检测/匹配 + 带峰标注的谱图】====
# ==========================================================
def smooth1d(A, k=5):
    """滑动平均（仅用于找峰/评估，不用于训练）"""
    k = int(k)
    if k <= 1:
        return A.copy()
    pad = k // 2
    A_pad = np.pad(A, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(A_pad, kernel, mode="valid")

def find_peaks_simple(A, lambda_vec, k_smooth=5, thr_rel=0.2, min_dist=3):
    """
    多峰检测（纯numpy）：
      1) 轻微平滑
      2) 找局部极大
      3) 相对阈值过滤
      4) 最小间距过滤
    返回：[(lam_i, amp_i, idx_i), ...]  按 amp 降序
    """
    A = np.asarray(A).astype(np.float32)
    lam = np.asarray(lambda_vec).astype(np.float32)

    As = smooth1d(A, k=k_smooth)
    M = len(As)
    if M < 3:
        return []

    # 局部极大：As[i-1] < As[i] >= As[i+1]
    candidates = np.where((As[1:-1] > As[:-2]) & (As[1:-1] >= As[2:]))[0] + 1
    if candidates.size == 0:
        idx = int(np.argmax(As))
        return [(float(lam[idx]), float(A[idx]), idx)]

    # 按平滑后峰高排序
    order = np.argsort(-As[candidates])
    candidates = candidates[order]

    # 相对阈值过滤
    Amax = float(As.max())
    keep = [i for i in candidates if float(As[i]) >= thr_rel * Amax]

    # 最小间距：贪心保留更高峰
    picked = []
    for i in keep:
        if all(abs(i - j) >= min_dist for j in picked):
            picked.append(i)

    peaks = [(float(lam[i]), float(A[i]), int(i)) for i in picked]
    peaks.sort(key=lambda x: -x[1])  # 按真实A高度排序
    return peaks

def match_peaks_by_lambda(peaks_true, peaks_pred, max_shift_idx=5):
    """
    按“索引距离最近”做贪心匹配
    返回：pairs, miss_true, false_pred
    """
    used = set()
    pairs = []
    miss = 0

    for t in peaks_true:
        best = None
        best_d = 10**9
        for p_i, p in enumerate(peaks_pred):
            if p_i in used:
                continue
            d = abs(p[2] - t[2])  # idx差
            if d < best_d:
                best_d = d
                best = (p_i, p)
        if best is None or best_d > max_shift_idx:
            miss += 1
        else:
            used.add(best[0])
            pairs.append((t, best[1]))

    false = len(peaks_pred) - len(used)
    return pairs, miss, false

def make_A_figure_with_peaks(lambda_vec, true_A, pred_A, peaks_true, peaks_pred, title="Absorption Spectrum"):
    """画 A(λ) 并标注真峰/预测峰位置"""
    fig = plt.figure(figsize=(6.5,4.2))
    plt.plot(lambda_vec, true_A, label="true A")
    plt.plot(lambda_vec, pred_A, "--", label="pred A")

    # 真峰：圆点
    if peaks_true:
        xs = [p[0] for p in peaks_true]
        ys = [p[1] for p in peaks_true]
        plt.scatter(xs, ys, marker="o", s=30, label="true peaks")

    # 预测峰：叉号
    if peaks_pred:
        xs = [p[0] for p in peaks_pred]
        ys = [p[1] for p in peaks_pred]
        plt.scatter(xs, ys, marker="x", s=40, label="pred peaks")

    plt.xlabel("lambda")
    plt.ylabel("A")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


# ==========================================================
# 5) 读取数据 + 过滤无效样本 + 截断前800有效样本
# ==========================================================
patterns = load_mat_auto(patterns_path)
sp = load_mat_auto(sparams_path)

selected = to_numpy_bool(patterns["selected"])                 # (11,11,N_total)
lambda_vec = np.array(sp["lambda_vec"]).squeeze()              # (M,)

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
# 6) 划分 train/val + DataLoader
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

train_ds = LambdaConditionalSParamsDataset(
    x_train, lambda_vec, S11_train, S21_train,
    k_lam_per_sample=K_LAM_PER_SAMPLE,
    peak_sampling=PEAK_SAMPLING, gamma=GAMMA_PEAK_SAMPLING
)
val_ds = LambdaConditionalSParamsDataset(
    x_val, lambda_vec, S11_val, S21_val,
    k_lam_per_sample=K_LAM_PER_SAMPLE,
    peak_sampling=False  # 验证集建议均匀采样
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)


# ==========================================================
# 7) 模型/优化器/评估
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FNO_LambdaConditional_SParams(
    modes=MODES, width=WIDTH, depth=DEPTH, lam_ff=LAM_FF, head_hidden=HEAD_HIDDEN
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def eval_val_losses(loader):
    """
    返回：val_total_loss, val_S_loss, val_Aaux_loss
    """
    model.eval()
    total_sum, s_sum, a_sum, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for x, lam, yS, a_true in loader:
            x, lam = x.to(device), lam.to(device)
            yS = yS.to(device)           # (B,4)
            a_true = a_true.to(device)   # (B,1)

            predS = model(x, lam)        # (B,4)
            S_loss = ((predS - yS) ** 2).mean()

            A_aux_loss = torch.tensor(0.0, device=device)
            if USE_PEAK_AUX_LOSS:
                ReS11, ImS11, ReS21, ImS21 = predS[:,0], predS[:,1], predS[:,2], predS[:,3]
                A_pred = 1.0 - (ReS11**2 + ImS11**2) - (ReS21**2 + ImS21**2)

                w = 1.0 + ALPHA_PEAK_WEIGHT * (a_true.clamp(0,1) ** P_PEAK_WEIGHT)  # (B,1)
                A_aux_loss = (w.squeeze(1) * (A_pred - a_true.squeeze(1))**2).mean()

            loss = S_loss + (LAMBDA_AUX * A_aux_loss if USE_PEAK_AUX_LOSS else 0.0)

            bs = x.size(0)
            total_sum += float(loss.item()) * bs
            s_sum += float(S_loss.item()) * bs
            a_sum += float(A_aux_loss.item()) * bs
            n += bs

    return total_sum / max(n,1), s_sum / max(n,1), a_sum / max(n,1)


# ==========================================================
# 8) 训练 + TensorBoard 可视化（包含整谱预测与峰指标）
# ==========================================================
run_name = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"{LOG_ROOT}/{run_name}")

train_total_hist, val_total_hist = [], []
train_S_hist, val_S_hist = [], []
train_Aaux_hist, val_Aaux_hist = [], []

if x_val.shape[2] == 0:
    raise RuntimeError("验证集为空：请检查 N_USE 或 训练/验证划分比例")
VIS_SID = 0  # val集中的第0个样本
# ================== Early Stopping（插入点1）==================
best_val = float("inf")
patience = 30       # 连续多少个epoch验证集不提升就停止（建议20~50）
min_delta = 1e-6    # 认为“有提升”的最小下降幅度
bad_count = 0

BEST_CKPT_PATH = r"C:\Users\90740\Desktop\final\fno_best.pt"
# =============================================================

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_ds.reshuffle_lambda()

    total_sum, s_sum, a_sum, n_count = 0.0, 0.0, 0.0, 0

    for x, lam, yS, a_true in train_loader:
        x, lam = x.to(device), lam.to(device)
        yS = yS.to(device)  # (B,4)
        a_true = a_true.to(device)  # (B,1)

        optimizer.zero_grad()
        predS = model(x, lam)  # (B,4)

        # 1) 主损失：复数S监督（Re/Im）
        S_loss = ((predS - yS) ** 2).mean()

        # 2) 辅助损失：从预测S计算吸收A_pred，对峰附近加权（你的原逻辑）
        A_aux_loss = torch.tensor(0.0, device=device)
        if USE_PEAK_AUX_LOSS:
            ReS11, ImS11, ReS21, ImS21 = predS[:, 0], predS[:, 1], predS[:, 2], predS[:, 3]
            A_pred = 1.0 - (ReS11 ** 2 + ImS11 ** 2) - (ReS21 ** 2 + ImS21 ** 2)
            w = 1.0 + ALPHA_PEAK_WEIGHT * (a_true.clamp(0, 1) ** P_PEAK_WEIGHT)  # (B,1)
            A_aux_loss = (w.squeeze(1) * (A_pred - a_true.squeeze(1)) ** 2).mean()

        loss = S_loss + (LAMBDA_AUX * A_aux_loss if USE_PEAK_AUX_LOSS else 0.0)

        # ==========================================================
        #插入：|S| 约束（防止幅值乱飙导致全带毛）
        # ==========================================================
        MAG_W = 0.05  # 0.01~0.1 之间试；先用0.05
        ReS11, ImS11, ReS21, ImS21 = predS[:, 0], predS[:, 1], predS[:, 2], predS[:, 3]
        mag11 = torch.sqrt(ReS11 ** 2 + ImS11 ** 2 + 1e-12)
        mag21 = torch.sqrt(ReS21 ** 2 + ImS21 ** 2 + 1e-12)
        mag_penalty = torch.relu(mag11 - 1.0).mean() + torch.relu(mag21 - 1.0).mean()
        loss = loss + MAG_W * mag_penalty

        # ==========================================================
        #  插入：λ 平滑正则（关键，抑制整条谱高频抖动）
        # ==========================================================
        SMOOTH_W = 0.1  # 0.05~0.2 之间试；先用0.1
        DELTA = 2.0 / (len(lambda_vec) - 1)  # lam_norm ∈ [-1,1] 的相邻步长

        lam2 = torch.clamp(lam + DELTA, -1.0, 1.0)  # 相邻λ点（归一化域）
        predS2 = model(x, lam2)
        smooth_loss = ((predS2 - predS) ** 2).mean()
        loss = loss + SMOOTH_W * smooth_loss

        # ==========================================================
        # 反传与更新
        # ==========================================================
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_sum += float(loss.item()) * bs
        s_sum += float(S_loss.item()) * bs
        a_sum += float(A_aux_loss.item()) * bs
        n_count += bs

    train_total = total_sum / max(n_count,1)
    train_S = s_sum / max(n_count,1)
    train_Aaux = a_sum / max(n_count,1)

    val_total, val_S, val_Aaux = eval_val_losses(val_loader)

    # ================== Early Stopping（插入点2）==================
    if val_total < best_val - min_delta:
        best_val = val_total
        bad_count = 0

        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "MODES": MODES, "WIDTH": WIDTH, "DEPTH": DEPTH, "LAM_FF": LAM_FF, "HEAD_HIDDEN": HEAD_HIDDEN,
                "ALPHA_PEAK_WEIGHT": ALPHA_PEAK_WEIGHT, "P_PEAK_WEIGHT": P_PEAK_WEIGHT, "LAMBDA_AUX": LAMBDA_AUX,
                "PEAK_SAMPLING": PEAK_SAMPLING, "GAMMA_PEAK_SAMPLING": GAMMA_PEAK_SAMPLING
            }
        }, BEST_CKPT_PATH)

    else:
        bad_count += 1

    writer.add_scalar("early_stop/best_val_total", best_val, epoch)
    writer.add_scalar("early_stop/bad_count", bad_count, epoch)

    if bad_count >= patience:
        print(f"Early stopping: val_total 连续 {patience} 个epoch未提升，停止训练。best_val={best_val:.6e}")
        break
    # =============================================================

    train_total_hist.append(train_total)
    val_total_hist.append(val_total)
    train_S_hist.append(train_S)
    val_S_hist.append(val_S)
    train_Aaux_hist.append(train_Aaux)
    val_Aaux_hist.append(val_Aaux)

    # --- TensorBoard 标量 ---
    writer.add_scalar("loss/train_total", train_total, epoch)
    writer.add_scalar("loss/val_total", val_total, epoch)
    writer.add_scalar("loss/train_S", train_S, epoch)
    writer.add_scalar("loss/val_S", val_S, epoch)
    if USE_PEAK_AUX_LOSS:
        writer.add_scalar("loss/train_Aaux", train_Aaux, epoch)
        writer.add_scalar("loss/val_Aaux", val_Aaux, epoch)

    # --- 每隔若干epoch：整谱预测 + 峰位/峰高 + 谱图 ---
    # --- 每隔若干epoch：整谱预测 + 多峰指标 + 谱图（含峰标注） ---
    if epoch % PLOT_EVERY == 0 or epoch == 1:
        # 真值整谱 A（从真值 S 得到）
        s11_true = S11_val[VIS_SID, :]
        s21_true = S21_val[VIS_SID, :]
        A_true = (1.0 - np.abs(s11_true) ** 2 - np.abs(s21_true) ** 2).astype(np.float32)

        # 预测整谱 S 和 A
        s11_pred, s21_pred, A_pred = predict_spectrum_S_and_A(model, x_val[:, :, VIS_SID], lambda_vec, device)

        # ============ 1) 保留单峰(top1)指标（方便你观察最高峰）============
        peak_err_top1 = abs(peak_pos(A_pred, lambda_vec) - peak_pos(A_true, lambda_vec))
        amp_err_top1 = abs(float(A_pred.max()) - float(A_true.max()))
        writer.add_scalar("metric/top1_peak_pos_err", peak_err_top1, epoch)
        writer.add_scalar("metric/top1_peak_amp_err", amp_err_top1, epoch)

        # ============ 2) 多峰检测与匹配（更符合你的谱线特性）============
        # 这些参数你可以在这里调（稳健优先）
        K_SMOOTH = 5  # 找峰时轻微平滑窗口：3~9
        THR_REL = 0.2  # 相对阈值：0.1~0.3
        MIN_DIST = 3  # 峰最小间距（点数）：2~6
        MAX_SHIFT = 5  # 匹配允许偏移（点数）：3~8

        peaks_true = find_peaks_simple(A_true, lambda_vec, k_smooth=K_SMOOTH, thr_rel=THR_REL, min_dist=MIN_DIST)
        peaks_pred = find_peaks_simple(A_pred, lambda_vec, k_smooth=K_SMOOTH, thr_rel=THR_REL, min_dist=MIN_DIST)

        pairs, miss, false = match_peaks_by_lambda(peaks_true, peaks_pred, max_shift_idx=MAX_SHIFT)

        if len(pairs) > 0:
            pos_errs = [abs(pt[0] - pp[0]) for (pt, pp) in pairs]  # λ误差
            amp_errs = [abs(pt[1] - pp[1]) for (pt, pp) in pairs]  # 峰高误差
            mean_pos_err = float(np.mean(pos_errs))
            mean_amp_err = float(np.mean(amp_errs))
        else:
            mean_pos_err = float("nan")
            mean_amp_err = float("nan")

        # --- TensorBoard（精简版）---
        writer.add_scalar("loss/train", train_total, epoch)
        writer.add_scalar("loss/val", val_total, epoch)

        # 如果你之后加了scheduler或想观察lr，就保留这一行
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    print(f"Epoch {epoch:03d} | train_total={train_total:.6e} val_total={val_total:.6e} | "
          f"train_S={train_S:.6e} val_S={val_S:.6e} | train_Aaux={train_Aaux:.6e} val_Aaux={val_Aaux:.6e}")

# 保存模型
torch.save({
    "state_dict": model.state_dict(),
    "config": {
        "MODES": MODES, "WIDTH": WIDTH, "DEPTH": DEPTH, "LAM_FF": LAM_FF, "HEAD_HIDDEN": HEAD_HIDDEN,
        "ALPHA_PEAK_WEIGHT": ALPHA_PEAK_WEIGHT, "P_PEAK_WEIGHT": P_PEAK_WEIGHT, "LAMBDA_AUX": LAMBDA_AUX,
        "PEAK_SAMPLING": PEAK_SAMPLING, "GAMMA_PEAK_SAMPLING": GAMMA_PEAK_SAMPLING
    }
}, "C:\\Users\\90740\\Desktop\\final\\fno_sparams_lambda_cond.pt")

writer.close()

# 训练结束画 loss 曲线（本地png）
plt.figure(figsize=(7,4))
plt.plot(train_total_hist, label="train_total")
plt.plot(val_total_hist, label="val_total")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_total_curve.png", dpi=200)
plt.show()

print("训练完成：已保存模型 fno_sparams_lambda_cond.pt 和 loss_total_curve.png")



# ==========================================================
# 9) 【示例】训练完后：给定一个 11×11 二值矩阵预测整谱
# ==========================================================
# 你可以把这里注释掉，或者替换成你想预测的新结构

