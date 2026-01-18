# ==========================================================
# 曲线段训练版（S参数复数监督）——优化版：
# - 逻辑不变：每个样本=一个结构+K个λ点，λ条件化预测S11/S21复数(Re/Im)
# - 主损失：K点整体S误差
# - 曲线一致性：同结构K点内部做差分平滑（idx排序后）
# - 可选：偏峰采样、峰加权辅助损失、|S|幅值约束
# - TensorBoard精简：train/val总loss + lr + 每隔PLOT_EVERY写一张A(λ)图
# - 新增：Early Stopping（val loss长时间不变自动停止）+ 保存best模型
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
# 0) 配置：把所有可调参数集中到 CFG（更精细、改起来更快）
# ==========================================================
CFG = {
    "paths": {
        "patterns_path": r"C:\Users\90740\Desktop\final\training_patterns_11x11.mat",
        "sparams_path":  r"C:\Users\90740\Desktop\final\Sparams_dataset.mat",
        "save_path":     r"C:\Users\90740\Desktop\final\fno_sparams_curve.pt",
        "log_root":      r"runs/fno_sparams_curve",
    },

    # ---- 数据与训练基本参数 ----
    "train": {
        "epochs": 500,             #最大训练轮次
        "batch_size": 64,          # 曲线段训练显存更吃紧；可用64/32
        "val_batch_size": 64,
        "k_lam_per_sample": 128,    # 每结构抽K个λ点
        "n_use": 1000,             # 只用前N条有效样本；None=全部
        "val_ratio": 0.2,
        "eps_valid": 1e-12,
        "seed": 0,
        "plot_every": 10,
        "vis_sid": 0,              # val集中第几个样本做可视化
    },

    # ---- 采样策略（只影响“同结构内选哪些λ点”）----
    "sampling": {
        "peak_sampling": True,
        "gamma_peak_sampling": 1.2,   # 1.2~2.0
        # 如你想更稳的全带覆盖：可做混合采样（不改变主逻辑，只改选点）
        "mix_uniform_ratio": 0.2,     # 0~1；0=不混合(保持你当前逻辑)；例如0.5表示一半均匀一半偏峰
    },

    # ---- 损失项开关与权重 ----
    "loss": {
        "use_smoothness": True,
        "smooth_w2": 0.15,   # 二阶差分
        "smooth_w1": 0.00,   # 一阶差分（一般不必）

        "use_peak_aux": True,
        "alpha_peak_weight": 5.0,
        "p_peak_weight": 2.0,
        "lambda_aux": 0.3,

        "use_mag_penalty": True,
        "mag_w": 0.05,
    },

    # ---- 模型结构 ----
    "model": {
        "modes": 6,
        "width": 64,
        "depth": 4,
        "lam_ff": 8,          # 建议8起步
        "head_hidden": 256,
    },

    # ---- 优化器 ----
    "optim": {
        "lr": 1e-4,
        "weight_decay": 1e-4,
    },

    # ---- Early Stopping（新增）----
    "early_stop": {
        "enable": True,
        "monitor": "val",     # 监控 val_total
        "patience": 40,       # 连续多少个epoch无明显改善就停
        "min_delta": 1e-5,    # 认为“有改善”的最小下降幅度
        "warmup": 20,         # 前多少epoch不做early stop（避免一开始抖动误判）
        "save_best": True,    # 保存best模型
        "best_name": "best_fno_sparams_curve.pt",
    },

    # ---- 推理/可视化 ----
    "viz": {
        "clamp_mag_in_viz": True,  # 可视化A(λ)时把|S|>1投影回去（更“物理”）
        "clamp_eps": 1e-6,
    },
}


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
    return (x != 0).astype(np.float32)

def load_complex_struct(arr):
    # 兼容MATLAB复数结构体
    if hasattr(arr, "dtype") and arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    return arr.astype(np.complex64)

def normalize_lambda(lam, lam_min, lam_max):
    return 2.0 * (lam - lam_min) / (lam_max - lam_min) - 1.0


# ==========================================================
# 2) Dataset：每个样本返回“同一结构的K个λ点”（曲线段）
# ==========================================================
class CurveSegmentSParamsDataset(Dataset):
    def __init__(self, patterns_11x11xN, lambda_vec, S11_by_sample, S21_by_sample,
                 k_lam_per_sample=32, peak_sampling=True, gamma=1.5, mix_uniform_ratio=0.0):
        self.x = patterns_11x11xN
        self.lam_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()
        self.S11 = S11_by_sample
        self.S21 = S21_by_sample
        self.N, self.M = self.S11.shape
        self.K = int(k_lam_per_sample)

        self.lam_min = float(self.lam_vec.min())
        self.lam_max = float(self.lam_vec.max())

        R = np.abs(self.S11)**2
        T = np.abs(self.S21)**2
        self.A = (1.0 - R - T).astype(np.float32)

        self.peak_sampling = bool(peak_sampling)
        self.gamma = float(gamma)
        self.mix_uniform_ratio = float(mix_uniform_ratio)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.x[:, :, i].astype(np.float32)

        # --- 选K个λ点 ---
        if self.peak_sampling:
            eps = 1e-6
            a = self.A[i, :]
            prob = (np.clip(a, 0, 1) + eps) ** self.gamma
            prob = prob / prob.sum()

            # 可选：混合均匀采样（默认0.0保持你原逻辑不变）
            if self.mix_uniform_ratio > 0:
                Ku = int(round(self.K * self.mix_uniform_ratio))
                Kp = self.K - Ku
                idx_u = np.random.choice(self.M, size=Ku, replace=False)
                idx_p = np.random.choice(self.M, size=Kp, replace=False, p=prob)
                idx = np.unique(np.concatenate([idx_u, idx_p]))
                # 去重后不足K则补齐
                while idx.size < self.K:
                    extra = np.random.choice(self.M, size=self.K - idx.size, replace=False)
                    idx = np.unique(np.concatenate([idx, extra]))
                idx = idx[:self.K]
            else:
                idx = np.random.choice(self.M, size=self.K, replace=False, p=prob)
        else:
            idx = np.random.choice(self.M, size=self.K, replace=False)

        idx = np.sort(idx)  # 排序后差分才有曲线含义

        lam = self.lam_vec[idx]
        lam_n = normalize_lambda(lam, self.lam_min, self.lam_max).astype(np.float32)

        s11 = self.S11[i, idx]
        s21 = self.S21[i, idx]

        yS = np.stack([np.real(s11), np.imag(s11), np.real(s21), np.imag(s21)], axis=-1).astype(np.float32)  # (K,4)
        a_true = self.A[i, idx].astype(np.float32).reshape(-1, 1)

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

    def forward(self, x, lam_norm):
        z = self.encoder(x)
        le = self.lam_embed(lam_norm)
        return self.head(torch.cat([z, le], dim=-1))


# ==========================================================
# 4) 推理：预测整条 S(λ) 并转A(λ)（仅用于可视化）
# ==========================================================
@torch.no_grad()
def predict_full_S_and_A(model, pattern_11x11, lambda_vec, device, clamp_mag=True, clamp_eps=1e-6):
    model.eval()
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32).squeeze()
    lam_min, lam_max = float(lambda_vec.min()), float(lambda_vec.max())

    x = torch.from_numpy(pattern_11x11.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    lam = torch.from_numpy(lambda_vec).unsqueeze(1)
    lam_n = normalize_lambda(lam, lam_min, lam_max).to(device)

    z = model.encoder(x)
    z = z.repeat(lam_n.size(0), 1)
    le = model.lam_embed(lam_n)
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
    fig = plt.figure(figsize=(6.5, 4.2))
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
patterns = load_mat_auto(CFG["paths"]["patterns_path"])
sp = load_mat_auto(CFG["paths"]["sparams_path"])

selected = to_numpy_bool(patterns["selected"])
lambda_vec = np.array(sp["lambda_vec"]).squeeze().astype(np.float32)

S11 = load_complex_struct(np.array(sp["S11_all"]))
S21 = load_complex_struct(np.array(sp["S21_all"]))

# 统一为 (N_total,M)
if S11.shape[0] == lambda_vec.shape[0]:
    S11 = S11.T
    S21 = S21.T

N_total = min(selected.shape[2], S11.shape[0], S21.shape[0])
selected = selected[:, :, :N_total]
S11 = S11[:N_total, :]
S21 = S21[:N_total, :]

eps_valid = CFG["train"]["eps_valid"]
valid = (np.any(np.abs(S11) > eps_valid, axis=1) | np.any(np.abs(S21) > eps_valid, axis=1))
idx_valid = np.where(valid)[0]
print("总样本数 =", N_total, "| 有效样本数 =", len(idx_valid))

n_use = CFG["train"]["n_use"]
if n_use is not None:
    idx_valid = idx_valid[:n_use]
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
np.random.seed(CFG["train"]["seed"])
idx = np.random.permutation(N)
n_train = int((1.0 - CFG["train"]["val_ratio"]) * N)
train_idx, val_idx = idx[:n_train], idx[n_train:]

x_train = selected[:, :, train_idx]
x_val   = selected[:, :, val_idx]
S11_train, S21_train = S11[train_idx, :], S21[train_idx, :]
S11_val,   S21_val   = S11[val_idx, :],   S21[val_idx, :]

train_ds = CurveSegmentSParamsDataset(
    x_train, lambda_vec, S11_train, S21_train,
    k_lam_per_sample=CFG["train"]["k_lam_per_sample"],
    peak_sampling=CFG["sampling"]["peak_sampling"],
    gamma=CFG["sampling"]["gamma_peak_sampling"],
    mix_uniform_ratio=CFG["sampling"]["mix_uniform_ratio"],
)
val_ds = CurveSegmentSParamsDataset(
    x_val, lambda_vec, S11_val, S21_val,
    k_lam_per_sample=CFG["train"]["k_lam_per_sample"],
    peak_sampling=False,  # 验证集用均匀采样
    gamma=1.0,
    mix_uniform_ratio=0.0,
)

train_loader = DataLoader(train_ds, batch_size=CFG["train"]["batch_size"], shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=CFG["train"]["val_batch_size"], shuffle=False, num_workers=0)


# ==========================================================
# 7) 模型 / 优化器 / eval
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
mc = CFG["model"]
model = FNO_LambdaConditional_SParams(
    modes=mc["modes"], width=mc["width"], depth=mc["depth"], lam_ff=mc["lam_ff"], head_hidden=mc["head_hidden"]
).to(device)

oc = CFG["optim"]
optimizer = optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])

lc = CFG["loss"]

def compute_batch_loss(predS, yS, a_true):
    """
    predS, yS: (B,K,4)
    a_true: (B,K,1)
    返回：loss(标量)
    """
    loss = ((predS - yS) ** 2).mean()  # 主损失：S复数(Re/Im) MSE

    B, K, _ = predS.shape

    # 曲线一致性（同结构K点内部差分）
    if lc["use_smoothness"] and K >= 3:
        if lc["smooth_w1"] > 0:
            d1 = predS[:, 1:, :] - predS[:, :-1, :]
            loss = loss + lc["smooth_w1"] * (d1**2).mean()
        if lc["smooth_w2"] > 0:
            d2 = predS[:, 2:, :] - 2*predS[:, 1:-1, :] + predS[:, :-2, :]
            loss = loss + lc["smooth_w2"] * (d2**2).mean()

    # 峰加权辅助（用预测S算A_pred，与A_true对齐）
    if lc["use_peak_aux"]:
        ReS11, ImS11, ReS21, ImS21 = predS[...,0], predS[...,1], predS[...,2], predS[...,3]
        A_pred = 1.0 - (ReS11**2 + ImS11**2) - (ReS21**2 + ImS21**2)
        w = 1.0 + lc["alpha_peak_weight"] * (a_true.clamp(0, 1) ** lc["p_peak_weight"])
        A_aux = (w.squeeze(-1) * (A_pred - a_true.squeeze(-1))**2).mean()
        loss = loss + lc["lambda_aux"] * A_aux

    # 幅值约束
    if lc["use_mag_penalty"]:
        ReS11, ImS11, ReS21, ImS21 = predS[...,0], predS[...,1], predS[...,2], predS[...,3]
        mag11 = torch.sqrt(ReS11**2 + ImS11**2 + 1e-12)
        mag21 = torch.sqrt(ReS21**2 + ImS21**2 + 1e-12)
        mag_penalty = torch.relu(mag11 - 1.0).mean() + torch.relu(mag21 - 1.0).mean()
        loss = loss + lc["mag_w"] * mag_penalty

    return loss


@torch.no_grad()
def eval_val_total(loader):
    model.eval()
    total_sum, n = 0.0, 0
    for x, lam_n, yS, a_true in loader:
        x = x.to(device)          # (B,1,11,11)
        lam_n = lam_n.to(device)  # (B,K,1)
        yS = yS.to(device)        # (B,K,4)
        a_true = a_true.to(device)

        B, K, _ = lam_n.shape
        xk = x.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, 1, 11, 11)
        lamk = lam_n.view(B*K, 1)

        predS = model(xk, lamk).view(B, K, 4)
        loss = compute_batch_loss(predS, yS, a_true)

        bs = x.size(0)
        total_sum += float(loss.item()) * bs
        n += bs
    return total_sum / max(n, 1)


# ==========================================================
# 8) 训练 + TensorBoard（精简）+ Early Stopping（新增）
# ==========================================================
run_name = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"{CFG['paths']['log_root']}/{run_name}")

train_hist, val_hist = [], []
epochs = CFG["train"]["epochs"]
plot_every = CFG["train"]["plot_every"]
vis_sid = CFG["train"]["vis_sid"]

# Early stopping 状态
es = CFG["early_stop"]
best_val = float("inf")
bad_epochs = 0
best_path = os.path.join(os.path.dirname(CFG["paths"]["save_path"]), es["best_name"])

for epoch in range(1, epochs + 1):
    model.train()
    total_sum, n_count = 0.0, 0

    for x, lam_n, yS, a_true in train_loader:
        x = x.to(device)
        lam_n = lam_n.to(device)
        yS = yS.to(device)
        a_true = a_true.to(device)

        B, K, _ = lam_n.shape

        optimizer.zero_grad()

        xk = x.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, 1, 11, 11)
        lamk = lam_n.view(B*K, 1)

        predS = model(xk, lamk).view(B, K, 4)
        loss = compute_batch_loss(predS, yS, a_true)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_sum += float(loss.item()) * bs
        n_count += bs

    train_total = total_sum / max(n_count, 1)
    val_total = eval_val_total(val_loader)

    train_hist.append(train_total)
    val_hist.append(val_total)

    # TensorBoard（精简）
    writer.add_scalar("loss/train", train_total, epoch)
    writer.add_scalar("loss/val", val_total, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    # 可视化：每隔若干epoch写一张A(λ)
    if (epoch % plot_every == 0) or (epoch == 1):
        s11_true = S11_val[vis_sid, :]
        s21_true = S21_val[vis_sid, :]
        A_true = (1.0 - np.abs(s11_true)**2 - np.abs(s21_true)**2).astype(np.float32)

        _, _, A_pred = predict_full_S_and_A(
            model, x_val[:, :, vis_sid], lambda_vec, device,
            clamp_mag=CFG["viz"]["clamp_mag_in_viz"],
            clamp_eps=CFG["viz"]["clamp_eps"]
        )

        fig = make_A_figure(lambda_vec, A_true, A_pred, title=f"Epoch {epoch} | A(λ)")
        writer.add_figure("viz/A_val0", fig, epoch)
        plt.close(fig)

    print(f"Epoch {epoch:03d} | train={train_total:.6e} | val={val_total:.6e}")

    # -------------------------
    # Early Stopping（新增）
    # -------------------------
    if es["enable"] and epoch >= es["warmup"]:
        improved = (val_total < best_val - es["min_delta"])
        if improved:
            best_val = val_total
            bad_epochs = 0
            if es["save_best"]:
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": {**CFG["model"], **CFG["train"], **CFG["sampling"], **CFG["loss"], **CFG["optim"]},
                    "best_val": best_val,
                    "epoch": epoch
                }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= es["patience"]:
                print(f"[EarlyStopping] val无明显改善达到 {es['patience']} 次，停止训练。best_val={best_val:.6e}")
                break


# 保存最终模型（含config）
torch.save({
    "state_dict": model.state_dict(),
    "config": {**CFG["model"], **CFG["train"], **CFG["sampling"], **CFG["loss"], **CFG["optim"]},
}, CFG["paths"]["save_path"])

writer.close()

# 训练结束画loss曲线
plt.figure(figsize=(7, 4))
plt.plot(train_hist, label="train")
plt.plot(val_hist, label="val")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_curve_training.png", dpi=200)
plt.show()

print(f"训练完成：已保存最终模型 {CFG['paths']['save_path']}")
if CFG["early_stop"]["enable"]:
    print(f"best模型（若保存）: {best_path}")
