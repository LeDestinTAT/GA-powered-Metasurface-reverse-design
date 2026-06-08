import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_fno_maxwell as core
from src.project_paths import FIELD_DATA_DIR, OPTIONAL_PATTERNS_PATH, SAMPLING_META_PATH, TRAIN_RUN_OUTPUTS_DIR

# ==========================================================
# 0) PyCharm 直接运行配置
#    这个脚本只是一个启动器，真正训练逻辑在 train_fno_maxwell.py
# ==========================================================
DATA_DIR = FIELD_DATA_DIR
PATTERNS_MAT = OPTIONAL_PATTERNS_PATH
SAMPLING_META = SAMPLING_META_PATH
SAVE_DIR = TRAIN_RUN_OUTPUTS_DIR / "fno_field_maxwell"

# 训练基本参数
SEED = 42
MAX_SAMPLES = 128
TRAIN_FRAC = 0.9
EPOCHS = 200
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = "cuda"
USE_AMP = True

# 下采样 / 裁剪
DOWN_X = 1
DOWN_Y = 1
DOWN_Z = 2
Z_MIN = None
Z_MAX = None

# FNO 结构
MODES_X = 12
MODES_Y = 12
MODES_Z = 12
WIDTH = 32
DEPTH = 4
HIDDEN_CHANNELS = 128

# 优化
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 10
LOG_EVERY = 5

# 监督 / 物理项
PREDICT_H = True
SUPERVISE_COMPONENTS = "Ex,Ey,Ez,Hx,Hy,Hz"
FIELD_WEIGHT = 1.0
CURL_E_WEIGHT = 0.1
CURL_H_WEIGHT = 0.1
DIV_WEIGHT = 0.0
TIME_CONVENTION = "exp_minus_iwt"

# 材料参数与分层
METAL_EPS_REAL = -200.0
METAL_EPS_IMAG = 80.0
DIELECTRIC_EPS_REAL = 2.25
DIELECTRIC_EPS_IMAG = 0.0
AIR_EPS_REAL = 1.0
AIR_EPS_IMAG = 0.0
BOTTOM_METAL_ZMAX = 100e-9
DIELECTRIC_ZMAX = 400e-9
TOP_PATTERN_ZMAX = 430e-9
PERIOD_X = 2.8e-6
PERIOD_Y = 2.8e-6


def build_argv() -> list[str]:
    argv = [
        "train_fno_maxwell.py",
        "--data-dir", str(DATA_DIR),
        "--save-dir", str(SAVE_DIR),
        "--seed", str(SEED),
        "--max-samples", str(MAX_SAMPLES),
        "--train-frac", str(TRAIN_FRAC),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--device", DEVICE,
        "--downsample-x", str(DOWN_X),
        "--downsample-y", str(DOWN_Y),
        "--downsample-z", str(DOWN_Z),
        "--modes-x", str(MODES_X),
        "--modes-y", str(MODES_Y),
        "--modes-z", str(MODES_Z),
        "--width", str(WIDTH),
        "--depth", str(DEPTH),
        "--hidden-channels", str(HIDDEN_CHANNELS),
        "--lr", str(LR),
        "--weight-decay", str(WEIGHT_DECAY),
        "--grad-clip", str(GRAD_CLIP),
        "--save-every", str(SAVE_EVERY),
        "--log-every", str(LOG_EVERY),
        "--supervise-components", SUPERVISE_COMPONENTS,
        "--field-weight", str(FIELD_WEIGHT),
        "--curl-e-weight", str(CURL_E_WEIGHT),
        "--curl-h-weight", str(CURL_H_WEIGHT),
        "--div-weight", str(DIV_WEIGHT),
        "--time-convention", TIME_CONVENTION,
        "--metal-eps-real", str(METAL_EPS_REAL),
        "--metal-eps-imag", str(METAL_EPS_IMAG),
        "--dielectric-eps-real", str(DIELECTRIC_EPS_REAL),
        "--dielectric-eps-imag", str(DIELECTRIC_EPS_IMAG),
        "--air-eps-real", str(AIR_EPS_REAL),
        "--air-eps-imag", str(AIR_EPS_IMAG),
        "--bottom-metal-zmax", str(BOTTOM_METAL_ZMAX),
        "--dielectric-zmax", str(DIELECTRIC_ZMAX),
        "--top-pattern-zmax", str(TOP_PATTERN_ZMAX),
        "--period-x", str(PERIOD_X),
        "--period-y", str(PERIOD_Y),
    ]

    if PATTERNS_MAT is not None:
        argv.extend(["--patterns-mat", str(PATTERNS_MAT)])
    if SAMPLING_META is not None:
        argv.extend(["--sampling-meta", str(SAMPLING_META)])
    if Z_MIN is not None:
        argv.extend(["--z-min", str(Z_MIN)])
    if Z_MAX is not None:
        argv.extend(["--z-max", str(Z_MAX)])
    if USE_AMP:
        argv.append("--amp")
    if not PREDICT_H:
        argv.append("--predict-e-only")

    return argv


if __name__ == "__main__":
    sys.argv = build_argv()
    core.main()
