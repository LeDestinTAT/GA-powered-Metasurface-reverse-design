from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = PROJECT_ROOT / "assets"
MATERIALS_DIR = ASSETS_DIR / "materials"

DATA_DIR = PROJECT_ROOT / "data"
FIELD_DATA_DIR = DATA_DIR / "field_batch_output_compressed_air"
SAMPLING_META_PATH = FIELD_DATA_DIR / "sampling_meta.mat"
CURVE_CACHE_DIR = DATA_DIR / "curve_cache"
CURVE_DATASET_CACHE_PATH = CURVE_CACHE_DIR / "curve_dataset_11x11_s11_a.npz"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_CURRENT_DIR = MODELS_DIR / "current"
MODELS_HISTORY_DIR = MODELS_DIR / "history"
BEST_MODEL_HISTORY_ROOT = MODELS_HISTORY_DIR

LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_RUNS_DIR = LOGS_DIR / "tensorboard" / "runs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INFERENCE_OUTPUTS_DIR = OUTPUTS_DIR / "inference"
PREDICTION_OUTPUTS_DIR = OUTPUTS_DIR / "predictions"
OPTIMIZATION_OUTPUTS_DIR = OUTPUTS_DIR / "optimization"
TRAIN_RUN_OUTPUTS_DIR = OUTPUTS_DIR / "train_runs"


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


OPTIONAL_PATTERNS_PATH = _first_existing(
    DATA_DIR / "training_patterns_11x11.mat",
    PROJECT_ROOT / "training_patterns_11x11.mat",
)


def ensure_standard_dirs() -> None:
    for path in [
        MATERIALS_DIR,
        FIELD_DATA_DIR,
        CURVE_CACHE_DIR,
        MODELS_CURRENT_DIR,
        MODELS_HISTORY_DIR,
        TENSORBOARD_RUNS_DIR,
        INFERENCE_OUTPUTS_DIR,
        PREDICTION_OUTPUTS_DIR,
        OPTIMIZATION_OUTPUTS_DIR,
        TRAIN_RUN_OUTPUTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
