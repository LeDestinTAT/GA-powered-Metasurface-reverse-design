from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import CURVE_DATASET_CACHE_PATH, TRAIN_RUN_OUTPUTS_DIR


LEGACY_FINAL_DIR = PROJECT_ROOT / "final"
PATTERNS_OUT = LEGACY_FINAL_DIR / "training_patterns_11x11_current91.mat"
SPARAMS_OUT = LEGACY_FINAL_DIR / "Sparams_dataset_current91.mat"


def main():
    if not CURVE_DATASET_CACHE_PATH.is_file():
        raise FileNotFoundError(f"Curve cache not found: {CURVE_DATASET_CACHE_PATH}")

    LEGACY_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_dir = TRAIN_RUN_OUTPUTS_DIR / "curve_cache"
    summary_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cache = np.load(CURVE_DATASET_CACHE_PATH, allow_pickle=False)

    pattern_11 = np.asarray(cache["pattern_11"], dtype=np.float32)  # (N,11,11)
    selected = np.transpose(pattern_11, (1, 2, 0)).astype(np.float32)  # (11,11,N)
    lambda_vec = np.asarray(cache["lambda_vec"], dtype=np.float32).reshape(-1, 1)
    s11_all = (
        np.asarray(cache["s11_real"], dtype=np.float32)
        + 1j * np.asarray(cache["s11_imag"], dtype=np.float32)
    ).T.astype(np.complex64)  # (M,N)
    s21_all = np.zeros_like(s11_all, dtype=np.complex64)

    savemat(PATTERNS_OUT, {"selected": selected}, do_compression=True)
    savemat(
        SPARAMS_OUT,
        {
            "lambda_vec": lambda_vec,
            "S11_all": s11_all,
            "S21_all": s21_all,
        },
        do_compression=True,
    )

    summary = {
        "cache_source": str(CURVE_DATASET_CACHE_PATH),
        "patterns_out": str(PATTERNS_OUT),
        "sparams_out": str(SPARAMS_OUT),
        "num_samples": int(pattern_11.shape[0]),
        "curve_length": int(lambda_vec.shape[0]),
        "elapsed_seconds": round(float(time.time() - t0), 3),
    }
    summary_path = summary_dir / f"legacy_fno_export_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Legacy FNO mats exported.")
    print(f"  patterns: {PATTERNS_OUT}")
    print(f"  sparams:  {SPARAMS_OUT}")
    print(f"  samples = {pattern_11.shape[0]}, curve_len = {lambda_vec.shape[0]}")
    print(f"  summary:  {summary_path}")


if __name__ == "__main__":
    main()
