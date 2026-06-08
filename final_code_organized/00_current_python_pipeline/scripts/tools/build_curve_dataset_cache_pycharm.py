from __future__ import annotations

import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    CURVE_DATASET_CACHE_PATH,
    FIELD_DATA_DIR,
    TRAIN_RUN_OUTPUTS_DIR,
    ensure_standard_dirs,
)

with redirect_stdout(io.StringIO()):
    from scripts.train.train_fno_curve_only_pycharm import (
        build_curve_weight,
        choose_background_index,
        detect_top_two_peaks,
        peak_um_to_bin_id,
        read_sample_header,
    )


OUTPUT_SUMMARY_DIR = TRAIN_RUN_OUTPUTS_DIR / "curve_cache"


def build_cache():
    ensure_standard_dirs()
    OUTPUT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    sample_name_pattern = re.compile(r"^sample_\d+\.mat$")
    sample_files = sorted([p for p in FIELD_DATA_DIR.glob("sample_*.mat") if sample_name_pattern.match(p.name)])
    if not sample_files:
        raise RuntimeError(f"No sample_*.mat files found under {FIELD_DATA_DIR}")

    print(f"Found {len(sample_files)} standard sample files.")
    t0 = time.time()

    lambda_ref = None
    sample_names: list[str] = []
    sample_ids: list[int] = []
    patterns: list[np.ndarray] = []
    s11_real: list[np.ndarray] = []
    s11_imag: list[np.ndarray] = []
    absorption: list[np.ndarray] = []
    curve_weight: list[np.ndarray] = []
    main_idx: list[int] = []
    secondary_idx: list[int] = []
    background_idx: list[int] = []
    main_peak_um: list[float] = []
    peak_bin: list[int] = []
    failed: list[dict[str, str]] = []

    for idx, path in enumerate(sample_files, start=1):
        try:
            sid = int(path.stem.split("_")[-1])
            lam, s11_curve, pattern_11 = read_sample_header(path)
            lam = np.asarray(lam, dtype=np.float32).reshape(-1)
            if lambda_ref is None:
                lambda_ref = lam.copy()
            elif lam.shape != lambda_ref.shape or not np.allclose(lam, lambda_ref, atol=1e-12, rtol=1e-7):
                raise ValueError("lambda grid mismatch")

            pattern_11 = (np.asarray(pattern_11).reshape(11, 11) != 0).astype(np.uint8)
            s11_curve = np.asarray(s11_curve, dtype=np.complex64).reshape(-1)
            a_curve = np.clip(1.0 - np.abs(s11_curve) ** 2, 0.0, 1.0).astype(np.float32)
            m_idx, s_idx = detect_top_two_peaks(a_curve)
            b_idx = choose_background_index(a_curve, m_idx, s_idx)
            m_peak_um = float(lam[m_idx] * 1e6)
            p_bin = peak_um_to_bin_id(m_peak_um)

            sample_names.append(path.name)
            sample_ids.append(sid)
            patterns.append(pattern_11)
            s11_real.append(np.real(s11_curve).astype(np.float32))
            s11_imag.append(np.imag(s11_curve).astype(np.float32))
            absorption.append(a_curve)
            curve_weight.append(build_curve_weight(lam, a_curve, m_idx, s_idx).astype(np.float32))
            main_idx.append(int(m_idx))
            secondary_idx.append(int(s_idx))
            background_idx.append(int(b_idx))
            main_peak_um.append(m_peak_um)
            peak_bin.append(int(p_bin))
        except Exception as exc:
            failed.append({"file": path.name, "error": str(exc)})

        if (idx % 250 == 0) or (idx == len(sample_files)):
            print(f"Processed {idx}/{len(sample_files)} files.")

    if lambda_ref is None or not sample_names:
        raise RuntimeError("No valid samples were collected.")

    cache_payload = {
        "sample_name": np.asarray(sample_names),
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "pattern_11": np.stack(patterns, axis=0).astype(np.uint8),
        "lambda_vec": np.asarray(lambda_ref, dtype=np.float32),
        "s11_real": np.stack(s11_real, axis=0).astype(np.float32),
        "s11_imag": np.stack(s11_imag, axis=0).astype(np.float32),
        "absorption": np.stack(absorption, axis=0).astype(np.float32),
        "curve_weight": np.stack(curve_weight, axis=0).astype(np.float32),
        "main_idx": np.asarray(main_idx, dtype=np.int64),
        "secondary_idx": np.asarray(secondary_idx, dtype=np.int64),
        "background_idx": np.asarray(background_idx, dtype=np.int64),
        "main_peak_um": np.asarray(main_peak_um, dtype=np.float32),
        "peak_bin": np.asarray(peak_bin, dtype=np.int64),
    }
    np.savez_compressed(CURVE_DATASET_CACHE_PATH, **cache_payload)

    summary = {
        "cache_path": str(CURVE_DATASET_CACHE_PATH),
        "source_dir": str(FIELD_DATA_DIR),
        "total_files_seen": len(sample_files),
        "valid_samples": int(len(sample_names)),
        "failed_samples": int(len(failed)),
        "curve_length": int(len(lambda_ref)),
        "elapsed_seconds": round(float(time.time() - t0), 3),
        "peak_bin_distribution": {
            f"bin_{bin_id}": int(np.sum(cache_payload["peak_bin"] == bin_id))
            for bin_id in range(4)
        },
    }
    summary_path = OUTPUT_SUMMARY_DIR / f"curve_cache_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "failed_examples": failed[:20]}, f, ensure_ascii=False, indent=2)

    print("Curve cache build finished.")
    print(f"  cache:   {CURVE_DATASET_CACHE_PATH}")
    print(f"  summary: {summary_path}")
    print(f"  valid samples = {len(sample_names)}")
    print(f"  failed samples = {len(failed)}")


if __name__ == "__main__":
    build_cache()
