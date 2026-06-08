from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.project_paths import MATERIALS_DIR


AU_NK_PATH = MATERIALS_DIR / "au_ciesielski.yml"
SIO2_NK_PATH = MATERIALS_DIR / "sio2_kischkat.yml"


def _extract_tabulated_nk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"data:\s*\|\s*\n(?P<body>(?:\s+[^\n]+\n?)*)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"未能在材料文件中找到 tabulated nk 数据: {path}")

    lam_um = []
    n_vals = []
    k_vals = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        lam_um.append(float(parts[0]))
        n_vals.append(float(parts[1]))
        k_vals.append(float(parts[2]))

    if not lam_um:
        raise ValueError(f"材料文件中没有可用的 nk 数据: {path}")

    lam_um = np.asarray(lam_um, dtype=np.float64)
    n_vals = np.asarray(n_vals, dtype=np.float64)
    k_vals = np.asarray(k_vals, dtype=np.float64)

    order = np.argsort(lam_um)
    return lam_um[order], n_vals[order], k_vals[order]


@lru_cache(maxsize=None)
def _load_material_table(path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _extract_tabulated_nk(Path(path_str))


def _interp_nk(path: Path, lambda_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam_tab, n_tab, k_tab = _load_material_table(str(path))
    lam = np.asarray(lambda_um, dtype=np.float64)

    lam_min = float(lam_tab[0])
    lam_max = float(lam_tab[-1])
    if np.any(lam < lam_min) or np.any(lam > lam_max):
        raise ValueError(
            f"波长 {lam.min():.4f}~{lam.max():.4f} um 超出材料表范围 {lam_min:.4f}~{lam_max:.4f} um: {path.name}"
        )

    n = np.interp(lam, lam_tab, n_tab)
    k = np.interp(lam, lam_tab, k_tab)
    return n, k


def nk_to_eps(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (n + 1j * k) ** 2


def au_eps_from_lambda_m(lambda_m: float | np.ndarray) -> np.ndarray:
    lambda_um = np.asarray(lambda_m, dtype=np.float64) * 1e6
    n, k = _interp_nk(AU_NK_PATH, lambda_um)
    return nk_to_eps(n, k)


def sio2_eps_from_lambda_m(lambda_m: float | np.ndarray) -> np.ndarray:
    lambda_um = np.asarray(lambda_m, dtype=np.float64) * 1e6
    n, k = _interp_nk(SIO2_NK_PATH, lambda_um)
    return nk_to_eps(n, k)


def air_eps_from_lambda_m(lambda_m: float | np.ndarray) -> np.ndarray:
    lambda_arr = np.asarray(lambda_m, dtype=np.float64)
    return np.ones_like(lambda_arr, dtype=np.complex128)
