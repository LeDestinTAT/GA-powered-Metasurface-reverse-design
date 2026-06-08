from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "FNO" / "try2.py"
FINAL_DIR = PROJECT_ROOT / "final"

PATTERNS_PATH = FINAL_DIR / "training_patterns_11x11_current91.mat"
SPARAMS_PATH = FINAL_DIR / "Sparams_dataset_current91.mat"
SAVE_PATH_BEST = FINAL_DIR / "fno_peak_curve_best_current91.pt"
SAVE_PATH_FINAL = FINAL_DIR / "fno_peak_curve_final_current91.pt"
LOG_ROOT = "runs/fno_peak_curve_current91"


def main():
    code = SOURCE_PATH.read_text(encoding="utf-8", errors="ignore")
    replacements = {
        r'C:\Users\90740\Desktop\final\training_patterns_11x11.mat': str(PATTERNS_PATH),
        r'C:\Users\90740\Desktop\final\Sparams_dataset.mat': str(SPARAMS_PATH),
        r'C:\Users\90740\Desktop\final\fno_peak_curve_best.pt': str(SAVE_PATH_BEST),
        r'C:\Users\90740\Desktop\final\fno_peak_curve_final.pt': str(SAVE_PATH_FINAL),
        'runs/fno_peak_curve': LOG_ROOT,
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    ns = {
        "__name__": "__main__",
        "__file__": str(SOURCE_PATH),
    }
    exec(compile(code, str(SOURCE_PATH), "exec"), ns)


if __name__ == "__main__":
    main()
