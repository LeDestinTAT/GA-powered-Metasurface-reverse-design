import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_utils import list_history_checkpoints
from src.project_paths import BEST_MODEL_HISTORY_ROOT, MODELS_CURRENT_DIR


def main() -> None:
    current_dir = MODELS_CURRENT_DIR
    history_root = BEST_MODEL_HISTORY_ROOT

    print(f"当前模型目录: {current_dir}")
    current_files = sorted(current_dir.glob("*.pt"))
    if not current_files:
        print("  没有找到 current checkpoint。")
    else:
        print("  current checkpoints:")
        for path in current_files:
            print(f"    - {path.name}")

    print()
    print(f"历史 best 目录: {history_root}")
    checkpoints = list_history_checkpoints(history_root)
    if not checkpoints:
        print("  没有找到历史 best checkpoint。")
        return

    grouped: dict[str, list] = {}
    for item in checkpoints:
        grouped.setdefault(item.run_name, []).append(item)

    for run_name, items in grouped.items():
        print(f"[run] {run_name}")
        run_dir = items[0].path.parent
        run_best = run_dir / "run_best.pt"
        run_final = run_dir / "run_final.pt"
        print(f"  run_best:  {'yes' if run_best.exists() else 'no'}")
        print(f"  run_final: {'yes' if run_final.exists() else 'no'}")
        for item in items:
            epoch_text = "?" if item.epoch is None else str(item.epoch)
            val_text = "?" if item.val_loss is None else f"{item.val_loss:.6e}"
            print(f"  {item.index:03d} | epoch={epoch_text} | val={val_text} | {item.path.name}")
        print()

    latest_run = sorted(grouped.keys())[-1]
    latest_items = grouped[latest_run]
    latest_best = latest_items[-1]
    print("推理脚本可直接这样选:")
    print('  MODEL_CHOICE = "history_best"')
    print(f'  MODEL_RUN_NAME = "{latest_run}"')
    print(f"  MODEL_BEST_INDEX = {latest_best.index}")
    print('  MODEL_CHOICE = "latest_run_best"')
    print('  MODEL_CHOICE = "latest_run_final"')


if __name__ == "__main__":
    main()
