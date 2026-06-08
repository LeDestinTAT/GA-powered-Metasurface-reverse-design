from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistoryCheckpoint:
    run_name: str
    index: int
    path: Path
    epoch: int | None
    val_loss: float | None


_BEST_RE = re.compile(r"best_(?P<idx>\d+)_epoch_(?P<epoch>\d+)_val_(?P<val>[0-9eE+\-.]+)\.pt$")


def _parse_history_checkpoint(path: Path) -> HistoryCheckpoint:
    match = _BEST_RE.match(path.name)
    index = int(match.group("idx")) if match else -1
    epoch = int(match.group("epoch")) if match else None
    val_loss = float(match.group("val")) if match else None
    return HistoryCheckpoint(
        run_name=path.parent.name,
        index=index,
        path=path,
        epoch=epoch,
        val_loss=val_loss,
    )


def list_history_checkpoints(history_root: Path) -> list[HistoryCheckpoint]:
    if not history_root.exists():
        return []
    out: list[HistoryCheckpoint] = []
    for run_dir in sorted([p for p in history_root.iterdir() if p.is_dir()]):
        for ckpt in sorted(run_dir.glob("best_*.pt")):
            out.append(_parse_history_checkpoint(ckpt))
    return out


def resolve_checkpoint_path(
    mode: str,
    default_checkpoint: Path,
    history_root: Path,
    *,
    custom_path: str | Path | None = None,
    run_name: str | None = None,
    best_index: int | None = None,
    project_root: Path | None = None,
) -> tuple[Path, str]:
    mode = str(mode).lower()

    if mode == "default":
        return default_checkpoint, "default best"

    if mode == "path":
        if custom_path is None:
            raise ValueError("CHECKPOINT_MODE='path' 时需要设置 CHECKPOINT_CUSTOM_PATH")
        path = Path(custom_path)
        if not path.is_absolute():
            if project_root is None:
                raise ValueError("相对 checkpoint 路径需要提供 project_root")
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"未找到 checkpoint: {path}")
        return path, "custom path"

    if mode == "history":
        if not history_root.exists():
            raise FileNotFoundError(f"未找到历史模型目录: {history_root}")

        run_dirs = sorted([p for p in history_root.iterdir() if p.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"历史模型目录为空: {history_root}")

        if run_name is None:
            run_dir = run_dirs[-1]
        else:
            run_dir = history_root / run_name
            if not run_dir.exists():
                raise FileNotFoundError(f"未找到指定训练轮次目录: {run_dir}")

        candidates = sorted(run_dir.glob("best_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"目录中没有历史 best 模型: {run_dir}")

        if best_index is None:
            ckpt = candidates[-1]
        else:
            idx = int(best_index) - 1
            if idx < 0 or idx >= len(candidates):
                raise IndexError(f"CHECKPOINT_BEST_INDEX={best_index} 超出范围 1..{len(candidates)}")
            ckpt = candidates[idx]
        return ckpt, f"history run={run_dir.name}"

    raise ValueError(f"未知 CHECKPOINT_MODE: {mode}")


def _resolve_run_dir(history_root: Path, run_name: str | None) -> Path:
    if not history_root.exists():
        raise FileNotFoundError(f"未找到历史模型目录: {history_root}")

    run_dirs = sorted([p for p in history_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"历史模型目录为空: {history_root}")

    if run_name is None:
        return run_dirs[-1]

    run_dir = history_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到指定训练轮次目录: {run_dir}")
    return run_dir


def resolve_checkpoint_choice(
    choice: str,
    *,
    current_best: Path,
    current_final: Path,
    history_root: Path,
    custom_path: str | Path | None = None,
    run_name: str | None = None,
    best_index: int | None = None,
    project_root: Path | None = None,
) -> tuple[Path, str]:
    choice = str(choice).lower().strip()

    if choice in {"current_best", "default", "best"}:
        return current_best, "current best"

    if choice in {"current_final", "final"}:
        return current_final, "current final"

    if choice == "path":
        return resolve_checkpoint_path(
            "path",
            current_best,
            history_root,
            custom_path=custom_path,
            run_name=run_name,
            best_index=best_index,
            project_root=project_root,
        )

    if choice in {"latest_history_best", "history_best", "history_best_index"}:
        return resolve_checkpoint_path(
            "history",
            current_best,
            history_root,
            custom_path=custom_path,
            run_name=run_name,
            best_index=best_index,
            project_root=project_root,
        )

    if choice in {"latest_run_best", "run_best", "history_run_best"}:
        run_dir = _resolve_run_dir(history_root, run_name)
        path = run_dir / "run_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"未找到 run_best.pt: {path}")
        return path, f"run_best run={run_dir.name}"

    if choice in {"latest_run_final", "run_final", "history_run_final"}:
        run_dir = _resolve_run_dir(history_root, run_name)
        path = run_dir / "run_final.pt"
        if not path.exists():
            raise FileNotFoundError(f"未找到 run_final.pt: {path}")
        return path, f"run_final run={run_dir.name}"

    raise ValueError(f"未知 MODEL_CHOICE / checkpoint choice: {choice}")
