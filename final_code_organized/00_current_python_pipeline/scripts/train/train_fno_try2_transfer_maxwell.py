from __future__ import annotations

import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train.train_fno_curvefield_hybrid as base
from src.fullfield_dual_surrogate import MODEL_FAMILY_TRY2_TRANSFER, Try2CurveFieldTransferModel
from src.project_paths import (
    BEST_MODEL_HISTORY_ROOT,
    FIELD_DATA_DIR,
    MODELS_CURRENT_DIR,
    SAMPLING_META_PATH,
    TENSORBOARD_RUNS_DIR,
    TRAIN_RUN_OUTPUTS_DIR,
    ensure_standard_dirs,
)


print("TensorBoard: run tensorboard --logdir logs/tensorboard/runs and open http://localhost:6006/")


DATA_DIR = FIELD_DATA_DIR
META_PATH = SAMPLING_META_PATH
SAVE_PATH_FINAL = MODELS_CURRENT_DIR / "fno_try2_transfer_maxwell_final.pt"
SAVE_PATH_BEST = MODELS_CURRENT_DIR / "fno_try2_transfer_maxwell_best.pt"
BEST_HISTORY_ROOT = BEST_MODEL_HISTORY_ROOT
RUN_OUTPUTS_ROOT = TRAIN_RUN_OUTPUTS_DIR

LEGACY_INIT_CKPT_CANDIDATES = (
    PROJECT_ROOT / "final" / "fno_peak_curve_best_current91.pt",
    PROJECT_ROOT / "final" / "fno_peak_curve_best.pt",
)

SEED = 42
TRAIN_RATIO = 0.85
TRAIN_SAMPLE_LIMIT = 2000
VAL_SAMPLE_LIMIT = 256

EPOCHS = 80
VAL_EVERY = 5
MIN_EPOCHS = 20
PATIENCE = 10
TRAIN_PROGRESS_EVERY = 10

BATCH_SAMPLES = 12
VAL_BATCH_SAMPLES = 12
NUM_WORKERS = 0
PIN_MEMORY = True

DOWN_X = 3
DOWN_Y = 3
DOWN_Z = 3

MODES_X = 6
MODES_Y = 6
MODES_Z = 6
WIDTH = 64
DEPTH = 4
LAM_FF = 8
HEAD_HIDDEN = 256
FIELD_WIDTH = 32
FIELD_DEPTH = 2

PRETRAINED_LR = 6.0e-5
NEW_HEAD_LR = 2.0e-4
WEIGHT_DECAY = 5.0e-5
GRAD_CLIP = 1.0
USE_AMP = True
AMP_DTYPE = "bfloat16"

LAMBDA_CURVE_S11 = 0.95
LAMBDA_CURVE_A = 2.10
LAMBDA_MAIN_PEAK_POS = 1.80
LAMBDA_MAIN_PEAK_HEIGHT = 1.25
LAMBDA_SECONDARY_PEAK_HEIGHT = 0.65
LAMBDA_MAIN_PEAK_CLASS = 0.0
LAMBDA_PEAK_RANK = 0.04
LAMBDA_CURVE_TV = 0.010
LAMBDA_FIELD = 0.22
LAMBDA_PASSIVE = 0.03
LAMBDA_CURL_E = 0.0025
LAMBDA_CURL_H = 0.0025
LAMBDA_DIV = 0.0

FIELD_START_EPOCH = 4
PHYSICS_START_EPOCH = 14
PHYSICS_WARMUP_EPOCHS = 24
PHYSICS_LOSS_INTERVAL = 4
VAL_WITH_PHYSICS = False

LOG_ROOT = TENSORBOARD_RUNS_DIR / "fno_try2_transfer_maxwell"


def configure_base_module() -> None:
    base.SEED = SEED
    base.TRAIN_RATIO = TRAIN_RATIO
    base.TRAIN_SAMPLE_LIMIT = TRAIN_SAMPLE_LIMIT
    base.VAL_SAMPLE_LIMIT = VAL_SAMPLE_LIMIT
    base.EPOCHS = EPOCHS
    base.VAL_EVERY = VAL_EVERY
    base.MIN_EPOCHS = MIN_EPOCHS
    base.PATIENCE = PATIENCE
    base.TRAIN_PROGRESS_EVERY = TRAIN_PROGRESS_EVERY
    base.BATCH_SAMPLES = BATCH_SAMPLES
    base.VAL_BATCH_SAMPLES = VAL_BATCH_SAMPLES
    base.NUM_WORKERS = NUM_WORKERS
    base.PIN_MEMORY = PIN_MEMORY
    base.DOWN_X = DOWN_X
    base.DOWN_Y = DOWN_Y
    base.DOWN_Z = DOWN_Z
    base.MODES_X = MODES_X
    base.MODES_Y = MODES_Y
    base.MODES_Z = MODES_Z
    base.WIDTH = WIDTH
    base.DEPTH = DEPTH
    base.LAM_FF = LAM_FF
    base.HEAD_HIDDEN = HEAD_HIDDEN
    base.FIELD_WIDTH = FIELD_WIDTH
    base.FIELD_DEPTH = FIELD_DEPTH
    base.WEIGHT_DECAY = WEIGHT_DECAY
    base.GRAD_CLIP = GRAD_CLIP
    base.USE_AMP = USE_AMP
    base.AMP_DTYPE = AMP_DTYPE
    base.LAMBDA_CURVE_S11 = LAMBDA_CURVE_S11
    base.LAMBDA_CURVE_A = LAMBDA_CURVE_A
    base.LAMBDA_MAIN_PEAK_POS = LAMBDA_MAIN_PEAK_POS
    base.LAMBDA_MAIN_PEAK_HEIGHT = LAMBDA_MAIN_PEAK_HEIGHT
    base.LAMBDA_SECONDARY_PEAK_HEIGHT = LAMBDA_SECONDARY_PEAK_HEIGHT
    base.LAMBDA_MAIN_PEAK_CLASS = LAMBDA_MAIN_PEAK_CLASS
    base.LAMBDA_PEAK_RANK = LAMBDA_PEAK_RANK
    base.LAMBDA_CURVE_TV = LAMBDA_CURVE_TV
    base.LAMBDA_FIELD = LAMBDA_FIELD
    base.LAMBDA_PASSIVE = LAMBDA_PASSIVE
    base.LAMBDA_CURL_E = LAMBDA_CURL_E
    base.LAMBDA_CURL_H = LAMBDA_CURL_H
    base.LAMBDA_DIV = LAMBDA_DIV
    base.FIELD_START_EPOCH = FIELD_START_EPOCH
    base.PHYSICS_START_EPOCH = PHYSICS_START_EPOCH
    base.PHYSICS_WARMUP_EPOCHS = PHYSICS_WARMUP_EPOCHS
    base.PHYSICS_LOSS_INTERVAL = PHYSICS_LOSS_INTERVAL
    base.VAL_WITH_PHYSICS = VAL_WITH_PHYSICS


def find_init_checkpoint() -> Path:
    for path in LEGACY_INIT_CKPT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No legacy try2 checkpoint found. Expected one of: "
        + ", ".join(str(p) for p in LEGACY_INIT_CKPT_CANDIDATES)
    )


def load_try2_weights(model: Try2CurveFieldTransferModel, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()
    loaded = []
    skipped = []

    def try_copy(dst_name: str, src_name: str) -> None:
        src_tensor = state_dict.get(src_name)
        dst_tensor = target_state.get(dst_name)
        if src_tensor is None or dst_tensor is None:
            skipped.append((src_name, dst_name, "missing"))
            return
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            skipped.append((src_name, dst_name, f"shape {tuple(src_tensor.shape)} != {tuple(dst_tensor.shape)}"))
            return
        target_state[dst_name] = src_tensor.detach().clone().to(dtype=dst_tensor.dtype)
        loaded.append((src_name, dst_name))

    try_copy("pattern_encoder.in_proj.weight", "encoder.in_proj.weight")
    try_copy("pattern_encoder.in_proj.bias", "encoder.in_proj.bias")
    try_copy("pattern_encoder.out_norm.weight", "encoder.out_norm.weight")
    try_copy("pattern_encoder.out_norm.bias", "encoder.out_norm.bias")

    for layer_idx in range(DEPTH):
        try_copy(f"pattern_encoder.spectral.{layer_idx}.weight_real", f"encoder.spectral.{layer_idx}.weight_real")
        try_copy(f"pattern_encoder.spectral.{layer_idx}.weight_imag", f"encoder.spectral.{layer_idx}.weight_imag")
        try_copy(f"pattern_encoder.pointwise.{layer_idx}.weight", f"encoder.pointwise.{layer_idx}.weight")
        try_copy(f"pattern_encoder.pointwise.{layer_idx}.bias", f"encoder.pointwise.{layer_idx}.bias")

    for seq_idx in (0, 2, 4):
        try_copy(f"head.{seq_idx}.weight", f"head.{seq_idx}.weight")
        try_copy(f"head.{seq_idx}.bias", f"head.{seq_idx}.bias")

    model.load_state_dict(target_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_tensors": len(loaded),
        "skipped_tensors": len(skipped),
        "loaded_pairs": [f"{src}->{dst}" for src, dst in loaded],
        "first_skips": [f"{src}->{dst}: {reason}" for src, dst, reason in skipped[:10]],
        "source_best_epoch": checkpoint.get("best_epoch"),
        "source_best_peak_score": checkpoint.get("best_peak_score"),
        "source_best_val_loss": checkpoint.get("best_val_loss"),
    }


def build_optimizer(model: Try2CurveFieldTransferModel):
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("pattern_encoder.") or name.startswith("head."):
            pretrained_params.append(param)
        else:
            new_params.append(param)
    return optim.AdamW(
        [
            {"params": pretrained_params, "lr": PRETRAINED_LR},
            {"params": new_params, "lr": NEW_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_checkpoint_payload(state_dict, best_epoch_value, best_val_loss_value, lambda_vec, init_report):
    return {
        "state_dict": state_dict,
        "config": {
            "MODEL_FAMILY": MODEL_FAMILY_TRY2_TRANSFER,
            "MODES_X": MODES_X,
            "MODES_Y": MODES_Y,
            "MODES_Z": MODES_Z,
            "WIDTH": WIDTH,
            "DEPTH": DEPTH,
            "FIELD_WIDTH": FIELD_WIDTH,
            "FIELD_DEPTH": FIELD_DEPTH,
            "LAM_FF": LAM_FF,
            "HEAD_HIDDEN": HEAD_HIDDEN,
            "DOWN_X": DOWN_X,
            "DOWN_Y": DOWN_Y,
            "DOWN_Z": DOWN_Z,
            "BATCH_SAMPLES": BATCH_SAMPLES,
            "VAL_BATCH_SAMPLES": VAL_BATCH_SAMPLES,
            "PRETRAINED_LR": PRETRAINED_LR,
            "NEW_HEAD_LR": NEW_HEAD_LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LAMBDA_CURVE_S11": LAMBDA_CURVE_S11,
            "LAMBDA_CURVE_A": LAMBDA_CURVE_A,
            "LAMBDA_MAIN_PEAK_POS": LAMBDA_MAIN_PEAK_POS,
            "LAMBDA_MAIN_PEAK_HEIGHT": LAMBDA_MAIN_PEAK_HEIGHT,
            "LAMBDA_SECONDARY_PEAK_HEIGHT": LAMBDA_SECONDARY_PEAK_HEIGHT,
            "LAMBDA_MAIN_PEAK_CLASS": LAMBDA_MAIN_PEAK_CLASS,
            "LAMBDA_PEAK_RANK": LAMBDA_PEAK_RANK,
            "LAMBDA_CURVE_TV": LAMBDA_CURVE_TV,
            "LAMBDA_FIELD": LAMBDA_FIELD,
            "LAMBDA_PASSIVE": LAMBDA_PASSIVE,
            "LAMBDA_CURL_E": LAMBDA_CURL_E,
            "LAMBDA_CURL_H": LAMBDA_CURL_H,
            "LAMBDA_DIV": LAMBDA_DIV,
            "FIELD_START_EPOCH": FIELD_START_EPOCH,
            "PHYSICS_START_EPOCH": PHYSICS_START_EPOCH,
            "PHYSICS_WARMUP_EPOCHS": PHYSICS_WARMUP_EPOCHS,
            "PHYSICS_LOSS_INTERVAL": PHYSICS_LOSS_INTERVAL,
            "FIELD_POINT_COUNT": 2,
            "FIELD_POINT_WEIGHTS": base.FIELD_POINT_WEIGHTS.tolist(),
            "T_ZERO_OVERRIDE": True,
            "INIT_FROM_TRY2": True,
            "INIT_CHECKPOINT": init_report["checkpoint_path"],
            "SEED": SEED,
        },
        "best_epoch": best_epoch_value,
        "best_val_loss": best_val_loss_value,
        "lambda_vec": torch.tensor(lambda_vec, dtype=torch.float32),
    }


def main():
    ensure_standard_dirs()
    base.set_seed(SEED)
    configure_base_module()

    meta = base.load_mat_auto(META_PATH)
    sample_files = sorted(DATA_DIR.glob("sample_*.mat"))
    if not sample_files:
        raise RuntimeError(f"No sample_*.mat files found under {DATA_DIR}")

    perm = np.random.default_rng(SEED).permutation(len(sample_files))
    n_train = int(TRAIN_RATIO * len(sample_files))
    train_files = sorted([sample_files[i] for i in perm[:n_train]], key=lambda p: p.name)
    val_files = sorted([sample_files[i] for i in perm[n_train:]], key=lambda p: p.name)
    if TRAIN_SAMPLE_LIMIT is not None:
        train_files = train_files[: int(TRAIN_SAMPLE_LIMIT)]
    if VAL_SAMPLE_LIMIT is not None:
        val_files = val_files[: int(VAL_SAMPLE_LIMIT)]
    if not val_files:
        raise RuntimeError("Validation set is empty. Check TRAIN_RATIO / VAL_SAMPLE_LIMIT.")

    train_ds = base.CurveFieldHybridDataset(train_files, meta, train=True)
    val_ds = base.CurveFieldHybridDataset(val_files, meta, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SAMPLES,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SAMPLES,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )

    print("Train sample files =", len(train_files))
    print("Val sample files =", len(val_files))
    print("Train batches =", len(train_loader))
    print("Val batches =", len(val_loader))
    print("Curve length =", len(train_ds.lambda_vec))
    print("Field wavelengths per sample = 2 (main peak / secondary peak)")
    print("Spatial shape =", train_ds.target_shape)
    print("Transfer mode = try2 curve backbone -> field + Maxwell finetuning")
    print(f"Field supervision starts at epoch {FIELD_START_EPOCH}, physics starts at epoch {PHYSICS_START_EPOCH}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device.type)
    if device.type == "cuda":
        print("AMP dtype =", AMP_DTYPE)

    model = Try2CurveFieldTransferModel(
        modes_x=MODES_X,
        modes_y=MODES_Y,
        modes_z=MODES_Z,
        width=WIDTH,
        depth=DEPTH,
        lam_ff=LAM_FF,
        head_hidden=HEAD_HIDDEN,
        field_width=FIELD_WIDTH,
        field_depth=FIELD_DEPTH,
    ).to(device)

    init_ckpt = find_init_checkpoint()
    init_report = load_try2_weights(model, init_ckpt)
    print("Initialized from try2 checkpoint:", init_report["checkpoint_path"])
    print("Transferred tensors:", init_report["loaded_tensors"], "| skipped:", init_report["skipped_tensors"])

    has_complex_params = any(torch.is_complex(param) for param in model.parameters())
    scaler_enabled = USE_AMP and device.type == "cuda" and not has_complex_params
    if USE_AMP and device.type == "cuda" and has_complex_params:
        print("Model contains complex parameters, so GradScaler is disabled while autocast remains enabled.")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=2e-6,
    )

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(LOG_ROOT / run_name))
    run_output_dir = RUN_OUTPUTS_ROOT / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    best_history_dir = BEST_HISTORY_ROOT / run_name
    best_history_dir.mkdir(parents=True, exist_ok=True)
    run_best_path = best_history_dir / "run_best.pt"
    run_final_path = best_history_dir / "run_final.pt"
    coords_base = {axis: tensor.to(device) for axis, tensor in train_ds.coord_tensors.items()}

    metric_keys = (
        "total",
        "curve_s11",
        "curve_a",
        "main_peak_pos",
        "main_peak_height",
        "secondary_peak_height",
        "main_peak_class",
        "peak_rank",
        "curve_tv",
        "field",
        "passive",
        "curl_e",
        "curl_h",
        "div",
    )

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    best_snapshot_count = 0
    bad_epochs = 0
    global_step = 0
    train_hist = []
    val_hist = []
    use_autocast = USE_AMP and device.type == "cuda"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_ds.refresh(epoch)
        train_ds.set_field_loading(epoch >= FIELD_START_EPOCH)
        val_ds.set_field_loading(epoch >= FIELD_START_EPOCH)
        sums = {key: 0.0 for key in metric_keys}
        count = 0
        num_train_batches = len(train_loader)
        nonfinite_batches = 0
        print(f"Epoch {epoch:03d} started | train_batches={num_train_batches} | field={'on' if epoch >= FIELD_START_EPOCH else 'off'}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx == 1:
                print(f"Epoch {epoch:03d} first batch loaded.")
            batch = base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            apply_physics = epoch >= PHYSICS_START_EPOCH and (global_step % PHYSICS_LOSS_INTERVAL == 0)
            include_field = epoch >= FIELD_START_EPOCH and batch["field_x"].shape[1] > 0

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_autocast,
                dtype=base.get_amp_dtype() if use_autocast else None,
            ):
                pred_curve_s_raw, pred_field, pred_main_peak_logits = base.forward_model(model, batch, include_field=include_field)
                loss, stats = base.compute_total_loss(
                    pred_curve_s_raw,
                    pred_field,
                    pred_main_peak_logits,
                    batch,
                    coords_base,
                    epoch,
                    True,
                    apply_physics,
                )

            if not torch.isfinite(loss):
                nonfinite_batches += 1
                if nonfinite_batches <= 8:
                    print(f"Epoch {epoch:03d} found non-finite loss, skip batch {batch_idx}/{num_train_batches}")
                continue

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            bs = batch["pattern_xy"].shape[0]
            for key in metric_keys:
                sums[key] += stats[key] * bs
            count += bs
            global_step += 1
            if (batch_idx % TRAIN_PROGRESS_EVERY == 0) or (batch_idx == num_train_batches):
                print(f"Epoch {epoch:03d} progress {batch_idx}/{num_train_batches} | running_train={sums['total'] / max(count, 1):.6e}")

        train_metrics = {f"train_{key}": sums[key] / max(count, 1) for key in metric_keys}
        train_total = train_metrics["train_total"]

        do_val = (epoch % VAL_EVERY == 0) or (epoch == 1)
        if do_val:
            val_metrics = base.evaluate(model, val_loader, coords_base, device, epoch)
            val_total = val_metrics["val_total"]
            old_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler.step(val_total)
            new_lrs = [group["lr"] for group in optimizer.param_groups]
            if new_lrs != old_lrs:
                print("[Scheduler] LR reduced:", " | ".join(f"{old:.3e}->{new:.3e}" for old, new in zip(old_lrs, new_lrs)))
        else:
            val_metrics = {f"val_{key}": float("nan") for key in metric_keys}
            val_total = float("nan")

        train_hist.append(train_total)
        val_hist.append(val_total)
        for key, value in train_metrics.items():
            writer.add_scalar(f"loss/{key}", value, epoch)
        writer.add_scalar("loss/physics_ramp", base.physics_ramp(epoch), epoch)
        writer.add_scalar("lr/pretrained", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("lr/new", optimizer.param_groups[1]["lr"], epoch)
        if do_val:
            for key, value in val_metrics.items():
                writer.add_scalar(f"loss/{key}", value, epoch)

        if do_val and epoch >= FIELD_START_EPOCH and val_total < best_score:
            best_score = val_total
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            checkpoint_payload = build_checkpoint_payload(best_state, best_epoch, best_score, train_ds.lambda_vec, init_report)
            torch.save(checkpoint_payload, SAVE_PATH_BEST)
            torch.save(checkpoint_payload, run_best_path)
            best_snapshot_count += 1
            numbered_best_path = best_history_dir / f"best_{best_snapshot_count:03d}_epoch_{best_epoch:03d}_val_{best_score:.6e}.pt"
            torch.save(checkpoint_payload, numbered_best_path)
        elif do_val and epoch >= FIELD_START_EPOCH:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | train={train_total:.6e} | val={val_total:.6e} | "
            f"curveS={train_metrics['train_curve_s11']:.4e} | curveA={train_metrics['train_curve_a']:.4e} | "
            f"mainPos={train_metrics['train_main_peak_pos']:.4e} | mainH={train_metrics['train_main_peak_height']:.4e} | "
            f"secondH={train_metrics['train_secondary_peak_height']:.4e} | rank={train_metrics['train_peak_rank']:.4e} | "
            f"field={train_metrics['train_field']:.4e} | curlE={train_metrics['train_curl_e']:.4e} | "
            f"curlH={train_metrics['train_curl_h']:.4e} | best_epoch={best_epoch}"
        )

        if do_val and epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    final_payload = build_checkpoint_payload(model.state_dict(), best_epoch, best_score, train_ds.lambda_vec, init_report)
    torch.save(final_payload, SAVE_PATH_FINAL)
    torch.save(final_payload, run_final_path)
    writer.close()

    summary = {
        "run_name": run_name,
        "model_family": MODEL_FAMILY_TRY2_TRANSFER,
        "init_checkpoint": init_report["checkpoint_path"],
        "loaded_tensors": init_report["loaded_tensors"],
        "skipped_tensors": init_report["skipped_tensors"],
        "best_epoch": best_epoch,
        "best_val_loss": best_score,
        "train_total_last": train_hist[-1] if train_hist else None,
        "val_total_last": val_hist[-1] if val_hist else None,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "batch_samples": BATCH_SAMPLES,
        "field_points_per_sample": 2,
        "downsample": {"x": DOWN_X, "y": DOWN_Y, "z": DOWN_Z},
    }
    with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"  init checkpoint: {init_report['checkpoint_path']}")
    print(f"  best model:  {SAVE_PATH_BEST}")
    print(f"  final model: {SAVE_PATH_FINAL}")
    print(f"  best history dir: {best_history_dir}")
    print(f"  this run best:    {run_best_path}")
    print(f"  this run final:   {run_final_path}")
    print(f"  best_epoch = {best_epoch}, best_val_loss = {best_score:.6e}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
