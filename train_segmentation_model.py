"""Train a real 3D U-Net segmentation model; no synthetic/noisy ground-truth predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader

from src.config import config
from src.augmentation import augment_flair_and_seg
from src.data_utils import BraTSSegmentationDataset, training_classes_to_segmentation
from src.reproducibility import save_run_metadata, set_global_seed
from src.segmentation_model import build_segmentation_model


def _validation_score(model, loader, device, max_cases=None):
    """Fast model-selection score: mean WT/TC/ET Dice on validation only.

    HD95 is intentionally NOT computed during every validation epoch. The thesis
    HD95 calculation is reserved for the final held-out test evaluation using
    MONAI and original NIfTI physical spacing. This saves substantial CPU time
    without changing model selection or final segmentation methodology.
    """
    model.eval()
    scores = []

    def dice(p, g):
        p, g = p.astype(bool), g.astype(bool)
        denom = p.sum() + g.sum()
        return float(2.0 * np.logical_and(p, g).sum() / denom) if denom else 1.0

    with torch.no_grad():
        for vi, batch in enumerate(loader):
            if max_cases is not None and vi >= max_cases:
                break
            x = batch["flair"].to(device)
            logits = model(x)
            cls = torch.argmax(logits, dim=1).cpu()
            for i in range(cls.shape[0]):
                pred = training_classes_to_segmentation(cls[i]).numpy()
                gt = batch["seg_brats"][i].numpy()
                wt = dice(pred > 0, gt > 0)
                tc = dice((pred == 1) | (pred == 4), (gt == 1) | (gt == 4))
                et = dice(pred == 4, gt == 4)
                scores.append(float(np.mean([wt, tc, et])))
    return float(np.mean(scores)) if scores else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test limiter; omit for final thesis training.")
    p.add_argument("--max-val-cases", type=int, default=None, help="Smoke-test limiter for validation; omit for final thesis training.")
    args = p.parse_args()
    set_global_seed(int(config["project"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cfg = config["segmentation"]
    train_ds = BraTSSegmentationDataset(config["data"]["manifest_path"], split="train", config=config)
    val_ds = BraTSSegmentationDataset(config["data"]["manifest_path"], split="val", config=config)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(cfg.get("num_workers", 0)))
    model = build_segmentation_model(config).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("scheduler_factor", 0.5)),
        patience=int(cfg.get("scheduler_patience", 4)),
        min_lr=float(cfg.get("min_learning_rate", 2e-5)),
    )
    epochs = int(args.epochs or cfg["epochs"])
    best = -1.0
    best_epoch = 0
    no_improve = 0
    min_epochs = int(cfg.get("min_epochs", min(20, epochs)))
    early_patience = int(cfg.get("early_stopping_patience", 10))
    history = []
    checkpoint = Path(config["paths"]["models_dir"]) / "segmentation_unet_best.pth"

    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches is not None and bi >= args.max_train_batches: break
            x = batch["flair"].to(device)
            y = batch["seg"].to(device)
            x, y = augment_flair_and_seg(
                x,
                y,
                probability=float(cfg.get("augment_probability", 0.65)),
                flip_probability=float(cfg.get("flip_probability", 0.25)),
                max_rotation_degrees=float(cfg.get("max_rotation_degrees", 10.0)),
                intensity_scale=float(cfg.get("intensity_scale", 0.10)),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_dice = _validation_score(model, val_loader, device, args.max_val_cases)
        scheduler.step(val_dice)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_mean_brats_dice": val_dice,
            "learning_rate": current_lr,
        }
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} loss={row['train_loss']:.4f} val_WT_TC_ET_dice={val_dice:.4f} lr={current_lr:.2e}")
        if val_dice > best + float(cfg.get("early_stopping_min_delta", 1e-4)):
            best = val_dice
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_mean_dice": val_dice, "config": config}, checkpoint)
        else:
            no_improve += 1
        if args.max_train_batches is None and epoch >= min_epochs and no_improve >= early_patience:
            print(f"Early stopping at epoch {epoch}; best validation Dice={best:.4f} at epoch {best_epoch}.")
            break
    Path("results/segmentation_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_run_metadata(
        "results/segmentation_run_metadata.json",
        config=config,
        extra={
            "checkpoint": str(checkpoint),
            "epochs_requested": epochs,
            "epochs_executed": len(history),
            "best_epoch": best_epoch,
            "learning_rate": float(cfg["learning_rate"]),
            "best_validation_mean_dice": best,
            "scheduler": "ReduceLROnPlateau(validation WT/TC/ET Dice)",
            "early_stopping_patience": early_patience,
        },
    )
    print("Saved:", checkpoint)


if __name__ == "__main__":
    main()
