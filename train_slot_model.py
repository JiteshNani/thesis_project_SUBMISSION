"""Train the leakage-free multi-task 3D CNN slot predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import config
from src.augmentation import augment_flair_and_seg
from src.data_utils import BraTSSlotDataset, derive_slots_from_segmentation, load_extent_thresholds
from src.constants import SLOT_KEYS
from src.reproducibility import save_run_metadata, set_global_seed
from src.slot_inference import build_slot_model, predict_slots
from src.slot_metrics import evaluate_slot_predictions


def _class_weights() -> Dict[str, torch.Tensor | None]:
    df = pd.read_csv(config["data"]["manifest_path"])
    df = df[df["split"] == "train"]
    thresholds = load_extent_thresholds(config["data"]["metadata_path"])
    labels = {k: [] for k in SLOT_KEYS}
    for _, row in df.iterrows():
        slots = derive_slots_from_segmentation(row["seg_path"], thresholds, flair_path=row["flair_path"])
        for k in SLOT_KEYS:
            labels[k].append(int(slots[k]))
    weights = {}
    for key in SLOT_KEYS:
        nclasses = 3 if key == "extent" else 2
        counts = np.bincount(labels[key], minlength=nclasses).astype(np.float64)
        if np.count_nonzero(counts) < 2:
            weights[key] = None
            print(f"{key}: non-informative training target distribution {counts.astype(int).tolist()} (no class weighting)")
            continue
        w = np.zeros_like(counts)
        nonzero = counts > 0
        w[nonzero] = counts.sum() / (nonzero.sum() * counts[nonzero])
        weights[key] = torch.tensor(w, dtype=torch.float32)
        print(f"{key}: counts={counts.astype(int).tolist()} weights={w.round(3).tolist()}")
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test limiter; omit for final thesis training.")
    args = parser.parse_args()

    seed = int(config["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = BraTSSlotDataset(config["data"]["manifest_path"], split="train", config=config)
    val_ds = BraTSSlotDataset(config["data"]["manifest_path"], split="val", config=config)
    cfg = config["slot_model"]
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg.get("num_workers", 0)))

    model = build_slot_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    weights = _class_weights() if bool(cfg.get("use_class_weights", True)) else {k: None for k in SLOT_KEYS}
    criteria = {k: nn.CrossEntropyLoss(weight=(weights[k].to(device) if weights[k] is not None else None)) for k in SLOT_KEYS}
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(args.epochs or cfg["epochs"])
    history = []
    best_score = -1.0
    out_path = Path(config["paths"]["models_dir"]) / "slot_model_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches is not None and bi >= args.max_train_batches:
                break
            x = batch["flair"].to(device)
            x, _ = augment_flair_and_seg(x, probability=0.5)
            targets = {k: batch["slots"][k].long().to(device) for k in SLOT_KEYS}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x)
                loss = sum(criteria[k](outputs[k], targets[k]) for k in SLOT_KEYS)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        pred, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["flair"].to(device)
                pred.extend(predict_slots(model, x))
                bs = x.shape[0]
                true.extend([{k: int(batch["slots"][k][i]) for k in SLOT_KEYS} for i in range(bs)])
        metrics = evaluate_slot_predictions(pred, true)
        score = metrics["macro_balanced_accuracy_informative_slots"]
        if score is None:
            score = metrics["micro_accuracy_all_slots"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "val_metrics": metrics}
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} loss={row['train_loss']:.4f} val_bal_acc={score:.4f} val_micro_acc={metrics['micro_accuracy_all_slots']:.4f}")
        if score > best_score:
            best_score = float(score)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_metrics": metrics, "config": config}, out_path)

    Path("results").mkdir(exist_ok=True)
    Path("results/slot_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_run_metadata("results/slot_run_metadata.json", config=config, extra={"checkpoint": str(out_path), "epochs_executed": epochs, "learning_rate": float(cfg["learning_rate"]), "best_validation_score": best_score})
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
