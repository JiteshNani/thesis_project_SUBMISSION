"""Evaluate the slot CNN against ground-truth targets and a majority-class baseline."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.data_utils import BraTSSlotDataset, derive_slots_from_segmentation, load_extent_thresholds
from src.constants import SLOT_KEYS
from src.reproducibility import set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.slot_metrics import evaluate_slot_predictions, majority_classes, majority_predictions


def main():
    set_global_seed(int(config["project"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BraTSSlotDataset(config["data"]["manifest_path"], split="test", config=config)
    loader = DataLoader(ds, batch_size=int(config["slot_model"]["batch_size"]), shuffle=False, num_workers=0)
    model = load_slot_model(config, device)
    predictions, targets, cases = [], [], []
    for batch in loader:
        x = batch["flair"].to(device)
        p = predict_slots(model, x)
        predictions.extend(p)
        for i in range(x.shape[0]):
            targets.append({k: int(batch["slots"][k][i]) for k in SLOT_KEYS})
            cases.append(str(batch["case_id"][i]))

    cnn = evaluate_slot_predictions(predictions, targets)

    # Majority baseline is learned from TRAIN labels only, then applied unchanged to TEST.
    train_df = pd.read_csv(config["data"]["manifest_path"])
    train_df = train_df[train_df["split"] == "train"]
    thresholds = load_extent_thresholds(config["data"]["metadata_path"])
    train_targets = []
    for _, row in train_df.iterrows():
        s = derive_slots_from_segmentation(row["seg_path"], thresholds, flair_path=row["flair_path"])
        train_targets.append({k: int(s[k]) for k in SLOT_KEYS})
    majority_cls = majority_classes(train_targets)
    majority_pred = majority_predictions(majority_cls, len(targets))
    majority = evaluate_slot_predictions(majority_pred, targets)
    payload = {
        "cnn": cnn,
        "majority_classes_learned_from_training_split": majority_cls,
        "majority_baseline": majority,
        "cnn_minus_majority_micro_accuracy": cnn["micro_accuracy_all_slots"] - majority["micro_accuracy_all_slots"],
        "predictions": [{"case_id": c, "target": t, "prediction": p, "majority_prediction": m} for c, t, p, m in zip(cases, targets, predictions, majority_pred)],
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/slot_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "predictions"}, indent=2))
    print("Saved results/slot_test_metrics.json")


if __name__ == "__main__":
    main()
