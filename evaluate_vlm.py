"""Evaluate the base hybrid structured-and-visual VLM on held-out test cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.checks import require_files
from src.config import config
from src.data_utils import BraTSSlotDataset
from src.reproducibility import set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_inference import load_vlm_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=None, help="Smoke-test limiter only.")
    args = ap.parse_args()
    require_files([
        (config["data"]["manifest_path"], "python prepare_data.py"),
        ("models/slot_model_best.pth", "python train_slot_model.py"),
        ("models/hybrid3dvlm_trainable.pth", "python train_vlm.py"),
    ])
    seed = int(config["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BraTSSlotDataset(config["data"]["manifest_path"], split="test", config=config, include_slots=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    slot_model = load_slot_model(config, device)
    vlm = load_vlm_model(config, device)
    records = []
    for idx, batch in enumerate(loader):
        if args.max_cases is not None and idx >= args.max_cases:
            break
        mri = batch["flair"].to(device)
        pred = predict_slots(slot_model, mri)[0]
        torch.manual_seed(seed + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + idx)
        generated = vlm.generate_report_detailed(mri, pred, retriever=None, case_id=str(batch["case_id"][0]))
        records.append({
            "case_id": str(batch["case_id"][0]),
            "reference": str(batch["report_text"][0]),
            "predicted_slots": pred,
            "vlm_base_raw": generated["raw_generation"],
            "vlm_base": generated["final_report"],
        })
        print(f"[{idx+1}/{len(ds)}] {records[-1]['case_id']}")
    Path("results").mkdir(exist_ok=True)
    Path("results/vlm_base_test_outputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved results/vlm_base_test_outputs.json")


if __name__ == "__main__":
    main()
