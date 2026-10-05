"""Validation-only gate for the V3.1 VLM/RAG design.

This script loads only the validation split. It is the architecture freeze gate
before the single held-out test evaluation. Ground-truth validation slots are
used only for evaluation, never generation or candidate selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checks import require_files
from src.config import config
from src.constants import SLOT_KEYS
from src.data_utils import BraTSSlotDataset
from src.rag_retriever import RAGRetriever
from src.report_metrics import (
    generation_diagnostics,
    prompt_leakage_flags,
    report_slot_consistency,
    structure_compliance,
    truncation_flag,
    unsupported_content_flags,
)
from src.reproducibility import set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_inference import load_vlm_model


def _plain_batch_slots(batch_slots: Dict) -> Dict[str, int]:
    out = {}
    for key in SLOT_KEYS:
        value = batch_slots[key]
        if hasattr(value, "reshape"):
            value = value.reshape(-1)[0]
        if hasattr(value, "item"):
            value = value.item()
        out[key] = int(value)
    return out


def _empty_acc():
    return {k: [] for k in (
        "raw_conditioning", "final_conditioning", "raw_ground_truth", "final_ground_truth",
        "raw_structure", "final_structure", "raw_flags", "final_flags",
        "raw_prompt_leakage", "final_prompt_leakage", "raw_truncation", "final_truncation",
    )}


def _add(acc, raw, final, pred_slots, gt_slots):
    acc["raw_conditioning"].append(report_slot_consistency(raw, pred_slots)["mean"])
    acc["final_conditioning"].append(report_slot_consistency(final, pred_slots)["mean"])
    acc["raw_ground_truth"].append(report_slot_consistency(raw, gt_slots)["mean"])
    acc["final_ground_truth"].append(report_slot_consistency(final, gt_slots)["mean"])
    acc["raw_structure"].append(float(structure_compliance(raw)))
    acc["final_structure"].append(float(structure_compliance(final)))
    acc["raw_flags"].append(float(bool(unsupported_content_flags(raw))))
    acc["final_flags"].append(float(bool(unsupported_content_flags(final))))
    acc["raw_prompt_leakage"].append(float(bool(prompt_leakage_flags(raw))))
    acc["final_prompt_leakage"].append(float(bool(prompt_leakage_flags(final))))
    acc["raw_truncation"].append(float(truncation_flag(raw)))
    acc["final_truncation"].append(float(truncation_flag(final)))


def _summary(acc):
    return {k: float(np.mean(v)) if v else None for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=None, help="Optional quick validation subset; omit before freezing V3.1.")
    args = ap.parse_args()

    require_files([
        (config["data"]["manifest_path"], "python prepare_data.py"),
        ("models/slot_model_best.pth", "python train_slot_model.py"),
        ("models/hybrid3dvlm_trainable.pth", "python train_vlm.py"),
        (Path(config["rag"]["index_path"]) / "faiss_index.bin", "python build_rag_index.py"),
        (Path(config["rag"]["index_path"]) / "reports.pkl", "python build_rag_index.py"),
    ])

    seed = int(config["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BraTSSlotDataset(config["data"]["manifest_path"], split="val", config=config, include_slots=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    slot_model = load_slot_model(config, device)
    vlm = load_vlm_model(config, device)
    rag = RAGRetriever(config)
    rag._load()

    base_acc, rag_acc = _empty_acc(), _empty_acc()
    records = []
    selections = []
    for idx, batch in enumerate(loader):
        if args.max_cases is not None and idx >= args.max_cases:
            break
        case_id = str(batch["case_id"][0])
        mri = batch["flair"].to(device)
        pred = predict_slots(slot_model, mri)[0]
        gt = _plain_batch_slots(batch["slots"])

        torch.manual_seed(seed + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + idx)
        base = vlm.generate_report_detailed(mri, pred, retriever=None, case_id=case_id)

        torch.manual_seed(seed + 100000 + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100000 + idx)
        rag_out = vlm.generate_report_detailed(mri, pred, retriever=rag, case_id=case_id)

        _add(base_acc, base["raw_generation"], base["final_report"], pred, gt)
        _add(rag_acc, rag_out["raw_generation"], rag_out["final_report"], pred, gt)
        selections.append(str(rag_out.get("selection")))
        records.append({
            "case_id": case_id,
            "predicted_slots": pred,
            "ground_truth_slots": gt,
            "base_raw": base["raw_generation"],
            "base_final": base["final_report"],
            "base_raw_diagnostics": generation_diagnostics(base["raw_generation"], pred),
            "rag_context": rag_out.get("rag_context", ""),
            "rag_raw": rag_out["raw_generation"],
            "rag_final": rag_out["final_report"],
            "rag_raw_diagnostics": generation_diagnostics(rag_out["raw_generation"], pred),
            "rag_selection": rag_out.get("selection"),
            "base_candidate_diagnostics": rag_out.get("base_candidate_diagnostics"),
            "rag_candidate_diagnostics": rag_out.get("rag_candidate_diagnostics"),
        })
        print(f"[{idx+1}/{len(ds)}] {case_id}")

    base = _summary(base_acc)
    rag_s = _summary(rag_acc)
    n = max(1, len(selections))
    rag_s["rag_candidate_acceptance_rate"] = float(sum(x == "rag_candidate" for x in selections) / n)
    rag_s["rag_fallback_rate"] = float(sum(x == "base_fallback" for x in selections) / n)

    summary = {
        "split": "validation",
        "n_cases": len(records),
        "test_split_used": False,
        "base_vlm": base,
        "vlm_rag": rag_s,
        "interpretation": (
            "Raw metrics judge the language-model generator. Final metrics judge the delivered slot-grounded pipeline. "
            "RAG candidate selection uses predicted slots and text-quality diagnostics only; validation ground truth is evaluation-only."
        ),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/v3_validation_outputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("results/v3_validation_report_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # V3.1 freeze gate. Do not weaken this after seeing test data.
    b, r = summary["base_vlm"], summary["vlm_rag"]
    eps = 1e-9
    if b["final_conditioning"] < 0.999 - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: base final conditioning consistency < 0.999")
    if r["final_conditioning"] < 0.999 - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: RAG final conditioning consistency < 0.999")
    if b["final_structure"] < 0.999 - eps or r["final_structure"] < 0.999 - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: delivered reports are not exactly two-section structured")
    if b["final_prompt_leakage"] > eps or r["final_prompt_leakage"] > eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: prompt/meta leakage remains in delivered reports")
    if b["final_flags"] > eps or r["final_flags"] > eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: unsupported-content flags remain in delivered reports")
    if b["final_truncation"] > eps or r["final_truncation"] > eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: delivered report truncation/incompleteness detected")
    if r["raw_prompt_leakage"] > 0.02 + eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: RAG raw prompt leakage > 2%")
    if r["raw_conditioning"] < 0.85 - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: RAG raw conditioning consistency < 0.85")
    if r["raw_conditioning"] + 0.10 < b["raw_conditioning"] - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: RAG raw conditioning is >0.10 below base")
    if r["raw_structure"] + 0.05 < b["raw_structure"] - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: RAG raw structure materially below base")
    if r["final_ground_truth"] + 0.02 < b["final_ground_truth"] - eps:
        raise SystemExit("V3.1 VALIDATION GATE FAIL: RAG final validation GT consistency materially below base")
    print("V3.1 VALIDATION GATE: PASS")


if __name__ == "__main__":
    main()
