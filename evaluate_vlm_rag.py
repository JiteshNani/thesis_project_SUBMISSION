"""Evaluate the hybrid VLM with a verified train-only RAG index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.checks import require_files
from src.config import config
from src.data_utils import BraTSSlotDataset
from src.rag_retriever import RAGRetriever
from src.reproducibility import set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_inference import load_vlm_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=None, help="Smoke-test limiter only.")
    args = ap.parse_args()
    rag_dir = Path(config["rag"]["index_path"])
    require_files([
        (config["data"]["manifest_path"], "python prepare_data.py"),
        ("models/slot_model_best.pth", "python train_slot_model.py"),
        ("models/hybrid3dvlm_trainable.pth", "python train_vlm.py"),
        (rag_dir / "faiss_index.bin", "python build_rag_index.py"),
        (rag_dir / "reports.pkl", "python build_rag_index.py"),
    ])
    seed = int(config["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BraTSSlotDataset(config["data"]["manifest_path"], split="test", config=config, include_slots=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    slot_model = load_slot_model(config, device)
    vlm = load_vlm_model(config, device)
    rag = RAGRetriever(config)
    # _load performs a hard train-only index assertion against the current manifest.
    rag._load()
    records = []
    for idx, batch in enumerate(loader):
        if args.max_cases is not None and idx >= args.max_cases:
            break
        case_id = str(batch["case_id"][0])
        mri = batch["flair"].to(device)
        pred = predict_slots(slot_model, mri)[0]
        torch.manual_seed(seed + 100000 + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100000 + idx)
        generated = vlm.generate_report_detailed(mri, pred, retriever=rag, case_id=case_id)
        records.append({
            "case_id": case_id,
            "reference": str(batch["report_text"][0]),
            "predicted_slots": pred,
            "vlm_rag_raw": generated["raw_generation"],
            "vlm_rag": generated["final_report"],
            "selection": generated.get("selection"),
            "base_candidate_raw": generated.get("base_candidate_raw"),
            "base_candidate_final": generated.get("base_candidate_final"),
            "rag_candidate_raw": generated.get("rag_candidate_raw"),
            "rag_candidate_final": generated.get("rag_candidate_final"),
            "base_candidate_conditioning_consistency": generated.get("base_candidate_conditioning_consistency"),
            "rag_candidate_conditioning_consistency": generated.get("rag_candidate_conditioning_consistency"),
            "base_candidate_unsupported_flag_count": generated.get("base_candidate_unsupported_flag_count"),
            "rag_candidate_unsupported_flag_count": generated.get("rag_candidate_unsupported_flag_count"),
            "rag_context": generated.get("rag_context"),
            "base_candidate_diagnostics": generated.get("base_candidate_diagnostics"),
            "rag_candidate_diagnostics": generated.get("rag_candidate_diagnostics"),
        })
        print(f"[{idx+1}/{len(ds)}] {case_id}")
    Path("results").mkdir(exist_ok=True)
    Path("results/vlm_rag_test_outputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved results/vlm_rag_test_outputs.json")


if __name__ == "__main__":
    main()
