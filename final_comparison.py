"""Assemble the four-method held-out comparison without rerunning the VLM."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.checks import require_files
from src.config import config
from src.constants import SLOT_KEYS
from src.data_utils import derive_slots_from_segmentation, load_extent_thresholds
from src.report_metrics import report_slot_consistency, structure_compliance, unsupported_content_flags
from src.template_baseline import GroundedTemplateBaseline


def _load_records(path: str, key: str):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(r["case_id"]): r for r in records if key in r}


def main():
    require_files([
        (config["data"]["manifest_path"], "python prepare_data.py"),
        ("results/slot_test_metrics.json", "python evaluate_slot_model.py"),
        ("results/vlm_base_test_outputs.json", "python evaluate_vlm.py"),
        ("results/vlm_rag_test_outputs.json", "python evaluate_vlm_rag.py"),
    ])
    manifest = pd.read_csv(config["data"]["manifest_path"])
    test = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    thresholds = load_extent_thresholds(config["data"]["metadata_path"])
    slot_eval = json.loads(Path("results/slot_test_metrics.json").read_text(encoding="utf-8"))
    pred_by_case = {str(r["case_id"]): r for r in slot_eval["predictions"]}
    base = _load_records("results/vlm_base_test_outputs.json", "vlm_base")
    rag = _load_records("results/vlm_rag_test_outputs.json", "vlm_rag")
    expected = set(test["case_id"].astype(str))
    for name, mapping in (("slot predictions", pred_by_case), ("base VLM", base), ("VLM+RAG", rag)):
        if set(mapping) != expected:
            raise RuntimeError(f"{name} case IDs do not exactly match the held-out test split; missing={sorted(expected-set(mapping))[:10]}, extra={sorted(set(mapping)-expected)[:10]}")

    template = GroundedTemplateBaseline()
    records = []
    for _, row in test.iterrows():
        cid = str(row["case_id"])
        gt_info = derive_slots_from_segmentation(row["seg_path"], thresholds, flair_path=row["flair_path"])
        gt = {k: int(gt_info[k]) for k in SLOT_KEYS}
        pred = {k: int(pred_by_case[cid]["prediction"][k]) for k in SLOT_KEYS}
        base_pred = {k: int(base[cid]["predicted_slots"][k]) for k in SLOT_KEYS}
        rag_pred = {k: int(rag[cid]["predicted_slots"][k]) for k in SLOT_KEYS}
        if base_pred != pred or rag_pred != pred:
            raise RuntimeError(f"CNN slot predictions drifted between evaluation stages for {cid}: slot_eval={pred}, base={base_pred}, rag={rag_pred}")
        methods = {
            "template_gt_oracle": template.generate_report(gt)["full_report"],
            "cnn_template": template.generate_report(pred)["full_report"],
            "vlm_base": base[cid]["vlm_base"],
            "vlm_rag": rag[cid]["vlm_rag"],
        }
        raw_generations = {
            "template_gt_oracle": methods["template_gt_oracle"],
            "cnn_template": methods["cnn_template"],
            "vlm_base": base[cid].get("vlm_base_raw", methods["vlm_base"]),
            "vlm_rag": rag[cid].get("vlm_rag_raw", methods["vlm_rag"]),
        }
        diagnostics = {}
        for name, text in methods.items():
            conditioning = gt if name == "template_gt_oracle" else pred
            diagnostics[name] = {
                "raw_structure_compliant": structure_compliance(raw_generations[name]),
                "final_pipeline_structure_compliant": structure_compliance(text),
                "consistency_with_conditioning_slots": report_slot_consistency(text, conditioning),
                "consistency_with_ground_truth_slots": report_slot_consistency(text, gt),
                "unsupported_content_flags": unsupported_content_flags(text),
            }
        reference = Path(row["report_path"]).read_text(encoding="utf-8", errors="replace")
        records.append({
            "case_id": cid,
            "reference": reference,
            "ground_truth_slots": gt,
            "predicted_slots": pred,
            **methods,
            "raw_generations": raw_generations,
            "diagnostics": diagnostics,
        })
    Path("results/all_model_test_outputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results/all_model_test_outputs.json for {len(records)} held-out cases")


if __name__ == "__main__":
    main()
