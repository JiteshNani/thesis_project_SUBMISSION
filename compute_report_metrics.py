"""Compute report metrics for the single authoritative test-output file."""
from __future__ import annotations

import json
from pathlib import Path
import argparse
import numpy as np

from src.report_metrics import report_slot_consistency, structure_compliance, unsupported_content_flags, prompt_leakage_flags, truncation_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Fail if standard thesis NLP metrics cannot be computed.")
    args = ap.parse_args()
    data = json.loads(Path("results/all_model_test_outputs.json").read_text(encoding="utf-8"))
    methods = ["template_gt_oracle", "cnn_template", "vlm_base", "vlm_rag"]
    out = {}

    # Load standard NLP metrics once. These may download metric/model resources on first use.
    try:
        import evaluate
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        meteor_metric = evaluate.load("meteor")
        bert_metric = evaluate.load("bertscore")
        standard_available = True
    except Exception as exc:
        if args.strict:
            raise RuntimeError(f"Standard thesis NLP metric loaders unavailable: {exc}") from exc
        print("WARNING: standard NLP metric loaders unavailable:", exc)
        standard_available = False

    references = [x["reference"] for x in data]
    for method in methods:
        preds = [x[method] for x in data]
        raw_preds = [x.get("raw_generations", {}).get(method, x[method]) for x in data]
        raw_structured = [structure_compliance(p) for p in raw_preds]
        final_structured = [structure_compliance(p) for p in preds]
        conditioning_slots = [
            x["ground_truth_slots"] if method == "template_gt_oracle" else x["predicted_slots"]
            for x in data
        ]
        raw_conditioning_consistency = [
            report_slot_consistency(raw, slots)["mean"]
            for raw, slots in zip(raw_preds, conditioning_slots)
        ]
        raw_ground_truth_consistency = [
            report_slot_consistency(raw, x["ground_truth_slots"])["mean"]
            for raw, x in zip(raw_preds, data)
        ]
        conditioning_consistency = [
            report_slot_consistency(pred, slots)["mean"]
            for pred, slots in zip(preds, conditioning_slots)
        ]
        ground_truth_consistency = [report_slot_consistency(x[method], x["ground_truth_slots"])["mean"] for x in data]
        raw_flag_lists = [unsupported_content_flags(p) for p in raw_preds]
        raw_flags = [len(x) > 0 for x in raw_flag_lists]
        flag_lists = [unsupported_content_flags(p) for p in preds]
        flags = [len(x) > 0 for x in flag_lists]
        raw_prompt_leakage = [bool(prompt_leakage_flags(p)) for p in raw_preds]
        final_prompt_leakage = [bool(prompt_leakage_flags(p)) for p in preds]
        raw_truncation = [bool(truncation_flag(p)) for p in raw_preds]
        final_truncation = [bool(truncation_flag(p)) for p in preds]
        all_categories = sorted({category for row in flag_lists for category in row})
        category_rates = {
            category: float(np.mean([category in row for row in flag_lists]))
            for category in all_categories
        }
        result = {
            "n_cases": len(preds),
            "raw_generation_two_section_structure_rate": float(np.mean(raw_structured)),
            "final_pipeline_two_section_structure_rate": float(np.mean(final_structured)),
            "structure_metric_note": "Raw-generation metrics measure the language-model text before slot-grounding/structural repair; final-pipeline metrics measure the delivered report after the explicit deployment-available slot guardrail.",
            "raw_conditioning_slot_consistency_rate": float(np.mean(raw_conditioning_consistency)),
            "raw_ground_truth_slot_consistency_rate": float(np.mean(raw_ground_truth_consistency)),
            "conditioning_slot_consistency_rate": float(np.mean(conditioning_consistency)),
            "ground_truth_slot_consistency_rate": float(np.mean(ground_truth_consistency)),
            "raw_rule_based_unsupported_content_flag_rate": float(np.mean(raw_flags)),
            "rule_based_unsupported_content_flag_rate": float(np.mean(flags)),
            "raw_prompt_leakage_rate": float(np.mean(raw_prompt_leakage)),
            "final_prompt_leakage_rate": float(np.mean(final_prompt_leakage)),
            "raw_truncation_or_incomplete_rate": float(np.mean(raw_truncation)),
            "final_truncation_or_incomplete_rate": float(np.mean(final_truncation)),
            "unsupported_content_flag_category_rates": category_rates,
            "unsupported_content_flagged_case_count": int(sum(flags)),
            "unsupported_content_diagnostic_note": "Conservative rule-based asserted unsupported-content proxy; not a clinically validated hallucination metric.",
        }
        if standard_available:
            try:
                result["rouge"] = {k: float(v) for k, v in rouge_metric.compute(predictions=preds, references=references).items()}
                result["bleu"] = float(bleu_metric.compute(predictions=preds, references=[[r] for r in references])["bleu"])
                result["meteor"] = float(meteor_metric.compute(predictions=preds, references=references)["meteor"])
                bs = bert_metric.compute(predictions=preds, references=references, lang="en")
                result["bertscore_f1"] = float(np.mean(bs["f1"]))
            except Exception as exc:
                if args.strict:
                    raise RuntimeError(f"Standard NLP metrics failed for {method}: {exc}") from exc
                result["standard_metric_error"] = str(exc)
        out[method] = result
    Path("results/report_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("Saved results/report_metrics.json")


if __name__ == "__main__":
    main()
