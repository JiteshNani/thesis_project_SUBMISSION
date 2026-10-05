"""Final integrity gate for the authoritative held-out test results.

This script intentionally fails rather than producing a thesis result bundle if
case coverage, split provenance, or method outputs are inconsistent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import config


def load_json(path: str):
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Missing required result file: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def assert_case_ids(name: str, got, expected):
    got_set = {str(x) for x in got}
    expected_set = {str(x) for x in expected}
    if got_set != expected_set:
        raise RuntimeError(
            f"{name} case-ID mismatch: missing={sorted(expected_set - got_set)[:10]}, "
            f"extra={sorted(got_set - expected_set)[:10]}"
        )


def main() -> None:
    manifest = pd.read_csv(config["data"]["manifest_path"])
    train_ids = manifest.loc[manifest["split"] == "train", "case_id"].astype(str).tolist()
    test_ids = manifest.loc[manifest["split"] == "test", "case_id"].astype(str).tolist()

    slots = load_json("results/slot_test_metrics.json")
    seg = load_json("results/segmentation_metrics.json")
    base = load_json("results/vlm_base_test_outputs.json")
    rag = load_json("results/vlm_rag_test_outputs.json")
    all_outputs = load_json("results/all_model_test_outputs.json")
    report_metrics = load_json("results/report_metrics.json")
    rag_meta = load_json(str(Path(config["rag"]["index_path"]) / "index_metadata.json"))

    if int(slots["cnn"]["n_cases"]) != len(test_ids):
        raise RuntimeError("Slot CNN evaluation case count does not equal held-out test count.")
    if int(seg["n_test_cases"]) != len(test_ids):
        raise RuntimeError("Segmentation evaluation case count does not equal held-out test count.")

    assert_case_ids("slot predictions", [r["case_id"] for r in slots["predictions"]], test_ids)
    assert_case_ids("base VLM", [r["case_id"] for r in base], test_ids)
    assert_case_ids("VLM+RAG", [r["case_id"] for r in rag], test_ids)
    assert_case_ids("combined comparison", [r["case_id"] for r in all_outputs], test_ids)

    expected_methods = {"template_gt_oracle", "cnn_template", "vlm_base", "vlm_rag"}
    if set(report_metrics) != expected_methods:
        raise RuntimeError(f"Report metrics methods are {sorted(report_metrics)}, expected {sorted(expected_methods)}")
    for method, values in report_metrics.items():
        n_cases = int(values.get("n_cases", -1))
        if n_cases != len(test_ids):
            raise RuntimeError(f"Report metric count for {method} is {n_cases}; expected {len(test_ids)}")
        for metric_name in (
            "raw_generation_two_section_structure_rate",
            "final_pipeline_two_section_structure_rate",
            "conditioning_slot_consistency_rate",
            "ground_truth_slot_consistency_rate",
            "rule_based_unsupported_content_flag_rate",
        ):
            value = values.get(metric_name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise RuntimeError(f"{method}.{metric_name} is outside [0,1]: {value}")

    indexed_ids = [str(x) for x in rag_meta.get("indexed_case_ids", [])]
    assert_case_ids("RAG index", indexed_ids, train_ids)
    if rag_meta.get("indexed_split") != "train":
        raise RuntimeError(f"RAG index is not marked as train-only: {rag_meta.get('indexed_split')}")
    if int(rag_meta.get("top_k", -1)) != int(config["rag"]["top_k"]):
        raise RuntimeError("RAG metadata top-k does not match configuration.")

    if len(all_outputs) != len(test_ids):
        raise RuntimeError("Combined model output count is incomplete.")
    for row in all_outputs:
        for method in expected_methods:
            if method not in row or not str(row[method]).strip():
                raise RuntimeError(f"Missing generated report for method={method}, case={row.get('case_id')}")

    print("FINAL RESULT INTEGRITY: PASS")
    print(f"Held-out test cases: {len(test_ids)}")
    print("Verified methods: template_gt_oracle, cnn_template, vlm_base, vlm_rag")
    print(f"Train-only RAG cases: {len(indexed_ids)}")


if __name__ == "__main__":
    main()
