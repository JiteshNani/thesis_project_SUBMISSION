"""Validation-only V3.2 RAG ablation.

Compares on a deterministic half of the VALIDATION split:
1) frozen V3.1 Base VLM, no retrieval;
2) V3.2 RAG-adapted VLM, retrieval disabled (fine-tune ablation);
3) V3.2 RAG-adapted VLM + top-2 sanitised-exemplar retrieval.

No held-out TEST case is loaded.  Development and confirmation halves are
fixed by a seed-42 hash ordering before scores are inspected.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import config as base_config
from src.constants import SLOT_KEYS
from src.data_utils import BraTSSlotDataset
from src.rag_retriever import RAGRetriever
from src.report_metrics import (
    generation_diagnostics,
    report_slot_consistency,
    structure_compliance,
    truncation_flag,
    unsupported_content_flags,
    prompt_leakage_flags,
)
from src.reproducibility import set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_model import Hybrid3DVLM

BASE_CHECKPOINT = Path("models/hybrid3dvlm_trainable.pth")
V32_CHECKPOINT = Path("models/hybrid3dvlm_rag_v32.pth")
V32_INDEX = "data/rag_index_v32"


def v32_config() -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["rag"].update({
        "top_k": 2,
        "index_path": V32_INDEX,
        "query_mode": "structured_slots",
        "context_mode": "sanitized_exemplars",
        "exemplar_chars_per_case": 520,
        "max_context_chars": 1200,
        "min_exemplar_slot_matches": 5,
        "preserve_safe_generation": True,
        "preserve_min_conditioning": 1.0,
        "candidate_min_raw_conditioning": 1.0,
        "max_raw_conditioning_drop_vs_base": 0.0,
    })
    cfg["vlm"]["temperature"] = 0.0
    cfg["vlm"]["do_sample"] = False
    return cfg


def _load_model(cfg: dict, checkpoint: Path, device: torch.device) -> Hybrid3DVLM:
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    model = Hybrid3DVLM(cfg).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {checkpoint}: {unexpected}")
    model.eval()
    return model


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


def _subset_membership(case_ids: List[str], seed: int = 42):
    ordered = sorted(
        case_ids,
        key=lambda x: hashlib.sha256(f"{seed}:{x}".encode("utf-8")).hexdigest(),
    )
    cut = len(ordered) // 2
    return set(ordered[:cut]), set(ordered[cut:])


def _report_diag(text: str, pred_slots: Dict, gt_slots: Dict) -> Dict[str, float]:
    return {
        "conditioning": float(report_slot_consistency(text, pred_slots)["mean"]),
        "ground_truth": float(report_slot_consistency(text, gt_slots)["mean"]),
        "structure": float(structure_compliance(text)),
        "unsupported": float(bool(unsupported_content_flags(text))),
        "prompt_leakage": float(bool(prompt_leakage_flags(text))),
        "truncation": float(truncation_flag(text)),
    }


def _mean_diag(records: List[dict], key: str) -> Dict[str, float]:
    names = ["conditioning", "ground_truth", "structure", "unsupported", "prompt_leakage", "truncation"]
    return {n: float(np.mean([r[key + "_diag"][n] for r in records])) for n in names}


def _lexical_metrics(preds: List[str], refs: List[str]) -> tuple[dict, dict]:
    """Return aggregate and per-case lexical metrics; BERTScore is best-effort."""
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import single_meteor_score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_case = {"rouge1": [], "rouge2": [], "rougeL": [], "meteor": []}
    for p, r in zip(preds, refs):
        rs = scorer.score(r, p)
        per_case["rouge1"].append(float(rs["rouge1"].fmeasure))
        per_case["rouge2"].append(float(rs["rouge2"].fmeasure))
        per_case["rougeL"].append(float(rs["rougeL"].fmeasure))
        try:
            per_case["meteor"].append(float(single_meteor_score(r.split(), p.split())))
        except Exception:
            per_case["meteor"].append(float("nan"))

    # Use the same evaluate/BERTScore route already present in the thesis code.
    try:
        import evaluate
        bert_metric = evaluate.load("bertscore")
        bs = bert_metric.compute(predictions=preds, references=refs, lang="en")
        per_case["bertscore_f1"] = [float(x) for x in bs["f1"]]
    except Exception as exc:
        per_case["bertscore_f1"] = []
        per_case["bertscore_error"] = str(exc)

    agg = {}
    for key, vals in per_case.items():
        if not isinstance(vals, list) or not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            agg[key] = float(np.mean(arr))
    return agg, per_case


def _bootstrap_delta(base: List[float], rag: List[float], *, seed: int = 42, n_boot: int = 5000) -> dict | None:
    a = np.asarray(base, dtype=float)
    b = np.asarray(rag, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return None
    d = b - a
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(int(n_boot), len(d)))
    means = d[draws].mean(axis=1)
    return {
        "mean_delta": float(d.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(d)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["development", "confirmation"], required=True)
    args = ap.parse_args()

    cfg = v32_config()
    seed = int(cfg["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not BASE_CHECKPOINT.exists() or not V32_CHECKPOINT.exists():
        raise FileNotFoundError("Need both frozen V3.1 Base checkpoint and models/hybrid3dvlm_rag_v32.pth")

    ds = BraTSSlotDataset(cfg["data"]["manifest_path"], split="val", config=cfg, include_slots=True)
    # This first pass only determines a fixed case-ID partition.  It never uses
    # test data and the partition is independent of model scores.
    all_ids = [str(ds[i]["case_id"]) for i in range(len(ds))]
    dev_ids, confirm_ids = _subset_membership(all_ids, seed=seed)
    selected_ids = dev_ids if args.subset == "development" else confirm_ids
    print(f"V3.2 subset={args.subset} n={len(selected_ids)} | test split used: False")

    slot_model = load_slot_model(cfg, device)
    slot_model.eval()

    # ------------------------------------------------------------------
    # Phase A: frozen Base VLM only.  The 6-GB target GPU should never hold
    # two TinyLlama copies at the same time.
    # ------------------------------------------------------------------
    base_model = _load_model(cfg, BASE_CHECKPOINT, device)
    records_by_id = {}
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for idx, batch in enumerate(loader):
        case_id = str(batch["case_id"][0])
        if case_id not in selected_ids:
            continue
        mri = batch["flair"].to(device)
        pred = predict_slots(slot_model, mri)[0]
        gt = _plain_batch_slots(batch["slots"])
        ref = str(batch["report_text"][0])
        torch.manual_seed(seed + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + idx)
        base = base_model.generate_report_detailed(mri, pred, retriever=None, case_id=case_id)
        records_by_id[case_id] = {
            "case_id": case_id,
            "reference": ref,
            "predicted_slots": pred,
            "ground_truth_slots": gt,
            "base": base["final_report"],
            "base_raw": base["raw_generation"],
            "base_diag": _report_diag(base["final_report"], pred, gt),
        }
        print(f"[base {len(records_by_id)}/{len(selected_ids)}] {case_id}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase B: one V3.2 model copy, evaluated once without retrieval and once
    # with retrieval.  This isolates fine-tune-only versus retrieval effects.
    # ------------------------------------------------------------------
    rag_model = _load_model(cfg, V32_CHECKPOINT, device)
    rag = RAGRetriever(cfg)
    rag._load()
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    done = 0
    for idx, batch in enumerate(loader):
        case_id = str(batch["case_id"][0])
        if case_id not in selected_ids:
            continue
        mri = batch["flair"].to(device)
        pred = predict_slots(slot_model, mri)[0]
        rec = records_by_id[case_id]
        if pred != rec["predicted_slots"]:
            raise AssertionError(f"Slot prediction changed between ablation phases for {case_id}")
        gt = rec["ground_truth_slots"]

        torch.manual_seed(seed + 50000 + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 50000 + idx)
        adapted = rag_model.generate_report_detailed(mri, pred, retriever=None, case_id=case_id)

        torch.manual_seed(seed + 100000 + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100000 + idx)
        rag_out = rag_model.generate_report_detailed(mri, pred, retriever=rag, case_id=case_id)

        rec.update({
            "adapted_no_rag": adapted["final_report"],
            "adapted_no_rag_raw": adapted["raw_generation"],
            "adapted_no_rag_diag": _report_diag(adapted["final_report"], pred, gt),
            "rag": rag_out["final_report"],
            "rag_raw": rag_out["raw_generation"],
            "rag_diag": _report_diag(rag_out["final_report"], pred, gt),
            "rag_context": rag_out.get("rag_context", ""),
            "rag_selection": rag_out.get("selection"),
            "base_candidate_diagnostics": rag_out.get("base_candidate_diagnostics"),
            "rag_candidate_diagnostics": rag_out.get("rag_candidate_diagnostics"),
        })
        done += 1
        print(f"[rag {done}/{len(selected_ids)}] {case_id} selection={rec['rag_selection']}")

    records = [records_by_id[k] for k in sorted(records_by_id)]
    refs = [r["reference"] for r in records]
    base_preds = [r["base"] for r in records]
    adapted_preds = [r["adapted_no_rag"] for r in records]
    rag_preds = [r["rag"] for r in records]

    base_lex, base_pc = _lexical_metrics(base_preds, refs)
    adapted_lex, adapted_pc = _lexical_metrics(adapted_preds, refs)
    rag_lex, rag_pc = _lexical_metrics(rag_preds, refs)

    deltas = {}
    for metric in ("rouge1", "rouge2", "rougeL", "meteor", "bertscore_f1"):
        if metric in base_pc and metric in rag_pc and isinstance(base_pc[metric], list) and isinstance(rag_pc[metric], list):
            if len(base_pc[metric]) == len(rag_pc[metric]) and len(base_pc[metric]) > 0:
                deltas[metric] = _bootstrap_delta(base_pc[metric], rag_pc[metric], seed=seed)

    selections = [r["rag_selection"] for r in records]
    output_changed = [r["rag"] != r["base"] for r in records]
    summary = {
        "experiment": "V3.2 RAG enhanced validation-only ablation",
        "subset": args.subset,
        "n_cases": len(records),
        "test_split_used": False,
        "design": {
            "retrieval_count": 2,
            "retrieval_query": "CNN-predicted structured clinical slots",
            "retrieval_source": "training split only",
            "retrieved_context": "sanitised report exemplars; raw reports never inserted",
            "base_checkpoint": str(BASE_CHECKPOINT),
            "rag_checkpoint": str(V32_CHECKPOINT),
            "comparison": ["frozen_base", "rag_adapted_no_retrieval", "rag_adapted_plus_retrieval"],
        },
        "base": {"diagnostics": _mean_diag(records, "base"), "lexical": base_lex},
        "rag_adapted_no_retrieval": {"diagnostics": _mean_diag(records, "adapted_no_rag"), "lexical": adapted_lex},
        "rag_enhanced": {
            "diagnostics": _mean_diag(records, "rag"),
            "lexical": rag_lex,
            "candidate_acceptance_rate": float(sum(x == "rag_candidate" for x in selections) / max(1, len(selections))),
            "fallback_rate": float(sum(x == "base_fallback" for x in selections) / max(1, len(selections))),
            "output_changed_vs_base_rate": float(np.mean(output_changed)) if output_changed else 0.0,
        },
        "rag_minus_base_paired_bootstrap": deltas,
        "interpretation_note": (
            "This is a validation-only RAG ablation. It must not replace the already frozen authoritative V3.1 held-out-test results. "
            "RAG is considered useful only if language/semantic metrics improve while structured grounding and safety diagnostics are preserved."
        ),
    }

    # Pre-declared development gate. Confirmation is reported once without tuning.
    b, r = summary["base"], summary["rag_enhanced"]
    safety_ok = (
        r["diagnostics"]["conditioning"] >= 0.999
        and r["diagnostics"]["structure"] >= 0.999
        and r["diagnostics"]["unsupported"] <= 1e-9
        and r["diagnostics"]["prompt_leakage"] <= 1e-9
        and r["diagnostics"]["truncation"] <= 1e-9
    )
    improvements = []
    for metric in ("rougeL", "meteor", "bertscore_f1"):
        if metric in b["lexical"] and metric in r["lexical"]:
            improvements.append((metric, r["lexical"][metric] - b["lexical"][metric]))
    positive_count = sum(delta > 0 for _, delta in improvements)
    benefit_ok = positive_count >= 2 if len(improvements) >= 2 else positive_count >= 1
    changed_ok = summary["rag_enhanced"]["output_changed_vs_base_rate"] > 0.0
    summary["predeclared_gate"] = {
        "safety_preserved": bool(safety_ok),
        "positive_language_metrics": [{"metric": m, "delta": float(d)} for m, d in improvements],
        "benefit_rule_passed": bool(benefit_ok),
        "rag_changes_output": bool(changed_ok),
        "passed": bool(safety_ok and benefit_ok and changed_ok),
    }

    Path("results").mkdir(exist_ok=True)
    stem = f"v32_rag_{args.subset}"
    Path(f"results/{stem}_outputs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(f"results/{stem}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved results/{stem}_metrics.json")
    if summary["predeclared_gate"]["passed"]:
        print(f"V3.2 RAG {args.subset.upper()} GATE: PASS")
    else:
        print(f"V3.2 RAG {args.subset.upper()} GATE: NO DEMONSTRATED BENEFIT")
        if args.subset == "development":
            raise SystemExit(3)


if __name__ == "__main__":
    main()
