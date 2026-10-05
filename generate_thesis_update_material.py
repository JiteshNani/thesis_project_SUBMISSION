"""Generate conservative chapter-update material from the authoritative results.

This script never invents numbers. It reads only the current machine-generated
result files and labels every diagnostic according to its actual scope.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results")


def load(path: str):
    p = ROOT / path
    if not p.is_file():
        raise FileNotFoundError(f"Missing authoritative result file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, digits=4):
    return "NA" if x is None else f"{float(x):.{digits}f}"


def main():
    bundle = load("thesis_results_bundle.json")
    slots = bundle["slot_prediction"]
    seg = bundle["segmentation"]
    reports = bundle["report_generation"]
    meta = bundle["dataset_and_split"]
    env = bundle["execution_environment"]
    vlm_meta = bundle["run_metadata"].get("vlm") or {}

    split_counts = meta.get("split_counts", {})
    methods = bundle["provenance"]["method_names"]

    chapter4 = f"""# Chapter 4 — Results update material

## Dataset and split
The authoritative run contained **{meta['complete_cases']} complete paired cases**, split deterministically using seed 42 into {split_counts.get('train')} training, {split_counts.get('val')} validation and {split_counts.get('test')} held-out test cases. The extent thresholds were calculated only from the training split: {meta['extent_thresholds_original_voxels']} original-mask voxels.

## Slot CNN
The held-out slot CNN micro-accuracy across all five clinical slots was **{fmt(slots['cnn']['micro_accuracy_all_slots'])}**. The training-split majority baseline was **{fmt(slots['majority_baseline']['micro_accuracy_all_slots'])}**, giving a CNN-minus-baseline difference of **{fmt(slots['cnn_minus_majority_micro_accuracy'])}**.

Per-slot results must be copied from `results/tables/slot_metrics.csv`; do not replace them with previous thesis values.

## Segmentation
Held-out segmentation metrics must be copied from `results/tables/segmentation_metrics.csv`. Dice, IoU, precision and recall are computed on the canonical original held-out BraTS grid. HD95 is reported in millimetres using the original NIfTI physical voxel spacing and MONAI's symmetric 95th-percentile Hausdorff implementation.

## Report generation
The comparison consists of four methods:
1. {methods['template_gt_oracle']}
2. {methods['cnn_template']}
3. {methods['vlm_base']}
4. {methods['vlm_rag']}

For report-generation results, use `results/tables/report_metrics.csv`. For VLM methods, distinguish **raw generator metrics** from **final slot-grounded pipeline metrics**. The final grounding layer uses only deployment-available CNN-predicted slots and is therefore a reliability guardrail, not evidence of clinical correctness. The unsupported-content diagnostic is a **conservative rule-based proxy**, not a clinical hallucination rate.

## VLM checkpoint selection
The VLM was selected by validation loss using the training/validation split only. Selected epoch: **{vlm_meta.get('best_epoch', 'NA')}**; best validation loss: **{vlm_meta.get('best_validation_loss', 'NA')}**.

## Reproducibility
Device: **{env.get('selected_device')}**; GPU: **{env.get('gpu_name')}**; CUDA build: **{env.get('cuda_build')}**.
"""

    report_order = ["template_gt_oracle", "cnn_template", "vlm_base", "vlm_rag"]
    chapter5 = """# Chapter 5 — Discussion update material

Use the actual values from the generated tables, not legacy values. The discussion should distinguish three different concepts:

- deterministic generation has no free-form stochastic text generation;
- CNN slot mistakes can still propagate into the deterministic report;
- VLM/RAG unsupported-content flags are automated diagnostics and are not clinical hallucination measurements;
- raw VLM output must be discussed separately from the final slot-grounded report delivered by the pipeline;
- the RAG candidate gate uses predicted slots only and must be described as preventing retrieval-induced degradation, not as proof that retrieval is clinically correct.

The key comparative interpretation should therefore be framed as a **reliability–expressiveness trade-off**, rather than as proof that the deterministic template is clinically safe.

The RAG branch is train-only retrieval with top-k=3. Retrieval is driven by CNN-predicted structured findings rather than a generated draft; retrieved reports are sanitised language/style context only and are explicitly not treated as evidence about the current patient.

Segmentation is an auxiliary spatial evaluation branch. Its predictions are **not passed into the VLM** in this experiment.
"""

    chapter6 = """# Chapter 6 — Conclusion update material

The final conclusion should state that the experiment provides computational evidence for the relative behaviour of structured templates, CNN-conditioned templates, a compact structured-and-visual VLM, and the same VLM with train-only RAG. It should not claim clinical validation, clinical safety, radiologist acceptance, or a clinically validated hallucination rate.

The strongest contribution is the empirical demonstration that increasing language-generation flexibility does not automatically improve imaging-grounded reliability under the tested FLAIR-only configuration.

The final numerical values should be taken only from `results/thesis_results_bundle.json` and the CSV tables generated in the same run.
"""

    (ROOT / "thesis_chapter4_update_material.md").write_text(chapter4, encoding="utf-8")
    (ROOT / "thesis_chapter5_update_material.md").write_text(chapter5, encoding="utf-8")
    (ROOT / "thesis_chapter6_update_material.md").write_text(chapter6, encoding="utf-8")
    print("Generated:")
    for name in ("thesis_chapter4_update_material.md", "thesis_chapter5_update_material.md", "thesis_chapter6_update_material.md"):
        print(" ", ROOT / name)


if __name__ == "__main__":
    main()
