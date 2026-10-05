"""Create the single authoritative machine-readable and human-readable result bundle."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from src.config import config
from src.reproducibility import code_fingerprint, sha256_file


def _load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _checkpoint_info(path: str, produced_by: str):
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.is_file(),
        "sha256": sha256_file(p) if p.is_file() else None,
        "produced_by": produced_by,
    }


def main():
    manifest_meta = _load(config["data"]["metadata_path"])
    slots = _load("results/slot_test_metrics.json")
    seg = _load("results/segmentation_metrics.json")
    reports = _load("results/report_metrics.json")
    run_meta = {
        "slot": _load("results/slot_run_metadata.json"),
        "segmentation": _load("results/segmentation_run_metadata.json"),
        "vlm": _load("results/vlm_run_metadata.json"),
        "rag": _load(str(Path(config["rag"]["index_path"]) / "index_metadata.json")),
    }
    device_info = {
        "selected_device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3) if torch.cuda.is_available() else None,
    }
    checkpoints = {
        "slot_cnn": _checkpoint_info("models/slot_model_best.pth", "train_slot_model.py"),
        "segmentation_unet": _checkpoint_info("models/segmentation_unet_best.pth", "train_segmentation_model.py"),
        "hybrid_vlm_trainable": _checkpoint_info("models/hybrid3dvlm_trainable.pth", "train_vlm.py"),
    }
    provenance = {
        "code_fingerprint_sha256": code_fingerprint(Path.cwd()),
        "manifest_sha256": sha256_file(config["data"]["manifest_path"]) if Path(config["data"]["manifest_path"]).exists() else None,
        "slot_metadata_sha256": sha256_file(config["data"]["metadata_path"]) if Path(config["data"]["metadata_path"]).exists() else None,
        "pip_freeze_path": "results/pip_freeze.txt",
        "pip_freeze_sha256": sha256_file("results/pip_freeze.txt") if Path("results/pip_freeze.txt").exists() else None,
        "random_seed": int(config["project"].get("seed", 42)),
        "method_names": {
            "template_gt_oracle": "Ground-truth-slot constrained template (oracle upper bound)",
            "cnn_template": "CNN-predicted-slot constrained template",
            "vlm_base": "Hybrid structured-and-visual VLM",
            "vlm_rag": "Hybrid structured-and-visual VLM + train-only RAG",
        },
        "interpretation_note": "Rule-based unsupported-content flags are conservative automated diagnostics, not a clinically validated hallucination metric.",
    }
    bundle = {
        "provenance": provenance,
        "execution_environment": device_info,
        "dataset_and_split": manifest_meta,
        "configuration": {
            "slot_model": config["slot_model"],
            "segmentation": config["segmentation"],
            "vlm": config["vlm"],
            "rag": config["rag"],
            "data": {k: config["data"].get(k) for k in ("split_seed", "train_ratio", "val_ratio", "test_ratio", "target_shape", "intensity_clip", "extent_quantiles")},
        },
        "checkpoints": checkpoints,
        "run_metadata": run_meta,
        "slot_prediction": slots,
        "majority_baseline": slots.get("majority_baseline") if slots else None,
        "segmentation": seg,
        "vlm_metrics": reports.get("vlm_base") if reports else None,
        "vlm_rag_metrics": reports.get("vlm_rag") if reports else None,
        "report_generation": reports,
    }
    Path("results/thesis_results_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    lines = [
        "FINAL THESIS RESULTS - GENERATED FROM CURRENT CODE",
        "=" * 60,
        f"Code fingerprint SHA-256: {provenance['code_fingerprint_sha256']}",
        f"Random seed: {provenance['random_seed']}",
        f"Device: {device_info['selected_device']} | GPU: {device_info['gpu_name']} | VRAM GiB: {device_info['gpu_vram_gib']}",
        "",
    ]
    if manifest_meta:
        lines += [
            f"Complete paired cases: {manifest_meta['complete_cases']}",
            f"Split counts: {manifest_meta['split_counts']}",
            f"Extent thresholds (training-only tertiles, original voxels): {manifest_meta['extent_thresholds_original_voxels']}",
            "",
        ]
    if slots:
        lines += [
            f"Slot CNN micro accuracy: {slots['cnn']['micro_accuracy_all_slots']:.4f}",
            f"Majority baseline micro accuracy: {slots['majority_baseline']['micro_accuracy_all_slots']:.4f}",
            f"CNN - majority: {slots['cnn_minus_majority_micro_accuracy']:.4f}",
            f"CNN macro balanced accuracy (informative slots): {slots['cnn']['macro_balanced_accuracy_informative_slots']}",
            "",
        ]
    if seg:
        lines.append("Segmentation mean held-out test metrics (HD95 in mm):")
        for region, m in seg["summary"].items():
            lines.append(f"  {region}: Dice={m['dice']}, IoU={m['iou']}, Precision={m['precision']}, Recall={m['recall']}, HD95_mm={m['hd95']}")
        lines.append("")
    if reports:
        lines.append("Report-generation metrics:")
        for method, m in reports.items():
            lines.append(
                f"  {method}: raw_structure={m['raw_generation_two_section_structure_rate']:.4f}, final_pipeline_structure={m['final_pipeline_two_section_structure_rate']:.4f}, "
                f"raw_conditioning_consistency={m.get('raw_conditioning_slot_consistency_rate', m['conditioning_slot_consistency_rate']):.4f}, "
                f"final_conditioning_consistency={m['conditioning_slot_consistency_rate']:.4f}, "
                f"gt_consistency={m['ground_truth_slot_consistency_rate']:.4f}, "
                f"raw_unsupported_flag_rate={m.get('raw_rule_based_unsupported_content_flag_rate', m['rule_based_unsupported_content_flag_rate']):.4f}, "
                f"final_unsupported_flag_rate={m['rule_based_unsupported_content_flag_rate']:.4f}"
            )
    lines += ["", "Do not describe the unsupported-content diagnostic as a clinical hallucination rate or as clinical validation/safety evidence."]
    Path("results/thesis_results_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
