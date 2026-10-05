"""Preflight checks before a long thesis run.

Use ``--final`` for the authoritative experiment. Final mode is intentionally
strict: missing dependencies, path problems, invalid split provenance, or a weak
laterality sanity audit stop the run before expensive training begins.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
import pandas as pd
import torch
from src.config import config
from src.checks import assert_split_disjointness

REQUIRED_MODULES = [
    "numpy", "pandas", "sklearn", "scipy", "yaml", "nibabel", "monai",
    "transformers", "peft", "accelerate", "sentence_transformers", "faiss",
    "evaluate", "bert_score", "nltk", "rouge_score",
]


def _dependency_check() -> list[str]:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    return missing


def _split_check(df: pd.DataFrame) -> list[str]:
    errors = []
    expected = {"train", "val", "test"}
    actual = set(df["split"].astype(str).unique()) if "split" in df.columns else set()
    if actual != expected:
        errors.append(f"split values are {sorted(actual)}, expected exactly {sorted(expected)}")
    try:
        assert_split_disjointness(df)
    except ValueError as exc:
        errors.append(str(exc))
    if len(df) < 3:
        errors.append("fewer than 3 complete cases")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="Enable strict final-thesis checks.")
    args = ap.parse_args()

    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA build:", torch.version.cuda)
        props = torch.cuda.get_device_properties(0)
        vram_gib = props.total_memory / (1024**3)
        print("GPU VRAM GiB:", round(vram_gib, 2))
        if vram_gib <= 6.5:
            print("VRAM NOTE: 6 GB-class GPU detected. VLM uses FP16, batch size 1, gradient checkpointing and shortened token limits; close other GPU-heavy applications before VLM training.")
    elif args.final:
        print("WARNING: final run has no CUDA GPU. The VLM/3D models may be impractically slow.")

    missing_deps = _dependency_check()
    if missing_deps:
        print("ERROR: missing/broken required dependencies:")
        for x in missing_deps:
            print("  ", x)
        return 10
    print("Dependencies: OK")

    # Fixed thesis architecture/configuration contracts.
    vcfg = config.get("vlm", {})
    expected_vlm = {
        "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_rank": 16,
        "lora_alpha": 32,
        "num_visual_tokens": 8,
    }
    for key, expected_value in expected_vlm.items():
        if vcfg.get(key) != expected_value:
            print(f"ERROR: VLM config {key}={vcfg.get(key)!r}; expected {expected_value!r}")
            return 12
    if list(vcfg.get("lora_target_modules", [])) != ["q_proj", "v_proj"]:
        print("ERROR: VLM LoRA target modules must be exactly q_proj and v_proj.")
        return 13
    if int(config.get("project", {}).get("seed", -1)) != 42 or int(config.get("data", {}).get("split_seed", -1)) != 42:
        print("ERROR: final project and split seed must both be 42.")
        return 14
    ratios = [float(config["data"].get(k, 0.0)) for k in ("train_ratio", "val_ratio", "test_ratio")]
    if any(abs(a-b) > 1e-9 for a,b in zip(ratios, [0.70, 0.15, 0.15])):
        print("ERROR: data split ratios must be 70/15/15.")
        return 15
    print("Fixed methodology configuration: OK")

    manifest = Path(config["data"]["manifest_path"])
    if not manifest.exists():
        print("Manifest: MISSING (run python prepare_data.py)")
        return 1
    df = pd.read_csv(manifest)
    print("Manifest cases:", len(df), df["split"].value_counts().to_dict())

    split_errors = _split_check(df)
    if split_errors:
        print("ERROR: manifest split validation failed:")
        for e in split_errors:
            print("  ", e)
        return 3

    missing = []
    for col in ("flair_path", "seg_path", "report_path"):
        if col not in df.columns:
            print(f"ERROR: missing manifest column {col}")
            return 4
        for p in df[col]:
            if not Path(p).exists():
                missing.append(str(p))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        print("ERROR: manifest contains missing paths. Rebuild it on this laptop with python prepare_data.py")
        for p in missing:
            print("  ", p)
        return 2
    print("Manifest paths: OK")
    non_flair = [str(p) for p in df["flair_path"] if "_flair.nii" not in Path(str(p)).name.lower()]
    if non_flair:
        print("ERROR: FLAIR-only contract violated; non-FLAIR input path(s) found:", non_flair[:10])
        return 16
    print("FLAIR-only model-input contract: OK")

    meta_path = Path(config["data"]["metadata_path"])
    if not meta_path.exists():
        print("ERROR: slot metadata is missing")
        return 5
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("Hemisphere definition:", meta.get("hemisphere_definition"))
    print("Extent definition:", meta.get("extent_definition"))
    print("Slot distributions:", json.dumps(meta.get("slot_distributions", {}), indent=2))

    if "world-x" not in str(meta.get("hemisphere_definition", "")):
        print("ERROR: final laterality target is not documented as world-coordinate based.")
        return 6
    if "training-set tertiles" not in str(meta.get("extent_definition", "")):
        print("ERROR: extent thresholds are not train-derived.")
        return 7

    if args.final:
        audit_path = Path("results/laterality_audit.json")
        if not audit_path.exists():
            print("ERROR: laterality audit missing. Run python audit_laterality_targets.py before final validation.")
            return 8
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        agreement = audit.get("agreement_rate")
        n_audit = int(audit.get("n_nonambiguous_audit_cases", audit.get("n_explicit_unilateral_reports", 0)))
        min_agreement = float(config.get("validation", {}).get("laterality_audit_min_agreement", 0.70))
        min_cases = int(config.get("validation", {}).get("laterality_audit_min_cases", 20))
        print("Laterality report-wording audit:", agreement, "cases:", n_audit, "minimum agreement:", min_agreement, "minimum cases:", min_cases)
        if n_audit < min_cases:
            print("ERROR: too few explicit unilateral reports for the configured laterality sanity audit.")
            return 11
        if agreement is not None and float(agreement) < min_agreement:
            print("ERROR: laterality audit is below the configured minimum. Inspect orientation/report conventions before training.")
            return 9

    print("Preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
