"""Sanity-check mask-derived laterality against explicit unilateral TextBraTS wording.

Report text is used only as an audit. It never defines slot labels and never enters
training targets. Geometrically tied/ambiguous masks are reported separately and
excluded from the primary non-ambiguous agreement rate.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from src.config import config
from src.data_utils import derive_slots_from_segmentation, load_extent_thresholds


def explicit_side(text: str):
    low = text.lower()
    left = bool(re.search(r"\bleft\b", low))
    right = bool(re.search(r"\bright\b", low))
    if left == right:
        return None
    return 0 if left else 1


def main():
    df = pd.read_csv(config["data"]["manifest_path"])
    thresholds = load_extent_thresholds(config["data"]["metadata_path"])
    records = []
    for _, row in df.iterrows():
        text = Path(row["report_path"]).read_text(encoding="utf-8", errors="replace")
        report_side = explicit_side(text)
        if report_side is None:
            continue
        slot_info = derive_slots_from_segmentation(row["seg_path"], thresholds, flair_path=row["flair_path"])
        mask_side = int(slot_info["hemisphere"])
        ambiguous = bool(slot_info.get("laterality_ambiguous", False))
        records.append({
            "case_id": row["case_id"],
            "split": row["split"],
            "mask_side": mask_side,
            "report_side": int(report_side),
            "agree": bool(mask_side == report_side),
            "laterality_ambiguous": ambiguous,
            "left_tumor_voxels": int(slot_info["left_tumor_voxels"]),
            "right_tumor_voxels": int(slot_info["right_tumor_voxels"]),
            "midline_tumor_voxels": int(slot_info.get("midline_tumor_voxels", 0)),
            "midline_world_x_mm": slot_info["midline_world_x_mm"],
            "midline_method": slot_info["midline_method"],
        })
    nonambiguous = [r for r in records if not r["laterality_ambiguous"]]
    agreement = sum(r["agree"] for r in nonambiguous) / len(nonambiguous) if nonambiguous else None
    result = {
        "n_explicit_unilateral_reports": len(records),
        "n_nonambiguous_audit_cases": len(nonambiguous),
        "n_ambiguous_audit_cases": len(records) - len(nonambiguous),
        "agreement_rate": agreement,
        "agreement_definition": "agreement on explicit unilateral TextBraTS wording among non-ambiguous mask-derived laterality cases",
        "note": "TextBraTS wording is used only as a sanity audit, not as model supervision.",
        "records": records,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/laterality_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
