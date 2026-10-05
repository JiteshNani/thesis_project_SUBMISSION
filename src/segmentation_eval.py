"""BraTS region metrics computed from genuine model predictions.

HD95 is calculated with MONAI's standard Hausdorff implementation in physical
space. Empty-mask edge cases are handled explicitly so thesis tables do not hide
undefined distances.
"""
from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import torch

REGIONS = {
    "whole_tumor": lambda s: s > 0,
    "tumor_core": lambda s: (s == 1) | (s == 4),
    "enhancing_tumor": lambda s: s == 4,
    "edema": lambda s: s == 2,
    "necrotic_non_enhancing": lambda s: s == 1,
}


def _monai_hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    try:
        from monai.metrics import compute_hausdorff_distance
    except Exception as exc:  # pragma: no cover - dependency preflight catches this
        raise RuntimeError("MONAI is required for thesis HD95 evaluation. Install requirements.txt.") from exc

    p = torch.from_numpy(pred.astype(np.float32))[None, None]
    g = torch.from_numpy(gt.astype(np.float32))[None, None]
    value = compute_hausdorff_distance(
        y_pred=p,
        y=g,
        include_background=True,
        distance_metric="euclidean",
        percentile=95.0,
        directed=False,
        spacing=tuple(float(x) for x in spacing),
    )
    return float(value[0, 0].item())


def _binary_metrics(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> Dict[str, float | None]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    denom = p.sum() + g.sum()
    dice = float(2 * tp / denom) if denom else 1.0
    union = np.logical_or(p, g).sum()
    iou = float(tp / union) if union else 1.0
    precision = float(tp / (tp + fp)) if tp + fp else (1.0 if not g.any() else 0.0)
    recall = float(tp / (tp + fn)) if tp + fn else 1.0

    if not p.any() and not g.any():
        hd = 0.0
        hd_status = "both_empty"
    elif not p.any() or not g.any():
        # There is no finite bidirectional surface distance when one mask is empty.
        hd = None
        hd_status = "undefined_one_empty"
    else:
        hd = _monai_hd95(p, g, spacing=spacing)
        hd_status = "monai_hd95"
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95": hd,
        "hd95_status": hd_status,
    }


def evaluate_segmentation_arrays(pred_seg: np.ndarray, gt_seg: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> Dict[str, Dict]:
    return {name: _binary_metrics(fn(pred_seg), fn(gt_seg), spacing) for name, fn in REGIONS.items()}


def aggregate_case_metrics(cases: Iterable[Dict[str, Dict]]) -> Dict[str, Dict[str, float | None]]:
    cases = list(cases)
    out = {}
    for region in REGIONS:
        out[region] = {}
        for metric in ("dice", "iou", "precision", "recall", "hd95"):
            vals = [c[region][metric] for c in cases if c[region][metric] is not None and np.isfinite(c[region][metric])]
            out[region][metric] = float(np.mean(vals)) if vals else None
            out[region][f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
            out[region][f"{metric}_n"] = len(vals)
        out[region]["hd95_undefined_cases"] = int(sum(c[region].get("hd95_status") == "undefined_one_empty" for c in cases))
    return out
