"""Evaluate genuine U-Net predictions on the held-out test set in original physical space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checks import require_file
from src.config import config
from src.data_utils import (
    BraTSSegmentationDataset,
    load_canonical_segmentation,
    resize_label_array,
    training_classes_to_segmentation,
)
from src.reproducibility import set_global_seed
from src.segmentation_eval import aggregate_case_metrics, evaluate_segmentation_arrays
from src.segmentation_model import build_segmentation_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-predictions", type=int, default=5, help="Number of original-grid prediction NIfTIs to save.")
    p.add_argument("--max-cases", type=int, default=None, help="Smoke-test limiter; omit for final thesis evaluation.")
    args = p.parse_args()
    set_global_seed(int(config["project"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BraTSSegmentationDataset(config["data"]["manifest_path"], split="test", config=config)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = build_segmentation_model(config).to(device)
    path = require_file(Path(config["paths"]["models_dir"]) / "segmentation_unet_best.pth", produced_by="python train_segmentation_model.py")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"] if "model_state" in payload else payload)
    model.eval()
    out_dir = Path("outputs/segmentation_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case = []
    use_amp = bool(config["segmentation"].get("use_amp", True)) and device.type == "cuda"
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if args.max_cases is not None and idx >= args.max_cases:
                break
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["flair"].to(device))
            pred_cls = torch.argmax(logits, dim=1)[0].cpu()
            pred_128 = training_classes_to_segmentation(pred_cls).numpy().astype(np.int16)

            seg_path = str(batch["seg_path"][0])
            gt_original, canonical_affine, spacing_mm = load_canonical_segmentation(seg_path)
            pred_original = resize_label_array(pred_128, gt_original.shape)
            metrics = evaluate_segmentation_arrays(pred_original, gt_original, spacing=spacing_mm)
            case_id = str(batch["case_id"][0])
            per_case.append({
                "case_id": case_id,
                "original_shape": list(gt_original.shape),
                "spacing_mm": list(spacing_mm),
                "metrics": metrics,
            })
            if idx < int(args.save_predictions):
                nib.save(nib.Nifti1Image(pred_original.astype(np.int16), canonical_affine), out_dir / f"{case_id}_pred_seg_canonical.nii.gz")

    summary = aggregate_case_metrics([x["metrics"] for x in per_case])
    result = {
        "model": "MONAI 3D U-Net",
        "training_input_space": "FLAIR resized to 128x128x128",
        "evaluation_space": "prediction nearest-neighbour mapped to canonical original BraTS grid; metrics against original held-out mask",
        "hd95_definition": "MONAI symmetric 95th-percentile Hausdorff distance using original canonical NIfTI physical voxel spacing; units=mm",
        "n_test_cases": len(per_case),
        "summary": summary,
        "per_case": per_case,
    }
    Path("results/segmentation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"n_test_cases": len(per_case), "summary": summary}, indent=2))
    print("Saved results/segmentation_metrics.json")


if __name__ == "__main__":
    main()
