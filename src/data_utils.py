"""BraTS2020 + TextBraTS loading, preprocessing, reproducible splits and slot targets.

Important leakage rule:
- Ground-truth segmentation masks are used to CREATE SUPERVISION TARGETS on train/val/test.
- Learned systems never receive a test segmentation mask as an inference input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.constants import SLOT_KEYS
from src.checks import assert_split_disjointness
from src.laterality import dominant_hemisphere_from_world_x


def _canonical_data(path: str | Path, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=dtype), img.affine.copy()


def _resize_tensor(array: np.ndarray, shape: Sequence[int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(array).float()[None, None]
    if mode == "nearest":
        out = F.interpolate(tensor, size=tuple(shape), mode="nearest")
    else:
        out = F.interpolate(tensor, size=tuple(shape), mode="trilinear", align_corners=False)
    return out[0, 0].cpu().numpy()


def preprocess_flair(path: str | Path, target_shape: Sequence[int] = (128, 128, 128), clip=(-3.0, 3.0)) -> torch.Tensor:
    data, _ = _canonical_data(path, np.float32)
    mask = data != 0
    if np.any(mask):
        values = data[mask]
        data = (data - float(values.mean())) / (float(values.std()) + 1e-8)
        data[~mask] = 0.0
    data = np.clip(data, float(clip[0]), float(clip[1]))
    data = _resize_tensor(data.astype(np.float32), target_shape, "trilinear")
    return torch.from_numpy(data).unsqueeze(0).float()


def preprocess_segmentation(path: str | Path, target_shape: Sequence[int] = (128, 128, 128)) -> torch.Tensor:
    data, _ = _canonical_data(path, np.float32)
    data = _resize_tensor(data, target_shape, "nearest").astype(np.int64)
    # Preserve official BraTS labels 0/1/2/4.
    return torch.from_numpy(data).long()


def load_canonical_segmentation(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load the original BraTS mask in canonical RAS+ orientation.

    Returns the integer label array, canonical affine, and physical voxel spacing
    in millimetres. This is used for final held-out segmentation metrics so HD95
    is evaluated in the original physical image grid rather than an approximated
    resized-grid spacing.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.int16)
    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(img.affine))
    return data, img.affine.copy(), spacing


def resize_label_array(array: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Nearest-neighbour resize for discrete segmentation labels."""
    return _resize_tensor(np.asarray(array, dtype=np.float32), target_shape, "nearest").astype(np.int16)


def segmentation_to_training_classes(seg: torch.Tensor) -> torch.Tensor:
    """Map BraTS labels 0/1/2/4 to consecutive CE classes 0/1/2/3."""
    out = torch.zeros_like(seg, dtype=torch.long)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out


def training_classes_to_segmentation(classes: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(classes, dtype=torch.long)
    out[classes == 1] = 1
    out[classes == 2] = 2
    out[classes == 3] = 4
    return out


def _world_x_coordinates(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return RAS world-x coordinates for N x 3 voxel indices."""
    if indices.size == 0:
        return np.empty((0,), dtype=np.float64)
    indices = np.asarray(indices, dtype=np.float64)
    return (indices @ np.asarray(affine, dtype=np.float64)[:3, :3].T + np.asarray(affine, dtype=np.float64)[:3, 3])[:, 0]




def _brain_midline_world_x(flair_path: str | Path | None, fallback_shape: Sequence[int], affine: np.ndarray) -> tuple[float, str]:
    """Define a reproducible left/right midline in RAS world coordinates.

    Preferred definition: midpoint of the left-most and right-most non-zero brain
    voxels in the canonicalised, skull-stripped FLAIR volume. BraTS volumes are
    co-registered and background is zero, so this is a transparent approximation
    of the midsagittal plane without assuming ``shape[0] / 2`` is anatomical x=0.

    If FLAIR is unavailable, fall back to the geometric world-x midpoint of the
    canonical segmentation grid. The caller records which definition was used.
    """
    if flair_path is not None and Path(flair_path).exists():
        flair, flair_affine = _canonical_data(flair_path, np.float32)
        brain_idx = np.argwhere(np.isfinite(flair) & (flair != 0))
        if len(brain_idx):
            x = _world_x_coordinates(brain_idx, flair_affine)
            return float((np.min(x) + np.max(x)) / 2.0), "canonical FLAIR nonzero-brain world-x bounds midpoint"

    # Geometric fallback in physical RAS coordinates.
    corners = np.array([
        [0, 0, 0],
        [max(int(fallback_shape[0]) - 1, 0), 0, 0],
    ], dtype=np.float64)
    x = _world_x_coordinates(corners, affine)
    return float((np.min(x) + np.max(x)) / 2.0), "canonical segmentation-grid world-x midpoint fallback"


def derive_slots_from_segmentation(
    seg_path: str | Path,
    extent_thresholds: tuple[float, float] | None = None,
    flair_path: str | Path | None = None,
) -> Dict[str, int | float | str]:
    """Derive structured supervision targets from a canonical BraTS mask.

    Laterality is computed in **RAS world coordinates**, not by splitting an array
    at ``shape[0] / 2``. A case-specific midline is defined from the non-zero
    skull-stripped FLAIR brain bounds when available. This is still an operational
    dominant-hemisphere target rather than a radiologist-validated anatomical label,
    and ``audit_laterality_targets.py`` must be reviewed before a final thesis run.
    """
    seg, affine = _canonical_data(seg_path, np.float32)
    tumor = seg > 0
    tumor_idx = np.argwhere(tumor)
    midline_x, midline_method = _brain_midline_world_x(flair_path, seg.shape, affine)
    tumor_x = _world_x_coordinates(tumor_idx, affine)
    laterality = dominant_hemisphere_from_world_x(tumor_x, midline_x)
    left_volume = int(laterality["left_tumor_voxels"])
    right_volume = int(laterality["right_tumor_voxels"])
    hemisphere = int(laterality["hemisphere"])

    total_volume = int(tumor.sum())
    if extent_thresholds is None:
        # Only used while creating metadata. Final datasets use train-derived tertiles.
        extent = 0
    else:
        low, high = extent_thresholds
        extent = 0 if total_volume <= low else 1 if total_volume <= high else 2

    return {
        "edema": int(np.any(seg == 2)),
        "enhancing": int(np.any(seg == 4)),
        "necrotic": int(np.any(seg == 1)),
        "hemisphere": hemisphere,
        "extent": extent,
        "tumor_voxels_original": total_volume,
        "midline_world_x_mm": float(midline_x),
        "midline_method": midline_method,
        "left_tumor_voxels": left_volume,
        "right_tumor_voxels": right_volume,
        "midline_tumor_voxels": int(laterality["midline_tumor_voxels"]),
        "laterality_ambiguous": bool(laterality["laterality_ambiguous"]),
        "tumor_centroid_world_x_mm": float(laterality["tumor_centroid_world_x_mm"]),
    }


def _find_report(raw_root: Path, case_id: str, suffix: str) -> Optional[Path]:
    candidates = [
        raw_root / case_id / f"{case_id}{suffix}",
        raw_root / "TextBraTSData" / case_id / f"{case_id}{suffix}",
        raw_root / "TextBraTSData" / f"{case_id}{suffix}",
        raw_root / f"{case_id}{suffix}",
    ]
    return next((p for p in candidates if p.exists()), None)


def create_manifest(
    raw_path: str = "data/raw",
    output_path: str = "data/manifest/manifest.csv",
    skipped_path: str = "data/manifest/skipped_incomplete_cases.csv",
    metadata_path: str = "data/manifest/slot_metadata.json",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    report_suffix: str = "_flair_text.txt",
) -> pd.DataFrame:
    """Create a portable manifest with relative paths and a fixed split.

    Exact counts follow from the number of complete paired cases available locally.
    If one BraTS case is missing a mask/report, it is recorded rather than silently
    pretending the dataset contains a fixed number of complete pairs.
    """
    raw_root = Path(raw_path).resolve()
    possible_roots = [
        raw_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
        raw_root / "MICCAI_BraTS2020_TrainingData",
        raw_root,
    ]
    mri_root = next((p for p in possible_roots if p.exists() and any(p.glob("BraTS20_Training_*"))), None)
    if mri_root is None:
        raise FileNotFoundError(
            "Could not find BraTS20_Training_* folders. Put the BraTS2020 and TextBraTS data under data/raw "
            "or edit data.raw_dir in config.yaml."
        )

    rows, skipped = [], []
    for case_dir in sorted(mri_root.glob("BraTS20_Training_*")):
        case_id = case_dir.name
        flair = next(case_dir.glob(f"{case_id}_flair.nii*"), None)
        seg = next(case_dir.glob(f"{case_id}_seg.nii*"), None)
        report = _find_report(raw_root, case_id, report_suffix)
        missing = []
        if flair is None:
            missing.append("flair")
        if seg is None:
            missing.append("seg")
        if report is None:
            missing.append("report")
        if missing:
            skipped.append({"case_id": case_id, "missing": "+".join(missing)})
            continue
        rows.append({
            "case_id": case_id,
            "flair_path": str(Path(flair).resolve().relative_to(Path.cwd().resolve())) if Path.cwd().resolve() in Path(flair).resolve().parents else str(Path(flair).resolve()),
            "seg_path": str(Path(seg).resolve().relative_to(Path.cwd().resolve())) if Path.cwd().resolve() in Path(seg).resolve().parents else str(Path(seg).resolve()),
            "report_path": str(Path(report).resolve().relative_to(Path.cwd().resolve())) if Path.cwd().resolve() in Path(report).resolve().parents else str(Path(report).resolve()),
        })

    if not rows:
        raise RuntimeError("No complete BraTS + TextBraTS pairs were found.")

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    if n_train + n_val >= len(df):
        n_val = max(1, len(df) - n_train - 1)
    df["split"] = "test"
    df.loc[indices[:n_train], "split"] = "train"
    df.loc[indices[n_train:n_train + n_val], "split"] = "val"
    assert_split_disjointness(df)

    # Extent categories are defined ONLY from training-set tumour volumes to avoid test leakage
    # and to make small/moderate/large informative rather than all falling into one class.
    train_volumes = []
    for _, row in df[df["split"] == "train"].iterrows():
        train_volumes.append(derive_slots_from_segmentation(row["seg_path"], flair_path=row["flair_path"])["tumor_voxels_original"])
    low, high = np.quantile(np.asarray(train_volumes, dtype=np.float64), [1/3, 2/3]).tolist()

    slot_distributions = {}
    thresholds = (float(low), float(high))
    for split_name in ("train", "val", "test"):
        counts = {k: {} for k in SLOT_KEYS}
        for _, row in df[df["split"] == split_name].iterrows():
            slots = derive_slots_from_segmentation(row["seg_path"], thresholds, flair_path=row["flair_path"])
            for key in SLOT_KEYS:
                value = str(int(slots[key]))
                counts[key][value] = counts[key].get(value, 0) + 1
        slot_distributions[split_name] = counts

    metadata = {
        "split_seed": seed,
        "complete_cases": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "extent_thresholds_original_voxels": [float(low), float(high)],
        "extent_definition": "training-set tertiles of whole-tumour voxel count",
        "hemisphere_definition": "dominant tumour volume in canonical RAS world-x coordinates; case-specific midline from nonzero FLAIR brain bounds; midline-plane voxels reported separately; exact side ties flagged ambiguous",
        "slot_distributions": slot_distributions,
        "skipped_cases": int(len(skipped)),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    pd.DataFrame(skipped, columns=["case_id", "missing"]).to_csv(skipped_path, index=False)
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df


def load_extent_thresholds(metadata_path: str = "data/manifest/slot_metadata.json") -> tuple[float, float]:
    data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    low, high = data["extent_thresholds_original_voxels"]
    return float(low), float(high)


class BraTSSlotDataset(Dataset):
    """FLAIR/report dataset with optional mask-derived supervision labels.

    VLM training sets ``include_slots=False`` so mask-derived slots are not loaded;
    the frozen slot CNN supplies its conditioning inputs instead.
    """

    def __init__(self, manifest_path: str, *, split: str, config: Dict, include_slots: bool = True):
        self.config = config
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.split = split
        self.include_slots = include_slots
        self.target_shape = tuple(config["data"].get("target_shape", [128, 128, 128]))
        self.clip = tuple(config["data"].get("intensity_clip", [-3.0, 3.0]))
        self.extent_thresholds = load_extent_thresholds(config["data"]["metadata_path"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        flair = preprocess_flair(row["flair_path"], self.target_shape, self.clip)
        report_text = Path(row["report_path"]).read_text(encoding="utf-8", errors="replace").strip()
        result = {
            "flair": flair,
            "report_text": report_text,
            "case_id": str(row["case_id"]),
            "flair_path": str(row["flair_path"]),
        }
        if self.include_slots:
            slots = derive_slots_from_segmentation(row["seg_path"], self.extent_thresholds, flair_path=row["flair_path"])
            result["slots"] = {k: int(slots[k]) for k in SLOT_KEYS}
        return result


class BraTSSegmentationDataset(Dataset):
    def __init__(self, manifest_path: str, *, split: str, config: Dict):
        self.config = config
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.target_shape = tuple(config["data"].get("target_shape", [128, 128, 128]))
        self.clip = tuple(config["data"].get("intensity_clip", [-3.0, 3.0]))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        flair = preprocess_flair(row["flair_path"], self.target_shape, self.clip)
        seg_brats = preprocess_segmentation(row["seg_path"], self.target_shape)
        seg_class = segmentation_to_training_classes(seg_brats)
        return {
            "flair": flair,
            "seg": seg_class.unsqueeze(0),
            "seg_brats": seg_brats,
            "seg_path": str(row["seg_path"]),
            "case_id": str(row["case_id"]),
        }
