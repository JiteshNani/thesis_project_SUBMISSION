from pathlib import Path
import pytest
nib = pytest.importorskip("nibabel")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.data_utils import derive_slots_from_segmentation, segmentation_to_training_classes, training_classes_to_segmentation
from src.segmentation_eval import evaluate_segmentation_arrays


def test_brats_class_round_trip():
    seg = torch.tensor([[[0, 1, 2, 4]]])
    cls = segmentation_to_training_classes(seg)
    restored = training_classes_to_segmentation(cls)
    assert torch.equal(seg, restored)


def test_canonical_axis0_laterality(tmp_path: Path):
    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[1:4, 5:10, 5:10] = 2
    p = tmp_path / "left.nii.gz"
    nib.save(nib.Nifti1Image(seg, np.eye(4)), p)
    assert derive_slots_from_segmentation(p, (10, 10000))["hemisphere"] == 0
    seg2 = np.zeros_like(seg); seg2[12:15, 5:10, 5:10] = 2
    p2 = tmp_path / "right.nii.gz"; nib.save(nib.Nifti1Image(seg2, np.eye(4)), p2)
    assert derive_slots_from_segmentation(p2, (10, 10000))["hemisphere"] == 1


def test_perfect_segmentation_metrics():
    gt = np.zeros((10, 10, 10), dtype=np.int16)
    gt[2:5, 2:5, 2:5] = 1
    gt[5:7, 2:5, 2:5] = 2
    gt[3:4, 3:4, 3:4] = 4
    m = evaluate_segmentation_arrays(gt, gt)
    for region in m.values():
        assert abs(region["dice"] - 1.0) < 1e-8
        assert abs(region["iou"] - 1.0) < 1e-8
        assert abs(region["hd95"] - 0.0) < 1e-8


def test_world_coordinate_laterality_uses_flair_brain_midline(tmp_path: Path):
    # Brain occupies x=2..11 -> world-x midline = 6.5 mm. Tumour at x=8 is
    # therefore right-dominant. A naive shape[0]//2 split (10) would call it left.
    flair = np.zeros((20, 12, 12), dtype=np.float32)
    flair[2:12, 2:10, 2:10] = 1.0
    seg = np.zeros((20, 12, 12), dtype=np.int16)
    seg[8:9, 4:8, 4:8] = 2
    fp = tmp_path / "flair.nii.gz"
    sp = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(flair, np.eye(4)), fp)
    nib.save(nib.Nifti1Image(seg, np.eye(4)), sp)
    info = derive_slots_from_segmentation(sp, (1, 1000), flair_path=fp)
    assert abs(info["midline_world_x_mm"] - 6.5) < 1e-6
    assert info["hemisphere"] == 1
