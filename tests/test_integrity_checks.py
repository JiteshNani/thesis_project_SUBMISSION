from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.checks import assert_rag_train_only, assert_split_disjointness, require_file
from src.laterality import dominant_hemisphere_from_world_x


def test_split_disjointness_accepts_unique_cases():
    df = pd.DataFrame({"case_id": ["a", "b", "c"], "split": ["train", "val", "test"]})
    assert_split_disjointness(df)


def test_split_disjointness_rejects_duplicate_case_leakage():
    df = pd.DataFrame({"case_id": ["a", "a", "c"], "split": ["train", "test", "val"]})
    with pytest.raises(ValueError, match="leakage|duplication"):
        assert_split_disjointness(df)


def test_rag_index_must_equal_training_ids_only():
    manifest = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "split": ["train", "train", "val", "test"]})
    assert_rag_train_only(manifest, [{"case_id": "a"}, {"case_id": "b"}])
    with pytest.raises(ValueError, match="RAG leakage"):
        assert_rag_train_only(manifest, [{"case_id": "a"}, {"case_id": "c"}])


def test_checkpoint_dependency_message(tmp_path: Path):
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="train_slot_model.py"):
        require_file(missing, produced_by="python train_slot_model.py")


def test_world_x_laterality_left_right_and_midline_ambiguous():
    left = dominant_hemisphere_from_world_x(np.array([-8.0, -7.0, -1.0, 2.0]), midline_x=0.0)
    assert left["hemisphere"] == 0 and not left["laterality_ambiguous"]

    right = dominant_hemisphere_from_world_x(np.array([-1.0, 2.0, 4.0, 8.0]), midline_x=0.0)
    assert right["hemisphere"] == 1 and not right["laterality_ambiguous"]

    midline = dominant_hemisphere_from_world_x(np.array([-2.0, 0.0, 2.0]), midline_x=0.0)
    assert midline["left_tumor_voxels"] == 1
    assert midline["right_tumor_voxels"] == 1
    assert midline["midline_tumor_voxels"] == 1
    assert midline["laterality_ambiguous"] is True
