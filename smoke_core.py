"""Dependency-light smoke test that does not require the BraTS dataset or model downloads."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.checks import assert_split_disjointness
from src.laterality import dominant_hemisphere_from_world_x
from src.model import MultiTask3DCNN
from src.report_metrics import report_slot_consistency


def main():
    assert_split_disjointness(pd.DataFrame({"case_id": ["a", "b", "c"], "split": ["train", "val", "test"]}))
    lat = dominant_hemisphere_from_world_x(np.array([-3.0, 2.0, 4.0]), 0.0)
    assert lat["hemisphere"] == 1
    model = MultiTask3DCNN(backbone_channels=4, dropout=0.0).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 32, 32, 32))
    assert out["edema"].shape == (1, 2) and out["extent"].shape == (1, 3)
    slots = {"edema": 0, "enhancing": 0, "necrotic": 0, "hemisphere": 0, "extent": 0}
    report = "Findings:\nSmall left-hemisphere tumour. No oedema, no enhancement, and no necrotic core.\n\nImpression:\nSmall left-sided tumour."
    assert report_slot_consistency(report, slots)["mean"] == 1.0
    print("CORE SMOKE PASS")


if __name__ == "__main__":
    main()
