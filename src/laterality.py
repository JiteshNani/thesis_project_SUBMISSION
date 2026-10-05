"""Pure world-coordinate laterality helpers (no NIfTI dependency)."""
from __future__ import annotations

from typing import Dict
import numpy as np


def dominant_hemisphere_from_world_x(tumor_x: np.ndarray, midline_x: float, atol_mm: float = 1e-6) -> Dict[str, int | float | bool]:
    """Classify dominant laterality from physical RAS world-x coordinates.

    Voxels strictly left/right of the case-specific midsagittal reference are
    counted on their corresponding side. Voxels lying on the reference plane
    (within ``atol_mm``) are reported separately and do not inflate either side.
    The binary target remains compatible with the thesis design: the larger side
    wins. Exact side-count ties are marked ambiguous and resolved deterministically
    by the tumour world-x centroid; a perfectly centred centroid uses a documented
    left fallback solely to keep the binary label defined.
    """
    x = np.asarray(tumor_x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {
            "hemisphere": 0,
            "left_tumor_voxels": 0,
            "right_tumor_voxels": 0,
            "midline_tumor_voxels": 0,
            "laterality_ambiguous": True,
            "tumor_centroid_world_x_mm": float(midline_x),
        }
    left = int(np.sum(x < (midline_x - atol_mm)))
    right = int(np.sum(x > (midline_x + atol_mm)))
    mid = int(x.size - left - right)
    ambiguous = left == right
    centroid = float(np.mean(x))
    if left > right:
        hemisphere = 0
    elif right > left:
        hemisphere = 1
    elif centroid < midline_x - atol_mm:
        hemisphere = 0
    elif centroid > midline_x + atol_mm:
        hemisphere = 1
    else:
        hemisphere = 0
    return {
        "hemisphere": int(hemisphere),
        "left_tumor_voxels": left,
        "right_tumor_voxels": right,
        "midline_tumor_voxels": mid,
        "laterality_ambiguous": bool(ambiguous),
        "tumor_centroid_world_x_mm": centroid,
    }
