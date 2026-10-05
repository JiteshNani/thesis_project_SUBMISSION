"""GPU-friendly 3D augmentation for FLAIR segmentation training."""
from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def augment_flair_and_seg(
    flair: torch.Tensor,
    seg: Optional[torch.Tensor] = None,
    *,
    max_rotation_degrees: float = 10.0,
    intensity_scale: float = 0.10,
    probability: float = 0.5,
    flip_probability: float = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply paired random 3D rotation, flips and MRI intensity scaling.

    The same spatial transform is applied to MRI and segmentation. MRI uses
    trilinear interpolation; the discrete segmentation uses nearest neighbour.
    Flips are safe for segmentation because tumour classes have no left/right
    semantic label.
    """
    if flair.dim() != 5:
        raise ValueError(f"Expected flair [B,C,D,H,W], got {tuple(flair.shape)}")
    b = flair.shape[0]
    device, dtype = flair.device, flair.dtype
    theta = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
    apply = torch.rand(b, device=device) < probability
    max_rad = math.radians(max_rotation_degrees)

    for i in range(b):
        if not bool(apply[i]):
            continue
        ax, ay, az = ((torch.rand(3, device=device, dtype=dtype) * 2 - 1) * max_rad)
        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)
        one = torch.ones((), device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)
        rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, cx, -sx]), torch.stack([zero, sx, cx])])
        ry = torch.stack([torch.stack([cy, zero, sy]), torch.stack([zero, one, zero]), torch.stack([-sy, zero, cy])])
        rz = torch.stack([torch.stack([cz, -sz, zero]), torch.stack([sz, cz, zero]), torch.stack([zero, zero, one])])
        theta[i, :, :3] = rz @ ry @ rx

    grid = F.affine_grid(theta, flair.size(), align_corners=False)
    flair_aug = F.grid_sample(flair, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    seg_aug = None
    if seg is not None:
        seg_aug = F.grid_sample(seg.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False).long()

    scales = torch.ones(b, device=device, dtype=dtype)
    if intensity_scale > 0:
        random_scale = 1.0 + (torch.rand(b, device=device, dtype=dtype) * 2 - 1) * intensity_scale
        scales = torch.where(apply, random_scale, scales)
    flair_aug = flair_aug * scales.view(b, 1, 1, 1, 1)

    # Independent paired flips along spatial dimensions D/H/W.
    for dim in (2, 3, 4):
        if torch.rand((), device=device).item() < flip_probability:
            flair_aug = torch.flip(flair_aug, dims=(dim,))
            if seg_aug is not None:
                seg_aug = torch.flip(seg_aug, dims=(dim,))

    return flair_aug, seg_aug
