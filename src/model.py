"""Multi-task 3D CNN for five structured clinical slots."""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class MultiTask3DCNN(nn.Module):
    """Compact 3D CNN that preserves coarse spatial layout for laterality."""

    def __init__(self, backbone_channels: int = 16, dropout: float = 0.2):
        super().__init__()
        c = backbone_channels
        self.backbone = nn.Sequential(
            nn.Conv3d(1, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c), nn.ReLU(inplace=True),
            nn.Conv3d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c * 2), nn.ReLU(inplace=True),
            nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        feat = c * 4 * 8
        self.shared = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(feat, c * 8), nn.ReLU(inplace=True))
        self.heads = nn.ModuleDict({
            "edema": nn.Linear(c * 8, 2),
            "enhancing": nn.Linear(c * 8, 2),
            "necrotic": nn.Linear(c * 8, 2),
            "hemisphere": nn.Linear(c * 8, 2),
            "extent": nn.Linear(c * 8, 3),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.shared(self.backbone(x))
        return {name: head(f) for name, head in self.heads.items()}
