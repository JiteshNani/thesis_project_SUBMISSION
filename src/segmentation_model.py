"""Real trainable 3D segmentation network for BraTS labels."""
from __future__ import annotations
from typing import Dict
from monai.networks.nets import UNet


def build_segmentation_model(config: Dict):
    cfg = config["segmentation"]
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,  # background + labels 1,2,4 mapped to 1,2,3 during training
        channels=tuple(int(x) for x in cfg["channels"]),
        strides=tuple(int(x) for x in cfg["strides"]),
        num_res_units=int(cfg.get("num_res_units", 1)),
        norm="INSTANCE",
    )
