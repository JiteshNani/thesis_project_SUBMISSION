"""Loading helpers for the compact trainable VLM checkpoint."""
from __future__ import annotations

from pathlib import Path
import torch

from src.checks import require_file
from src.vlm_model import Hybrid3DVLM


def load_vlm_model(config: dict, device: torch.device, checkpoint: str | Path = "models/hybrid3dvlm_trainable.pth") -> Hybrid3DVLM:
    path = require_file(checkpoint, produced_by="python train_vlm.py")
    model = Hybrid3DVLM(config).to(device)
    payload = torch.load(path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected VLM checkpoint keys: {unexpected}")
    # Missing keys are expected because the frozen base LLM is downloaded from HF
    # and only LoRA + visual trainable state is stored locally.
    print(f"Loaded compact VLM checkpoint; frozen/base keys restored from model hub, missing-local={len(missing)}")
    model.eval()
    return model
