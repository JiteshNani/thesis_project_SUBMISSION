"""Load the trained slot model and perform leakage-free inference."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import torch

from src.constants import SLOT_KEYS
from src.model import MultiTask3DCNN


def build_slot_model(config: Dict, device: torch.device) -> MultiTask3DCNN:
    cfg = config["slot_model"]
    return MultiTask3DCNN(int(cfg["backbone_channels"]), float(cfg.get("dropout", 0.2))).to(device)


def load_slot_model(config: Dict, device: torch.device, checkpoint: str = "models/slot_model_best.pth") -> MultiTask3DCNN:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Slot checkpoint not found: {path}. Run train_slot_model.py first.")
    model = build_slot_model(config, device)
    payload = torch.load(path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_slots(model: MultiTask3DCNN, flair_batch: torch.Tensor) -> List[Dict[str, int]]:
    outputs = model(flair_batch)
    batch_size = flair_batch.shape[0]
    return [
        {key: int(torch.argmax(outputs[key][i]).item()) for key in SLOT_KEYS}
        for i in range(batch_size)
    ]
