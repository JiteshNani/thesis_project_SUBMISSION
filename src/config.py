"""Central project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("models_dir", "outputs_dir", "results_dir", "figures_dir", "experiments_dir"):
        value = cfg.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
    return cfg


config = load_config()
