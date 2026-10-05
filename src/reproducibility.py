"""Reproducibility helpers used by every training/evaluation entry point."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by CUDA for deterministic matrix-multiplication paths where supported.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some third-party kernels do not provide deterministic variants; warn_only
        # preserves runnability while making any exception visible in the log.
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def code_fingerprint(root: str | Path = ".") -> str:
    """Hash the executable thesis source/config files in deterministic path order."""
    root = Path(root).resolve()
    candidates = []
    for pattern in ("*.py", "*.yaml", "*.yml", "requirements.txt"):
        candidates.extend(root.rglob(pattern))
    excluded_parts = {".venv", "__pycache__", ".pytest_cache", "data", "models", "outputs", "results", "experiments"}
    files = [p for p in candidates if not any(part in excluded_parts for part in p.relative_to(root).parts)]
    h = hashlib.sha256()
    for path in sorted(set(files), key=lambda x: x.as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel + b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()

def save_run_metadata(output_path: str | Path, *, config: Dict[str, Any], extra: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "code_fingerprint_sha256": code_fingerprint(Path.cwd()),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": config,
    }
    manifest_path = Path(str(config.get("data", {}).get("manifest_path", "")))
    metadata_path = Path(str(config.get("data", {}).get("metadata_path", "")))
    if manifest_path.is_file():
        payload["manifest_sha256"] = sha256_file(manifest_path)
    if metadata_path.is_file():
        payload["slot_metadata_sha256"] = sha256_file(metadata_path)
    if extra:
        payload.update(extra)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
