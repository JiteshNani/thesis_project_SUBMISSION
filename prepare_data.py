"""Build the authoritative 70/15/15 manifest and train-derived slot metadata."""
from __future__ import annotations
import json
from pathlib import Path
from src.config import config
from src.data_utils import create_manifest
from src.reproducibility import set_global_seed


def main():
    seed = int(config["project"].get("seed", 42))
    set_global_seed(seed)
    d = config["data"]
    df = create_manifest(
        raw_path=d["raw_dir"],
        output_path=d["manifest_path"],
        skipped_path=d["skipped_path"],
        metadata_path=d["metadata_path"],
        seed=int(d.get("split_seed", seed)),
        train_ratio=float(d.get("train_ratio", 0.70)),
        val_ratio=float(d.get("val_ratio", 0.15)),
        report_suffix=d.get("report_suffix", "_flair_text.txt"),
    )
    print(f"Complete paired cases: {len(df)}")
    print("Split counts:", df["split"].value_counts().to_dict())
    print(Path(d["metadata_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
