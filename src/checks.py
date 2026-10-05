"""Small dependency and provenance checks shared by pipeline entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_file(path: str | Path, *, produced_by: str | None = None) -> Path:
    p = Path(path)
    if not p.is_file():
        hint = f" Run `{produced_by}` first." if produced_by else ""
        raise FileNotFoundError(f"Required file is missing: {p}.{hint}")
    return p


def require_files(items: Iterable[tuple[str | Path, str | None]]) -> list[Path]:
    return [require_file(path, produced_by=producer) for path, producer in items]


def assert_split_disjointness(df: pd.DataFrame) -> None:
    """Raise when a case identifier leaks across train/validation/test splits."""
    required = {"case_id", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if df["case_id"].isna().any():
        raise ValueError("Manifest contains missing case_id values.")
    duplicates = df[df["case_id"].astype(str).duplicated(keep=False)][["case_id", "split"]]
    if not duplicates.empty:
        raise ValueError("Dataset leakage/duplication detected: case_id occurs more than once: " + repr(duplicates.to_dict(orient="records")))
    sets = {name: set(df.loc[df["split"] == name, "case_id"].astype(str)) for name in ("train", "val", "test")}
    overlaps = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"Dataset split leakage detected: {bad}")


def assert_rag_train_only(manifest_df: pd.DataFrame, entries: list[dict]) -> None:
    """Raise if any RAG index entry is outside the training split."""
    if "case_id" not in manifest_df.columns or "split" not in manifest_df.columns:
        raise ValueError("Manifest must contain case_id and split columns for RAG leakage validation.")
    train_ids = set(manifest_df.loc[manifest_df["split"] == "train", "case_id"].astype(str))
    nontrain_ids = set(manifest_df.loc[manifest_df["split"].isin(["val", "test"]), "case_id"].astype(str))
    indexed_ids = [str(item.get("case_id")) for item in entries]
    duplicates = sorted({x for x in indexed_ids if indexed_ids.count(x) > 1})
    invalid = sorted(set(indexed_ids) - train_ids)
    overlap = sorted(set(indexed_ids) & nontrain_ids)
    if duplicates or invalid or overlap:
        raise ValueError(
            "RAG leakage/integrity failure: "
            f"duplicates={duplicates[:10]}, non_train={invalid[:10]}, val_or_test={overlap[:10]}"
        )
    if set(indexed_ids) != train_ids:
        missing = sorted(train_ids - set(indexed_ids))
        raise ValueError(f"RAG index does not contain exactly the training cases; missing={missing[:10]}")
