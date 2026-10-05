"""Train-only FAISS retrieval over TextBraTS reports.

V3.1 uses retrieval only for *style cues*. Raw training reports remain stored for
provenance but are never inserted into the VLM prompt. Retrieval is driven by
current CNN-predicted structured slots and excludes the current training case
when used during retrieval-aware fine-tuning.
"""
from __future__ import annotations

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data_utils import derive_slots_from_segmentation, load_extent_thresholds
from src.checks import assert_rag_train_only
from src.reproducibility import sha256_file
from src.report_targets import sanitize_reference_for_conditioning, sanitize_reference_as_exemplar, sanitize_reference_as_v321_snippet, slot_signature, structured_report_from_slots


class RAGRetriever:
    def __init__(self, config: Dict):
        self.config = config
        cfg = config["rag"]
        self.manifest_path = config["data"]["manifest_path"]
        self.metadata_path = config["data"]["metadata_path"]
        self.embedding_device = str(cfg.get("embedding_device", "cpu"))
        self.model = SentenceTransformer(cfg["embedding_model"], device=self.embedding_device)
        self.top_k = int(cfg.get("top_k", 3))
        self.add_slot_signature = bool(cfg.get("add_slot_signature", True))
        self.query_mode = str(cfg.get("query_mode", "structured_slots"))
        self.style_chars = int(cfg.get("context_chars_per_case", 120))
        self.context_mode = str(cfg.get("context_mode", "style_cues"))
        self.exemplar_chars = int(cfg.get("exemplar_chars_per_case", 520))
        self.visual_rerank = bool(cfg.get("visual_rerank", False))
        self.rerank_pool = int(cfg.get("rerank_pool", 48))
        self.index_path = Path(cfg.get("index_path", "data/rag_index"))
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.entries: Optional[List[Dict]] = None
        self.visual_vectors: Optional[Dict[str, np.ndarray]] = None

    def build_index(self) -> None:
        df = pd.read_csv(self.manifest_path)
        train = df[df["split"] == "train"].reset_index(drop=True)
        thresholds = load_extent_thresholds(self.metadata_path)
        entries, texts = [], []
        for _, row in train.iterrows():
            raw_report = Path(row["report_path"]).read_text(encoding="utf-8", errors="replace").strip()
            slots = derive_slots_from_segmentation(row["seg_path"], thresholds, flair_path=row["flair_path"])
            slots = {k: int(slots[k]) for k in ("edema", "enhancing", "necrotic", "hemisphere", "extent")}
            style_text = sanitize_reference_for_conditioning(raw_report, slots, max_chars=self.style_chars)
            exemplar_text = sanitize_reference_as_exemplar(raw_report, slots, max_chars=self.exemplar_chars)
            v321_snippet = sanitize_reference_as_v321_snippet(raw_report)

            # Retrieval representation emphasizes the structured signature. The
            # style cue has very low weight in the natural-language query and can
            # never inject patient-specific content into generation.
            embed_text = structured_report_from_slots(slots)
            if style_text:
                embed_text += "\nSTYLE CUES: " + style_text
            if self.add_slot_signature:
                embed_text += "\nSTRUCTURED FINDINGS: " + slot_signature(slots)

            entries.append({
                "case_id": str(row["case_id"]),
                "text": raw_report,  # provenance only; never sent to VLM prompt
                "style_text": style_text,
                "exemplar_text": exemplar_text,
                "v321_snippet": v321_snippet,
                "slots": slots,
            })
            texts.append(embed_text)

        assert_rag_train_only(df, entries)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.entries = entries
        faiss.write_index(self.index, str(self.index_path / "faiss_index.bin"))
        with (self.index_path / "reports.pkl").open("wb") as f:
            pickle.dump(entries, f)
        metadata = {
            "embedding_model": self.config["rag"]["embedding_model"],
            "embedding_device": self.embedding_device,
            "top_k": self.top_k,
            "query_mode": self.query_mode,
            "context_mode": self.context_mode,
            "context_type": (
                "sanitized TRAIN-report signal-wording snippets + structured slot signature"
                if self.context_mode == "sanitized_exemplar_snippets"
                else (
                    "sanitized report exemplars + structured slot signature"
                    if self.context_mode == "sanitized_exemplars"
                    else "patient-independent style cues only + structured slot signature"
                )
            ),
            "raw_reports_inserted_into_prompt": False,
            "sanitized_reports_inserted_into_prompt": self.context_mode in {"sanitized_exemplars", "sanitized_exemplar_snippets"},
            "visual_rerank": self.visual_rerank,
            "rerank_pool": self.rerank_pool,
            "indexed_split": "train",
            "n_indexed_cases": len(entries),
            "indexed_case_ids": [str(e["case_id"]) for e in entries],
            "manifest_sha256": sha256_file(self.manifest_path),
            "integrity": "assert_rag_train_only passed before persistence",
        }
        (self.index_path / "index_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if self.index is None:
            self.index = faiss.read_index(str(self.index_path / "faiss_index.bin"))
        if self.entries is None:
            with (self.index_path / "reports.pkl").open("rb") as f:
                self.entries = pickle.load(f)
        assert_rag_train_only(pd.read_csv(self.manifest_path), self.entries)

    def save_visual_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        """Persist frozen-V3.1 visual retrieval vectors for TRAIN cases only."""
        clean = {}
        for case_id, vec in mapping.items():
            arr = np.asarray(vec, dtype="float32").reshape(-1)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            clean[str(case_id)] = arr
        with (self.index_path / "visual_embeddings.pkl").open("wb") as f:
            pickle.dump(clean, f)
        self.visual_vectors = clean

    def _load_visual_embeddings(self) -> None:
        if self.visual_vectors is not None:
            return
        path = self.index_path / "visual_embeddings.pkl"
        if path.exists():
            with path.open("rb") as f:
                self.visual_vectors = pickle.load(f)
        else:
            self.visual_vectors = {}

    def retrieve(
        self,
        slots: Dict,
        *,
        query_text: Optional[str] = None,
        exclude_case_id: Optional[str] = None,
        query_visual: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Retrieve top-k TRAIN reports.

        Clinical-slot compatibility remains the primary criterion, matching the
        thesis design. When available, a frozen 3D visual embedding is used only
        to rank otherwise slot-compatible candidates, so retrieval contributes
        case-specific information without using validation/test labels.
        """
        self._load()
        if self.query_mode == "structured_slots":
            query = structured_report_from_slots(slots)
        else:
            query = query_text or structured_report_from_slots(slots)
        if self.add_slot_signature:
            query += "\nSTRUCTURED FINDINGS: " + slot_signature(slots)

        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        k = min(len(self.entries), max(self.rerank_pool, self.top_k * 8, self.top_k + 1))
        scores, ids = self.index.search(emb, k)

        qv = None
        if self.visual_rerank and query_visual is not None:
            self._load_visual_embeddings()
            qv = np.asarray(query_visual, dtype="float32").reshape(-1)
            qnorm = float(np.linalg.norm(qv))
            if qnorm > 0:
                qv = qv / qnorm
            else:
                qv = None

        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            entry = dict(self.entries[int(idx)])
            if exclude_case_id and str(entry["case_id"]) == str(exclude_case_id):
                continue
            match = sum(
                int(int(entry["slots"].get(key)) == int(slots.get(key)))
                for key in ("edema", "enhancing", "necrotic", "hemisphere", "extent")
            )
            entry["similarity"] = float(score)
            entry["slot_matches"] = int(match)
            visual_similarity = -2.0
            if qv is not None and self.visual_vectors:
                rv = self.visual_vectors.get(str(entry["case_id"]))
                if rv is not None and len(rv) == len(qv):
                    visual_similarity = float(np.dot(qv, rv))
            entry["visual_similarity"] = float(visual_similarity)
            candidates.append(entry)

        # Slot match is still primary. Visual similarity breaks ties among
        # clinically compatible cases; text-embedding similarity is tertiary.
        candidates.sort(
            key=lambda x: (
                x["slot_matches"],
                x.get("visual_similarity", -2.0),
                x["similarity"],
            ),
            reverse=True,
        )
        result = candidates[: self.top_k]
        if exclude_case_id and any(str(x["case_id"]) == str(exclude_case_id) for x in result):
            raise AssertionError("RAG self-retrieval exclusion failed")
        return result
