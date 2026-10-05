"""Train the V3.2.1 RAG branch from the frozen V3.1 checkpoint.

V3.2.1 development changes:
- primary retrieval remains CNN-predicted clinical-slot similarity;
- frozen V3.1 3D visual embeddings rerank slot-compatible TRAIN cases;
- exactly two TRAIN reports are retrieved and reduced to compact safe signal-wording snippets;
- prompt truncation is prohibited;
- the RAG training target is explicitly conditioned on the retrieved snippet;
- only TinyLlama LoRA weights are updated; frozen vision BatchNorm buffers remain frozen;
- no TEST or validation case enters optimisation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import config as base_config
from src.data_utils import BraTSSlotDataset
from src.rag_retriever import RAGRetriever
from src.report_targets import grounded_vlm_training_target_v321
from src.reproducibility import save_run_metadata, set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_model import Hybrid3DVLM


BASE_CHECKPOINT = Path("models/hybrid3dvlm_trainable.pth")
V321_CHECKPOINT = Path("models/hybrid3dvlm_rag_v321.pth")
V321_INDEX = "data/rag_index_v321"


def v321_config() -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["rag"].update({
        "top_k": 2,
        "index_path": V321_INDEX,
        "query_mode": "structured_slots",
        "context_mode": "sanitized_exemplar_snippets",
        "visual_rerank": True,
        "rerank_pool": 48,
        "preserve_safe_generation": True,
        "preserve_min_conditioning": 1.0,
        "candidate_min_raw_conditioning": 1.0,
        "max_raw_conditioning_drop_vs_base": 0.0,
        "v321_lora_learning_rate": 2e-5,
        "v321_epochs": 2,
    })
    cfg["vlm"]["temperature"] = 0.0
    cfg["vlm"]["do_sample"] = False
    cfg["vlm"]["max_new_tokens"] = 160
    # Short snippets should fit within 512, but V3.2.1 raises instead of
    # silently truncating if that contract is ever violated.
    cfg["vlm"]["max_prompt_length"] = 512
    return cfg


def _load_compact_checkpoint(model: Hybrid3DVLM, path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return payload if isinstance(payload, dict) else {}


def _compact_state(model: Hybrid3DVLM):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("vision_encoder.")
        or name.startswith("vision_projector.")
        or ".lora_" in name
    }


def _build_text_index(cfg: dict) -> RAGRetriever:
    rag = RAGRetriever(cfg)
    index_dir = Path(cfg["rag"]["index_path"])
    if not (index_dir / "faiss_index.bin").exists() or not (index_dir / "reports.pkl").exists():
        print("Building V3.2.1 TRAIN-only FAISS report index...")
        rag.build_index()
    else:
        rag._load()
    return rag


@torch.no_grad()
def _build_visual_index(model: Hybrid3DVLM, dataset, rag: RAGRetriever, device: torch.device) -> None:
    """Build TRAIN-only frozen-V3.1 visual vectors; validation/test never enter."""
    print("Building frozen-V3.1 TRAIN visual reranking index...")
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mapping = {}
    for i, batch in enumerate(loader, 1):
        case_id = str(batch["case_id"][0])
        mri = batch["flair"].to(device)
        mapping[case_id] = model.visual_retrieval_vector(mri)[0]
        if i % 50 == 0 or i == len(loader):
            print(f"  visual index {i}/{len(loader)}")
    rag.save_visual_embeddings(mapping)
    if len(mapping) != len(dataset):
        raise AssertionError("Visual RAG index does not cover the complete TRAIN split")


def main() -> None:
    cfg = v321_config()
    seed = int(cfg["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing frozen V3.1 checkpoint: {BASE_CHECKPOINT}")

    train_ds = BraTSSlotDataset(cfg["data"]["manifest_path"], split="train", config=cfg, include_slots=False)
    slot_model = load_slot_model(cfg, device)
    slot_model.eval()
    for p in slot_model.parameters():
        p.requires_grad_(False)

    model = Hybrid3DVLM(cfg).to(device)
    base_meta = _load_compact_checkpoint(model, BASE_CHECKPOINT, device)

    rag = _build_text_index(cfg)
    _build_visual_index(model, train_ds, rag, device)

    # Only TinyLlama LoRA parameters are trainable.
    for name, p in model.named_parameters():
        p.requires_grad_(bool(name.startswith("llm.") and ".lora_" in name))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No LoRA parameters are trainable in V3.2.1")

    lr = float(cfg["rag"]["v321_lora_learning_rate"])
    epochs = int(cfg["rag"]["v321_epochs"])
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=float(cfg["vlm"].get("weight_decay", 0.01)))
    use_amp = bool(cfg["vlm"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = int(cfg["vlm"].get("gradient_accumulation_steps", 4))
    max_grad_norm = float(cfg["vlm"].get("max_grad_norm", 1.0))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Frozen really means frozen: prevent BatchNorm running-stat drift.
        model.vision_encoder.eval()
        model.vision_projector.eval()
        slot_model.eval()

        loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        micro_count = 0
        context_count = 0
        prompt_tokens = []

        for batch in loader:
            mri = batch["flair"].to(device)
            case_id = str(batch["case_id"][0])
            with torch.no_grad():
                slots = predict_slots(slot_model, mri)[0]
                qv = model.visual_retrieval_vector(mri)
                rag_context = model._rag_context(
                    slots,
                    rag,
                    exclude_case_id=case_id,
                    query_visual=qv,
                )
            if rag_context:
                context_count += 1
            target = grounded_vlm_training_target_v321(slots, rag_context)

            # Record actual prompt length and guarantee the generation boundary
            # is present; _prompt raises if the budget is exceeded.
            prompt = model._prompt(slots, rag_context)
            prompt_tokens.append(len(model.tokenizer(prompt, add_special_tokens=False).input_ids))

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.compute_loss(
                    mri,
                    slots,
                    target,
                    retriever=None,
                    case_id=case_id,
                    rag_context_override=rag_context,
                )
                scaled = loss / accum
            scaler.scale(scaled).backward()
            micro_count += 1

            if micro_count % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))

        remainder = micro_count % accum
        if remainder:
            scaler.unscale_(optimizer)
            correction = float(accum) / float(remainder)
            for p in trainable:
                if p.grad is not None:
                    p.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "n_train_cases": len(losses),
            "rag_context_coverage": float(context_count / max(1, len(losses))),
            "mean_prompt_tokens": float(np.mean(prompt_tokens)),
            "max_prompt_tokens": int(max(prompt_tokens)),
            "max_prompt_length": int(cfg["vlm"]["max_prompt_length"]),
        }
        history.append(row)
        print(
            f"V3.2.1 epoch {epoch:02d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"context={row['rag_context_coverage']:.3f} "
            f"prompt_tokens(mean/max)={row['mean_prompt_tokens']:.1f}/{row['max_prompt_tokens']}"
        )

    V321_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": _compact_state(model),
        "config": cfg,
        "source_checkpoint": str(BASE_CHECKPOINT),
        "source_checkpoint_metadata": {
            "selected_epoch": base_meta.get("selected_epoch"),
            "validation_loss": base_meta.get("validation_loss"),
            "training_phase": base_meta.get("training_phase"),
        },
        "training_phase": "v321_retrieval_conditioned_lora_adaptation",
        "epochs": epochs,
        "learning_rate": lr,
        "visual_rerank": True,
        "test_split_used_during_training": False,
    }
    torch.save(payload, V321_CHECKPOINT)

    Path("results").mkdir(exist_ok=True)
    Path("results/v321_rag_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_run_metadata(
        "results/v321_rag_run_metadata.json",
        config=cfg,
        extra={
            "source_v31_checkpoint": str(BASE_CHECKPOINT),
            "v321_checkpoint": str(V321_CHECKPOINT),
            "v321_design": "top-2 slot-primary retrieval + frozen visual reranking + sanitised TRAIN-report signal snippets",
            "trainable_branch": "TinyLlama LoRA only; visual encoder/projector parameters and BatchNorm buffers frozen",
            "raw_reports_inserted_into_prompt": False,
            "retrieved_report_snippets_inserted": True,
            "test_split_used_during_training": False,
            "epochs": epochs,
            "learning_rate": lr,
        },
    )
    print("Saved V3.2.1 checkpoint:", V321_CHECKPOINT)
    print("Frozen V3.1 checkpoint remains untouched:", BASE_CHECKPOINT)


if __name__ == "__main__":
    main()
