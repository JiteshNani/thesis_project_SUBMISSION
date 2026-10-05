"""Train the V3.2 RAG-enhanced VLM branch from the frozen V3.1 checkpoint.

Scientific contract
-------------------
- Slot CNN, segmentation U-Net, data split and V3.1 Base-VLM checkpoint are frozen.
- Training uses the TRAIN split only.
- Retrieval index contains TRAIN cases only and excludes the current case.
- RAG uses top-2 structured-slot-similar sanitised report exemplars.
- Only TinyLlama LoRA parameters are updated; vision features remain frozen.
- Output is saved to a NEW checkpoint; V3.1 is never overwritten.
- The held-out TEST split is never loaded here.
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
from src.report_targets import grounded_vlm_training_target_v32
from src.reproducibility import save_run_metadata, set_global_seed
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_model import Hybrid3DVLM


BASE_CHECKPOINT = Path("models/hybrid3dvlm_trainable.pth")
V32_CHECKPOINT = Path("models/hybrid3dvlm_rag_v32.pth")
V32_INDEX = "data/rag_index_v32"


def v32_config() -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["rag"].update({
        "top_k": 2,
        "index_path": V32_INDEX,
        "query_mode": "structured_slots",
        "context_mode": "sanitized_exemplars",
        "exemplar_chars_per_case": 520,
        "max_context_chars": 1200,
        "min_exemplar_slot_matches": 5,
        "preserve_safe_generation": True,
        "preserve_min_conditioning": 1.0,
        "candidate_min_raw_conditioning": 1.0,
        "max_raw_conditioning_drop_vs_base": 0.0,
        "v32_lora_learning_rate": 1e-5,
        "v32_epochs": 1,
    })
    # Keep the same deterministic decoding used by the authoritative Base VLM.
    cfg["vlm"]["temperature"] = 0.0
    cfg["vlm"]["do_sample"] = False
    cfg["vlm"]["max_new_tokens"] = int(cfg["vlm"].get("max_new_tokens", 160))
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


def _build_rag(cfg: dict) -> RAGRetriever:
    rag = RAGRetriever(cfg)
    index_dir = Path(cfg["rag"]["index_path"])
    index_file = index_dir / "faiss_index.bin"
    reports_file = index_dir / "reports.pkl"
    if not index_file.exists() or not reports_file.exists():
        print("Building V3.2 verified TRAIN-only top-2 sanitised-exemplar RAG index...")
        rag.build_index()
    else:
        rag._load()
    return rag


def main() -> None:
    cfg = v32_config()
    seed = int(cfg["project"].get("seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing frozen V3.1 checkpoint: {BASE_CHECKPOINT}. Do not retrain V3.1; restore the authoritative checkpoint first."
        )

    # Explicitly TRAIN only. No validation/test split enters optimisation.
    train_ds = BraTSSlotDataset(cfg["data"]["manifest_path"], split="train", config=cfg, include_slots=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    slot_model = load_slot_model(cfg, device)
    slot_model.eval()
    for p in slot_model.parameters():
        p.requires_grad_(False)

    model = Hybrid3DVLM(cfg).to(device)
    base_meta = _load_compact_checkpoint(model, BASE_CHECKPOINT, device)

    # RAG extension adapts language LoRA only.  Vision encoder/projector are
    # frozen so the already validated MRI representation is not retuned.
    for name, p in model.named_parameters():
        p.requires_grad_(bool(name.startswith("llm.") and ".lora_" in name))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No LoRA parameters are trainable in V3.2")

    rag = _build_rag(cfg)
    lr = float(cfg["rag"].get("v32_lora_learning_rate", 1e-5))
    epochs = int(cfg["rag"].get("v32_epochs", 1))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=float(cfg["vlm"].get("weight_decay", 0.01)))
    use_amp = bool(cfg["vlm"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = int(cfg["vlm"].get("gradient_accumulation_steps", 4))
    max_grad_norm = float(cfg["vlm"].get("max_grad_norm", 1.0))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        slot_model.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        micro_count = 0
        nonempty_context = 0
        for bi, batch in enumerate(train_loader):
            mri = batch["flair"].to(device)
            case_id = str(batch["case_id"][0])
            with torch.no_grad():
                slots = predict_slots(slot_model, mri)[0]
            target = grounded_vlm_training_target_v32(
                batch["report_text"][0],
                slots,
                max_reference_chars=int(cfg["rag"].get("exemplar_chars_per_case", 520)),
            )
            if model._rag_context(slots, rag, exclude_case_id=case_id):
                nonempty_context += 1
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.compute_loss(mri, slots, target, retriever=rag, case_id=case_id)
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

        mean_loss = float(np.mean(losses)) if losses else None
        coverage = float(nonempty_context / max(1, len(losses)))
        history.append({
            "epoch": epoch,
            "train_loss": mean_loss,
            "n_train_cases": len(losses),
            "rag_context_coverage": coverage,
        })
        print(f"V3.2 RAG epoch {epoch:02d}/{epochs} train_loss={mean_loss:.4f} context_coverage={coverage:.3f}")

    V32_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": _compact_state(model),
        "config": cfg,
        "source_checkpoint": str(BASE_CHECKPOINT),
        "source_checkpoint_metadata": {
            "selected_epoch": base_meta.get("selected_epoch"),
            "validation_loss": base_meta.get("validation_loss"),
            "training_phase": base_meta.get("training_phase"),
        },
        "training_phase": "v32_rag_only_lora_adaptation",
        "epochs": epochs,
        "learning_rate": lr,
        "test_split_used_during_training": False,
    }
    torch.save(payload, V32_CHECKPOINT)

    Path("results").mkdir(exist_ok=True)
    Path("results/v32_rag_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_run_metadata(
        "results/v32_rag_run_metadata.json",
        config=cfg,
        extra={
            "source_v31_checkpoint": str(BASE_CHECKPOINT),
            "v32_checkpoint": str(V32_CHECKPOINT),
            "v32_design": "top-2 structured-slot retrieval of sanitised TRAIN-report exemplars",
            "trainable_branch": "TinyLlama LoRA only; visual encoder/projector frozen",
            "raw_reports_inserted_into_prompt": False,
            "test_split_used_during_training": False,
            "epochs": epochs,
            "learning_rate": lr,
        },
    )
    print("Saved V3.2 checkpoint:", V32_CHECKPOINT)
    print("V3.1 checkpoint remains untouched:", BASE_CHECKPOINT)


if __name__ == "__main__":
    main()
