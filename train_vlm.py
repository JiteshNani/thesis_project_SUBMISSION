"""Train the hybrid TinyLlama VLM using frozen CNN-predicted slots.

V3.1 uses two train-only phases:
1) base predicted-slot + visual training with validation-loss checkpoint selection;
2) one low-LR mixed retrieval-aware epoch (base prompts + safe style-cue RAG
   prompts) so inference-time retrieval is not an unseen prompt distribution.

Ground-truth segmentation-derived slots are never passed to the VLM and the
held-out test split is never touched here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.data_utils import BraTSSlotDataset
from src.reproducibility import save_run_metadata, set_global_seed
from src.report_targets import grounded_vlm_training_target
from src.slot_inference import load_slot_model, predict_slots
from src.vlm_model import Hybrid3DVLM
from src.rag_retriever import RAGRetriever


def _compact_state(model: Hybrid3DVLM):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("vision_encoder.")
        or name.startswith("vision_projector.")
        or ".lora_" in name
    }


def _save_trainable_checkpoint(model: Hybrid3DVLM, path: Path, **metadata) -> None:
    payload = {"model_state": _compact_state(model), "config": config, **metadata}
    torch.save(payload, path)


def _restore_trainable_checkpoint(model: Hybrid3DVLM, path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected VLM checkpoint keys while restoring best base checkpoint: {unexpected}")
    return payload if isinstance(payload, dict) else {}


def _make_optimizer(model: Hybrid3DVLM, lora_lr: float, projector_lr: float, weight_decay: float):
    llm_params = [p for n, p in model.named_parameters() if n.startswith("llm.") and p.requires_grad]
    visual_params = [p for n, p in model.named_parameters() if not n.startswith("llm.") and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": llm_params, "lr": float(lora_lr)},
            {"params": visual_params, "lr": float(projector_lr)},
        ],
        weight_decay=float(weight_decay),
    )


def _target(model: Hybrid3DVLM, report_text: str, slots: dict) -> str:
    return grounded_vlm_training_target(
        report_text,
        slots,
        max_reference_chars=int(model.config["vlm"].get("grounded_target_reference_chars", 220)),
    )


@torch.no_grad()
def _validation_loss(model, slot_model, loader, device, use_amp: bool, max_cases: int | None = None, retriever=None, rag_fraction: float = 0.0) -> float | None:
    model.eval()
    slot_model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if max_cases is not None and idx >= max_cases:
            break
        mri = batch["flair"].to(device)
        slots = predict_slots(slot_model, mri)[0]
        target = _target(model, batch["report_text"][0], slots)
        use_rag = retriever is not None and rag_fraction > 0 and ((idx % 100) / 100.0) < float(rag_fraction)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model.compute_loss(
                mri, slots, target,
                retriever=retriever if use_rag else None,
                case_id=str(batch["case_id"][0]),
            )
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else None


def _train_epoch(model, slot_model, loader, optimizer, scaler, device, use_amp, accum, max_grad_norm, *, max_train_batches=None, retriever=None, rag_fraction: float = 0.0):
    model.train()
    slot_model.eval()
    losses = []
    micro_count = 0
    optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(loader):
        if max_train_batches is not None and bi >= int(max_train_batches):
            break
        mri = batch["flair"].to(device)
        with torch.no_grad():
            slots = predict_slots(slot_model, mri)[0]
        target = _target(model, batch["report_text"][0], slots)
        # Deterministic mixed exposure: e.g. fraction=0.5 -> every other batch uses RAG.
        use_rag = retriever is not None and rag_fraction > 0 and ((bi % 100) / 100.0) < float(rag_fraction)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model.compute_loss(
                mri, slots, target,
                retriever=retriever if use_rag else None,
                case_id=str(batch["case_id"][0]),
            )
            scaled_loss = loss / accum
        scaler.scale(scaled_loss).backward()
        micro_count += 1
        if micro_count % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

    remainder = micro_count % accum
    if remainder:
        scaler.unscale_(optimizer)
        correction = float(accum) / float(remainder)
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else None


def _ensure_train_only_rag_index() -> RAGRetriever:
    rag = RAGRetriever(config)
    index = Path(config["rag"]["index_path"]) / "faiss_index.bin"
    reports = Path(config["rag"]["index_path"]) / "reports.pkl"
    if not index.exists() or not reports.exists():
        print("RAG-aware phase: building verified train-only style-cue index...")
        rag.build_index()
    else:
        rag._load()
    return rag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test limiter; omit for final thesis training.")
    p.add_argument("--max-val-cases", type=int, default=None, help="Smoke-test validation limiter; omit for final thesis training.")
    p.add_argument("--skip-rag-aware", action="store_true", help="Debug only; final V3.1 validation must not use this.")
    args = p.parse_args()

    set_global_seed(int(config["project"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type != "cuda":
        print("WARNING: VLM training is running on CPU. For the authoritative thesis run, use CUDA.")

    train_ds = BraTSSlotDataset(config["data"]["manifest_path"], split="train", config=config, include_slots=False)
    val_ds = BraTSSlotDataset(config["data"]["manifest_path"], split="val", config=config, include_slots=False)
    cfg = config["vlm"]
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    slot_model = load_slot_model(config, device)
    slot_model.eval()
    for param in slot_model.parameters():
        param.requires_grad_(False)

    model = Hybrid3DVLM(config).to(device)
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = int(cfg.get("gradient_accumulation_steps", 4))
    epochs = int(args.epochs or cfg["epochs"])
    max_train_batches = args.max_train_batches if args.max_train_batches is not None else cfg.get("max_train_batches")
    max_val_cases = args.max_val_cases
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    history = []
    best_val = float("inf")
    best_epoch = None
    checkpoint = Path(config["paths"]["models_dir"]) / "hybrid3dvlm_trainable.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    optimizer = _make_optimizer(model, cfg["lora_learning_rate"], cfg["projector_learning_rate"], cfg.get("weight_decay", 0.01))

    # Phase 1: stable base VLM.
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model, slot_model, train_loader, optimizer, scaler, device, use_amp, accum, max_grad_norm,
            max_train_batches=max_train_batches,
        )
        val_loss = _validation_loss(model, slot_model, val_loader, device, use_amp, max_val_cases)
        history.append({"phase": "base", "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss is not None and val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = epoch
            _save_trainable_checkpoint(model, checkpoint, selected_epoch=int(epoch), validation_loss=float(best_val), training_phase="base_best")

    if best_epoch is None:
        raise RuntimeError("No validation loss was produced; VLM checkpoint was not selected.")

    # Always restore the selected base checkpoint before the low-LR RAG-aware phase.
    _restore_trainable_checkpoint(model, checkpoint, device)

    rag_epochs = 0 if args.skip_rag_aware else int(config.get("rag", {}).get("rag_aware_epochs", 1))
    rag_val_loss = None
    if rag_epochs > 0:
        rag = _ensure_train_only_rag_index()
        rag_fraction = float(config["rag"].get("rag_aware_fraction", 0.5))
        optimizer = _make_optimizer(
            model,
            config["rag"].get("rag_aware_lora_learning_rate", 2e-5),
            config["rag"].get("rag_aware_projector_learning_rate", 1e-5),
            cfg.get("weight_decay", 0.01),
        )
        for r_epoch in range(1, rag_epochs + 1):
            train_loss = _train_epoch(
                model, slot_model, train_loader, optimizer, scaler, device, use_amp, accum, max_grad_norm,
                max_train_batches=max_train_batches, retriever=rag, rag_fraction=rag_fraction,
            )
            rag_val_loss = _validation_loss(
                model, slot_model, val_loader, device, use_amp, max_val_cases,
                retriever=rag, rag_fraction=rag_fraction,
            )
            history.append({"phase": "rag_aware_mixed", "epoch": r_epoch, "train_loss": train_loss, "validation_loss": rag_val_loss, "rag_fraction": rag_fraction})
            print(f"RAG-aware epoch {r_epoch:02d}/{rag_epochs} train_loss={train_loss:.4f} mixed_val_loss={rag_val_loss:.4f}")

        _save_trainable_checkpoint(
            model, checkpoint,
            selected_epoch=int(best_epoch),
            validation_loss=float(best_val),
            training_phase="base_best_plus_rag_aware_mixed",
            rag_aware_epochs=int(rag_epochs),
            rag_aware_validation_loss=float(rag_val_loss) if rag_val_loss is not None else None,
        )

    Path("results").mkdir(exist_ok=True)
    Path("results/vlm_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_run_metadata(
        "results/vlm_run_metadata.json",
        config=config,
        extra={
            "checkpoint": str(checkpoint),
            "base_epochs_executed": epochs,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_val),
            "rag_aware_epochs": int(rag_epochs),
            "rag_aware_validation_loss": float(rag_val_loss) if rag_val_loss is not None else None,
            "conditioning": "frozen slot-CNN predictions from FLAIR MRI for train/validation/test consistency",
            "training_target": "explicit five-slot anchor plus at most one patient-independent generic MRI signal descriptor",
            "rag_training_context": "patient-independent style cues only; mixed base/RAG prompt exposure; self-retrieval excluded",
            "prompt_supervision": "visual tokens and all prompt/RAG tokens masked with -100; only assistant report target supervised",
            "test_split_used_during_training": False,
        },
    )
    print("Saved:", checkpoint)


if __name__ == "__main__":
    main()
