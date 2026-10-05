"""Hybrid structured-and-visual 3D visual-prefix + LoRA language model.

V3.1 keeps the successful V3 base-VLM grounding and fixes RAG prompt leakage by:
- using the TinyLlama chat template consistently;
- inserting only compact patient-independent style cues from retrieval;
- supervising only assistant/report target tokens;
- using strict deployment-time candidate diagnostics and base fallback;
- composing the delivered report from auditable slots plus safe descriptors.
"""
from __future__ import annotations

from typing import Dict, Optional
import re
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.report_targets import extent_text, ground_generated_report, hemisphere_text, preserve_safe_generated_report, ensure_two_section_report


class Hybrid3DVLM(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        vcfg = config["vlm"]
        channels = [int(x) for x in vcfg.get("vision_channels", [16, 32, 64])]
        num_visual_tokens = int(vcfg.get("num_visual_tokens", 8))
        if num_visual_tokens != 8:
            raise ValueError("This thesis architecture is fixed to 8 visual tokens (2x2x2); config must set num_visual_tokens: 8.")
        self.num_visual_tokens = num_visual_tokens
        self.vision_encoder = nn.Sequential(
            nn.Conv3d(1, channels[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(channels[0]), nn.GELU(),
            nn.Conv3d(channels[0], channels[1], 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(channels[1]), nn.GELU(),
            nn.Conv3d(channels[1], channels[2], 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(channels[2]), nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )

        model_name = vcfg["llm_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, low_cpu_mem_usage=True)
        except TypeError:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
        if bool(vcfg.get("gradient_checkpointing", True)):
            self.llm.gradient_checkpointing_enable()
            self.llm.enable_input_require_grads()
            self.llm.config.use_cache = False
        lora = LoraConfig(
            r=int(vcfg["lora_rank"]),
            lora_alpha=int(vcfg["lora_alpha"]),
            target_modules=list(vcfg["lora_target_modules"]),
            lora_dropout=float(vcfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora)
        hidden = int(self.llm.config.hidden_size)
        self.vision_projector = nn.Sequential(nn.Linear(channels[-1], hidden), nn.LayerNorm(hidden))

    def visual_tokens(self, mri: torch.Tensor) -> torch.Tensor:
        if mri.dim() == 4:
            mri = mri.unsqueeze(0)
        feat = self.vision_encoder(mri)
        feat = feat.flatten(2).transpose(1, 2)
        return self.vision_projector(feat)

    def _system_message(self) -> str:
        mode = str(self.config.get("rag", {}).get("context_mode", "style_cues"))
        if mode == "sanitized_exemplar_snippets":
            return (
                "Generate a concise experimental brain-tumour MRI report. "
                "The current CNN-predicted structured findings are the only patient-specific factual contract. "
                "Two retrieved TRAINING-report snippets may guide signal wording only; never copy patient facts from them. "
                "State all five structured findings without contradiction. "
                "Do not add measurements, precise lobar locations, history, treatment, recommendations, surgery, mass effect, "
                "ventricular findings, haemorrhage, diffusion findings, pathology labels, or other diagnoses. "
                "Output exactly one Findings section followed by exactly one Impression section."
            )
        if mode == "sanitized_exemplars":
            retrieval_rule = (
                "Retrieved material consists of sanitised TRAINING-report exemplars selected by structured clinical-slot similarity. "
                "They may guide report wording and organisation, but they are not evidence for the current patient and must never override the current structured findings. "
            )
        else:
            retrieval_rule = (
                "Retrieved material, when present, supplies writing-style vocabulary only and is not patient evidence. "
            )
        return (
            "You generate concise experimental brain-tumour MRI reports for research. "
            "The current CNN-predicted structured findings are the factual contract. "
            "State every structured finding and never contradict it. "
            + retrieval_rule +
            "Do not output instructions, prompt text, retrieved-context labels, patient history, recommendations, surgery, "
            "measurements, precise lobar location, pathology labels, mass effect, midline shift, hydrocephalus, diffusion restriction, haemorrhage, "
            "inflammation, treatment response, ischemia, or other diagnoses. "
            "Return exactly one Findings section followed by exactly one Impression section."
        )
    def _user_message(self, slots: Dict, rag_context: str = "") -> str:
        s = (
            "Current structured findings:\n"
            f"Oedema: {'present' if int(slots['edema']) else 'absent'}\n"
            f"Enhancing tumour component: {'present' if int(slots['enhancing']) else 'absent'}\n"
            f"Necrotic/non-enhancing core: {'present' if int(slots['necrotic']) else 'absent'}\n"
            f"Dominant hemisphere: {hemisphere_text(slots)}\n"
            f"Approximate tumour extent: {extent_text(slots)}\n"
        )
        mode = str(self.config.get("rag", {}).get("context_mode", "style_cues"))
        if rag_context:
            if mode == "sanitized_exemplar_snippets":
                s += (
                    "\nTwo retrieved TRAINING-report wording snippets (style/signal wording only):\n"
                    + rag_context.strip() + "\n"
                )
            elif mode == "sanitized_exemplars":
                s += (
                    "\nRetrieved sanitised training-report exemplars for wording/organisation only. "
                    "Current structured findings above remain authoritative:\n" + rag_context.strip() + "\n"
                )
            else:
                s += "\nOptional style vocabulary from similar training reports: " + rag_context.strip() + "\n"
        s += (
            "\nWrite exactly:\n"
            "Findings:\n"
            "One concise paragraph explicitly stating all five structured findings. "
            "You may use at most one retrieved generic signal-wording sentence.\n\n"
            "Impression:\n"
            "One concise summary sentence consistent with the structured findings."
        )
        return s
    def _prompt(self, slots: Dict, rag_context: str = "") -> str:
        messages = [
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": self._user_message(slots, rag_context)},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = (
                "System: " + messages[0]["content"] + "\n\n"
                "User: " + messages[1]["content"] + "\n\nAssistant:\n"
            )
        # V3.2 failed because long retrieved exemplars could be silently truncated
        # at the right edge, removing the output-format instruction / assistant
        # generation marker. V3.2.1 never silently truncates a prompt.
        max_prompt = int(self.config["vlm"]["max_prompt_length"])
        n_tokens = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        if n_tokens > max_prompt:
            raise RuntimeError(
                f"Prompt length {n_tokens} exceeds max_prompt_length={max_prompt}. "
                "Reduce RAG context instead of truncating the assistant-generation boundary."
            )
        return prompt
    def _rag_context(
        self,
        slots: Dict,
        retriever=None,
        exclude_case_id: Optional[str] = None,
        query_visual=None,
        return_items: bool = False,
    ):
        if retriever is None:
            return ("", []) if return_items else ""
        items = retriever.retrieve(
            slots,
            exclude_case_id=exclude_case_id,
            query_visual=query_visual,
        )
        rcfg = self.config.get("rag", {})
        mode = str(rcfg.get("context_mode", "style_cues"))

        if mode == "sanitized_exemplar_snippets":
            blocks = []
            for item in items[: int(rcfg.get("top_k", 2))]:
                snippet = str(item.get("v321_snippet") or "").strip()
                if not snippet:
                    continue
                blocks.append(f"Exemplar {len(blocks)+1}: {snippet}")
            context = "\n".join(blocks)
            return (context, items) if return_items else context

        if mode == "sanitized_exemplars":
            max_context = int(rcfg.get("max_context_chars", 1200))
            min_matches = int(rcfg.get("min_exemplar_slot_matches", 5))
            blocks = []
            for item in items:
                if int(item.get("slot_matches", 0)) < min_matches:
                    continue
                exemplar = str(item.get("exemplar_text") or "").strip()
                if not exemplar:
                    continue
                block = f"Exemplar {len(blocks)+1}:\n{exemplar}"
                candidate = "\n\n".join(blocks + [block])
                if len(candidate) > max_context:
                    break
                blocks.append(block)
                if len(blocks) >= int(rcfg.get("top_k", 2)):
                    break
            context = "\n\n".join(blocks)
            return (context, items) if return_items else context

        max_cues = int(rcfg.get("max_style_cues", 3))
        cues = []
        for item in items:
            for cue in str(item.get("style_text") or "").split(";"):
                cue = re.sub(r"\s+", " ", cue).strip(" ,;:-")
                if cue and cue not in cues:
                    cues.append(cue)
                if len(cues) >= max_cues:
                    break
            if len(cues) >= max_cues:
                break
        context = "; ".join(cues)
        return (context, items) if return_items else context

    @torch.no_grad()
    def visual_retrieval_vector(self, mri: torch.Tensor):
        """Frozen visual representation used only to rerank slot-compatible TRAIN cases."""
        if mri.dim() == 4:
            mri = mri.unsqueeze(0)
        device = next(self.parameters()).device
        tokens = self.visual_tokens(mri.to(device)).float()
        vec = tokens.mean(dim=1)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.detach().cpu().numpy()
    def _tokenize_prompt_target(self, prompt: str, target_report: str, device: torch.device):
        """Tokenise without silent prompt truncation; only report target tokens are supervised."""
        vcfg = self.config["vlm"]
        target = target_report.strip() + self.tokenizer.eos_token
        p = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        t = self.tokenizer(target, return_tensors="pt", truncation=False)
        if p.input_ids.shape[1] > int(vcfg["max_prompt_length"]):
            raise RuntimeError("Prompt exceeded configured token budget before training.")
        if t.input_ids.shape[1] > int(vcfg["max_target_length"]):
            raise RuntimeError("Target exceeded configured token budget before training.")
        prompt_ids, target_ids = p.input_ids.to(device), t.input_ids.to(device)
        ignore_prompt = torch.full_like(prompt_ids, -100)
        return prompt_ids, target_ids, ignore_prompt
    def compute_loss(
        self,
        mri: torch.Tensor,
        slots: Dict,
        target_report: str,
        retriever=None,
        case_id: Optional[str] = None,
        rag_context_override: Optional[str] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if mri.dim() == 4:
            mri = mri.unsqueeze(0)
        mri = mri.to(device)
        visual = self.visual_tokens(mri)
        if rag_context_override is None:
            qv = None
            if retriever is not None and bool(self.config.get("rag", {}).get("visual_rerank", False)):
                qv = torch.nn.functional.normalize(visual.float().mean(dim=1), p=2, dim=1).detach().cpu().numpy()
            rag_context = self._rag_context(slots, retriever, exclude_case_id=case_id, query_visual=qv)
        else:
            rag_context = str(rag_context_override)
        prompt = self._prompt(slots, rag_context)
        prompt_ids, target_ids, ignore_prompt = self._tokenize_prompt_target(prompt, target_report, device)
        text_emb = self.llm.get_input_embeddings()(torch.cat([prompt_ids, target_ids], dim=1))
        visual = visual.to(dtype=text_emb.dtype)
        inputs_embeds = torch.cat([visual, text_emb], dim=1)
        ignore_visual = torch.full((1, visual.shape[1]), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore_visual, ignore_prompt, target_ids], dim=1)
        attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention, labels=labels)
        return out.loss
    def _generate_once_detailed(self, mri: torch.Tensor, slots: Dict, rag_context: str = "") -> Dict[str, str]:
        device = next(self.parameters()).device
        if mri.dim() == 4:
            mri = mri.unsqueeze(0)
        mri = mri.to(device)
        visual = self.visual_tokens(mri)
        prompt = self._prompt(slots, rag_context)
        vcfg = self.config["vlm"]
        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if tokenized.input_ids.shape[1] > int(vcfg["max_prompt_length"]):
            raise RuntimeError("Prompt exceeded configured token budget before generation.")
        ids = tokenized.input_ids.to(device)
        text_emb = self.llm.get_input_embeddings()(ids)
        visual = visual.to(dtype=text_emb.dtype)
        inputs_embeds = torch.cat([visual, text_emb], dim=1)
        attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        temperature = float(vcfg.get("temperature", 0.0))
        do_sample = bool(vcfg.get("do_sample", False)) and temperature > 0.0
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention,
            "max_new_tokens": int(vcfg.get("max_new_tokens", 160)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # With inputs_embeds these constraints apply only to generated tokens.
        # Keep them optional and disabled by default in V3.1 to avoid noisy warnings.
        rep = float(vcfg.get("repetition_penalty", 1.0))
        ngram = int(vcfg.get("no_repeat_ngram_size", 0))
        if rep != 1.0:
            gen_kwargs["repetition_penalty"] = rep
        if ngram > 0:
            gen_kwargs["no_repeat_ngram_size"] = ngram
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = float(vcfg.get("top_p", 0.92))
        generated = self.llm.generate(**gen_kwargs)
        raw_text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        final_report = ground_generated_report(raw_text, slots)
        return {"raw_generation": raw_text, "final_report": final_report}

    @torch.no_grad()
    def generate_report_detailed(self, mri: torch.Tensor, slots: Dict, retriever=None, case_id: Optional[str] = None) -> Dict[str, object]:
        """Generate base/RAG candidates and apply a strict predicted-slot-only gate."""
        from src.report_metrics import generation_diagnostics, candidate_is_acceptable

        preserve = bool(self.config.get("rag", {}).get("preserve_safe_generation", False))
        preserve_min = float(self.config.get("rag", {}).get("preserve_min_conditioning", 1.0))

        base = self._generate_once_detailed(mri, slots, rag_context="")
        base_gate_text = ensure_two_section_report(base["raw_generation"]) if base["raw_generation"].strip() else base["raw_generation"]
        base_diag = generation_diagnostics(base_gate_text, slots)
        if preserve and candidate_is_acceptable(base_gate_text, slots, min_conditioning=preserve_min):
            base["final_report"] = preserve_safe_generated_report(base_gate_text)

        if retriever is None:
            return {
                **base,
                "selection": "base_only",
                "raw_diagnostics": generation_diagnostics(base["raw_generation"], slots),
                "gate_diagnostics": base_diag,
            }

        qv = self.visual_retrieval_vector(mri) if bool(self.config.get("rag", {}).get("visual_rerank", False)) else None
        rag_context, retrieved_items = self._rag_context(
            slots,
            retriever,
            exclude_case_id=case_id,
            query_visual=qv,
            return_items=True,
        )
        rag_candidate = self._generate_once_detailed(mri, slots, rag_context=rag_context)
        rag_gate_text = ensure_two_section_report(rag_candidate["raw_generation"]) if rag_candidate["raw_generation"].strip() else rag_candidate["raw_generation"]
        rag_raw_diag = generation_diagnostics(rag_candidate["raw_generation"], slots)
        rag_diag = generation_diagnostics(rag_gate_text, slots)

        if preserve and candidate_is_acceptable(rag_gate_text, slots, min_conditioning=preserve_min):
            rag_candidate["final_report"] = preserve_safe_generated_report(rag_gate_text)

        min_cond = float(self.config.get("rag", {}).get("candidate_min_raw_conditioning", 0.90))
        rag_ok = candidate_is_acceptable(rag_gate_text, slots, min_conditioning=min_cond)
        tolerance = float(self.config.get("rag", {}).get("max_raw_conditioning_drop_vs_base", 0.02))
        if bool(base_diag["structure_compliant"]) and float(base_diag["conditioning_consistency"]) >= min_cond:
            rag_ok = rag_ok and (
                float(rag_diag["conditioning_consistency"]) + tolerance
                >= float(base_diag["conditioning_consistency"])
            )

        chosen = rag_candidate if rag_ok else base
        retrieval_log = [
            {
                "case_id": str(x.get("case_id")),
                "slot_matches": int(x.get("slot_matches", 0)),
                "text_similarity": float(x.get("similarity", 0.0)),
                "visual_similarity": float(x.get("visual_similarity", -2.0)),
                "snippet": str(x.get("v321_snippet") or ""),
            }
            for x in retrieved_items[: int(self.config.get("rag", {}).get("top_k", 2))]
        ]
        return {
            "raw_generation": rag_candidate["raw_generation"],
            "final_report": chosen["final_report"],
            "selection": "rag_candidate" if rag_ok else "base_fallback",
            "rag_context": rag_context,
            "retrieved_cases": retrieval_log,
            "base_candidate_raw": base["raw_generation"],
            "base_candidate_final": base["final_report"],
            "rag_candidate_raw": rag_candidate["raw_generation"],
            "rag_candidate_normalized": rag_gate_text,
            "rag_candidate_final": rag_candidate["final_report"],
            "base_candidate_diagnostics": base_diag,
            "rag_candidate_raw_diagnostics": rag_raw_diag,
            "rag_candidate_diagnostics": rag_diag,
            "base_candidate_conditioning_consistency": float(base_diag["conditioning_consistency"]),
            "rag_candidate_conditioning_consistency": float(rag_diag["conditioning_consistency"]),
            "base_candidate_unsupported_flag_count": int(base_diag["unsupported_flag_count"]),
            "rag_candidate_unsupported_flag_count": int(rag_diag["unsupported_flag_count"]),
        }

    @torch.no_grad()
    def generate_report(self, mri: torch.Tensor, slots: Dict, retriever=None, case_id: Optional[str] = None) -> str:
        return str(self.generate_report_detailed(mri, slots, retriever=retriever, case_id=case_id)["final_report"])
