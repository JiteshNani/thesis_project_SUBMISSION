"""Report-generation metrics and transparent reliability diagnostics.

These functions are reproducible text/slot diagnostics. They are not clinical
hallucination metrics and must not be described as radiologist-validated safety
measures.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional
import numpy as np
from src.report_targets import extent_text, hemisphere_text


def structure_compliance(text: str) -> bool:
    return (
        (text or "").count("Findings:") == 1
        and (text or "").count("Impression:") == 1
        and (text or "").find("Findings:") < (text or "").find("Impression:")
    )


def _sentences(text: str) -> List[str]:
    return [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _concept_state(text: str, terms: List[str]) -> Optional[int]:
    positive = False
    negative = False
    escaped = [re.escape(t.lower()) for t in terms]
    term_group = r"(?:" + "|".join(escaped) + r")"
    for sentence in _sentences(text):
        if not re.search(rf"\b{term_group}\b", sentence):
            continue
        neg_patterns = [
            rf"\b(?:no evidence of|absence of|without|no)\b[^.!?]{{0,80}}\b{term_group}\b",
            rf"\b{term_group}\b[^.!?]{{0,50}}\b(?:is|are|was|were)?\s*(?:absent|not present|not seen|not identified|not visuali[sz]ed|not demonstrated)\b",
        ]
        is_neg = any(re.search(p, sentence) for p in neg_patterns)
        if not is_neg and re.match(r"^\s*(?:no|without)\b", sentence):
            before_contrast = re.split(r"\b(?:but|however|although|whereas)\b", sentence, maxsplit=1)[0]
            is_neg = bool(re.search(rf"\b{term_group}\b", before_contrast))
        if is_neg:
            negative = True
        else:
            positive = True
    if positive and not negative:
        return 1
    if negative and not positive:
        return 0
    return None


def report_slot_consistency(text: str, slots: Dict) -> Dict[str, float | int | None]:
    low = (text or "").lower()
    checks: Dict[str, float | int | None] = {}
    states: Dict[str, int | None] = {}
    terms = {
        "edema": ["edema", "oedema"],
        "enhancing": ["enhancing", "enhancement", "enhancing component"],
        "necrotic": ["necrotic", "necrosis", "necrotic core", "non-enhancing core"],
    }
    for key, words in terms.items():
        concept_text = text
        if key == "enhancing":
            concept_text = re.sub(r"\b(?:necrotic\s*/\s*)?non[- ]enhancing\s+core\b", " core ", concept_text, flags=re.I)
        state = _concept_state(concept_text, words)
        states[key] = state
        checks[key] = float(state is not None and state == int(slots[key]))

    hemi = hemisphere_text(slots)
    opposite = "left" if hemi == "right" else "right"
    checks["hemisphere"] = float(bool(re.search(rf"\b{hemi}\b", low)) and not bool(re.search(rf"\b{opposite}\b", low)))
    extent = extent_text(slots)
    other_extents = [x for x in ("small", "moderate", "large") if x != extent]
    checks["extent"] = float(bool(re.search(rf"\b{extent}\b", low)) and not any(re.search(rf"\b{x}\b", low) for x in other_extents))
    checks["mean"] = float(np.mean([float(checks[k]) for k in ("edema", "enhancing", "necrotic", "hemisphere", "extent")]))
    checks["parsed_states"] = states
    return checks


def _is_explicitly_negated(sentence: str, term_pattern: str) -> bool:
    patterns = [
        rf"\b(?:no evidence of|absence of|without|no|denies|negative for)\b[^.!?]{{0,100}}{term_pattern}",
        rf"{term_pattern}[^.!?]{{0,60}}\b(?:absent|not present|not seen|not identified|not demonstrated|excluded)\b",
    ]
    return any(re.search(p, sentence, flags=re.I) for p in patterns)


def unsupported_content_flags(text: str) -> List[str]:
    flags: set[str] = set()
    for sentence in _sentences(text):
        patterns = [
            ("recommendation", r"\b(?:recommend(?:ed|ation|s)?|follow[- ]?up|clinical correlation)\b"),
            ("asserted_surgery", r"\b(?:surgery|surgical|operative|resection)\b"),
            ("asserted_inflammation", r"\b(?:inflammation|inflammatory)\b"),
            ("asserted_treatment_context", r"\b(?:treatment response|radiotherapy|chemotherapy)\b"),
            ("asserted_ischemia_or_injury", r"\b(?:ischemia|ischaemia|brain injury)\b"),
            ("asserted_precise_lobar_location", r"\b(?:frontal|parietal|temporal|occipital|frontotemporal|temporoparietal|frontoparietal|parieto[- ]occipital)\b"),
            ("asserted_numeric_measurement", r"\b\d+(?:\.\d+)?\s*(?:mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?)\b"),
            ("asserted_patient_history", r"\b(?:history of|previous surgery|prior surgery|symptoms include)\b"),
            ("asserted_mass_effect_or_shift", r"\b(?:mass effect|midline shift|ventricular compression|ventricular effacement)\b"),
            ("asserted_hydrocephalus", r"\bhydrocephalus\b"),
            ("asserted_diffusion_or_haemorrhage", r"\b(?:diffusion restriction|restricted diffusion|haemorrhage|hemorrhage|haemorrhagic|hemorrhagic)\b"),
            ("asserted_unverified_pathology", r"\b(?:glioblastoma|gbm|glioma|astrocytoma|lymphoma|metastasis|metastatic)\b"),
            ("asserted_precise_deep_anatomy", r"\b(?:corpus callosum|thalamus|basal ganglia|cerebellum|brainstem|insula)\b"),
            ("multiple_patient_content", r"\b(?:two|three|four|multiple)\s+patients?\b"),
        ]
        for label, pat in patterns:
            if re.search(pat, sentence, flags=re.I) and not _is_explicitly_negated(sentence, pat):
                flags.add(label)
    return sorted(flags)


def prompt_leakage_flags(text: str) -> List[str]:
    checks = {
        "angle_placeholder": r"<[^>]{1,200}>",
        "training_reference_meta": r"\btraining[- ]set reference\b",
        "writing_style_meta": r"\bwriting style only\b",
        "mandatory_instruction": r"\bmandatory\b|\bmust be stated\b",
        "sanitizer_meta": r"\bsaniti[sz]ed\b",
        "retrieval_instruction": r"\bnever treat retrieved\b|\bnever override\b",
        "output_header_leak": r"\boutput\s*:\s*findings\b",
        "current_patient_prompt": r"\bcurrent patient structured findings\b",
    }
    low = text or ""
    return [name for name, pat in checks.items() if re.search(pat, low, flags=re.I)]


def non_english_fragment_flag(text: str) -> bool:
    # British/European punctuation remains ASCII in generated target language.
    # This intentionally catches obvious foreign-script leakage only.
    return bool(re.search(r"[^\x00-\x7F]", text or ""))


def truncation_flag(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    # Ends in a section label, dangling connector, or alphanumeric fragment
    # without sentence closure. Conservative: this is a generation-quality flag.
    if re.search(r"(?:Findings|Impression|Output)\s*:\s*$", s, flags=re.I):
        return True
    if re.search(r"\b(?:and|with|of|the|a|an|in|on|to|for|from|than|is|are)\s*$", s, flags=re.I):
        return True
    return len(s) > 80 and s[-1] not in ".!?"


def generation_diagnostics(text: str, slots: Dict) -> Dict[str, object]:
    unsupported = unsupported_content_flags(text)
    leakage = prompt_leakage_flags(text)
    return {
        "conditioning_consistency": float(report_slot_consistency(text, slots)["mean"]),
        "structure_compliant": bool(structure_compliance(text)),
        "unsupported_flags": unsupported,
        "unsupported_flag_count": int(len(unsupported)),
        "prompt_leakage_flags": leakage,
        "prompt_leakage": bool(leakage),
        "non_english_fragment": bool(non_english_fragment_flag(text)),
        "truncated_or_incomplete": bool(truncation_flag(text)),
    }


def candidate_is_acceptable(text: str, slots: Dict, *, min_conditioning: float = 0.90) -> bool:
    d = generation_diagnostics(text, slots)
    return bool(
        d["structure_compliant"]
        and float(d["conditioning_consistency"]) >= float(min_conditioning)
        and int(d["unsupported_flag_count"]) == 0
        and not d["prompt_leakage"]
        and not d["non_english_fragment"]
        and not d["truncated_or_incomplete"]
    )
