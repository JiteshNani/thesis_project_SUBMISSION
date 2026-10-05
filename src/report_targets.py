"""Structured report utilities shared by template, RAG and VLM.

V3.1 tightens the evidence contract:
- all five structured slots are stated explicitly;
- training targets contain only slot-grounded facts plus a very small set of
  generic MRI signal-style descriptors extracted from the training report;
- retrieved reports are converted to style cues, not patient-specific prose;
- delivered reports are composed clause-by-clause without global word
  replacement and always contain exactly one Findings and one Impression.

The deterministic finaliser is a reporting guardrail. It is not a claim of
clinical validation or clinical safety.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List
import re


SLOT_KEYS = ("edema", "enhancing", "necrotic", "hemisphere", "extent")

# Phrases that are generic enough to be used as style cues without importing
# another patient's anatomy, measurements, diagnosis, history or treatment.
_STYLE_CUE_PATTERNS = (
    (r"\bheterogeneous\b", "heterogeneous signal characteristics"),
    (r"\bmixed\s+signal(?:s|\s+intensit(?:y|ies))?\b", "mixed signal characteristics"),
    (r"\bvariable\s+signal(?:\s+intensit(?:y|ies))?\b", "variable signal intensity"),
    (r"\bpatchy\s+(?:high|low|mixed)?\s*signal", "patchy signal variation"),
    (r"\buneven\s+signal", "uneven signal intensity"),
)

_META_PATTERNS = (
    r"<[^>]{1,200}>",
    r"\btraining[- ]set reference\b",
    r"\bwriting style only\b",
    r"\bmust be stated\b",
    r"\bmandatory\b",
    r"\bsaniti[sz]ed\b",
    r"\bnever treat retrieved\b",
    r"\boutput\s*:\s*findings\b",
    r"\bcurrent patient structured findings\b",
)

_UNSUPPORTED_PATTERNS = (
    r"\b(?:recommend(?:ed|ation|s)?|follow[- ]?up|clinical correlation)\b",
    r"\b(?:surgery|surgical|operative|resection)\b",
    r"\b(?:treatment response|radiotherapy|chemotherapy)\b",
    r"\b(?:inflammation|inflammatory)\b",
    r"\b(?:ischemia|ischaemia|brain injury)\b",
    r"\b(?:history of|previous surgery|prior surgery|symptoms include)\b",
    r"\b(?:three|four|multiple)\s+patients?\b",
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?)\b",
    r"\b(?:frontal|parietal|temporal|occipital|frontotemporal|temporoparietal|frontoparietal|parieto[- ]occipital)\b",
)


def _plain(v: Any) -> int:
    try:
        return int(v.item())
    except Exception:
        return int(v)


def slots_to_plain_dict(slots: Dict[str, Any]) -> Dict[str, int]:
    return {k: _plain(v) for k, v in slots.items() if k in set(SLOT_KEYS)}


def hemisphere_text(slots: Dict[str, Any]) -> str:
    return "right" if _plain(slots.get("hemisphere", 1)) == 1 else "left"


def extent_text(slots: Dict[str, Any]) -> str:
    return {0: "small", 1: "moderate", 2: "large"}.get(_plain(slots.get("extent", 1)), "moderate")


def slot_signature(slots: Dict[str, Any]) -> str:
    s = slots_to_plain_dict(slots)
    return (
        f"edema={s.get('edema', 0)}; enhancing={s.get('enhancing', 0)}; necrotic={s.get('necrotic', 0)}; "
        f"hemisphere={hemisphere_text(s)}; extent={extent_text(s)}"
    )


def structured_findings_from_slots(slots: Dict[str, Any]) -> str:
    """Explicitly state all five structured findings, including negatives."""
    s = slots_to_plain_dict(slots)
    hemi, extent = hemisphere_text(s), extent_text(s)
    lines = [f"There is a {extent} {hemi}-hemisphere intra-axial tumour."]
    lines.append("Surrounding oedema is present." if s.get("edema", 0) else "No surrounding oedema is identified.")
    lines.append(
        "An enhancing tumour component is present."
        if s.get("enhancing", 0)
        else "No enhancing tumour component is identified."
    )
    lines.append(
        "A necrotic/non-enhancing core is present."
        if s.get("necrotic", 0)
        else "No necrotic/non-enhancing core is identified."
    )
    return " ".join(lines)


def structured_report_from_slots(slots: Dict[str, Any]) -> str:
    s = slots_to_plain_dict(slots)
    hemi, extent = hemisphere_text(s), extent_text(s)
    findings = structured_findings_from_slots(s)
    impression = f"{extent.capitalize()} {hemi}-sided brain tumour with structured findings as described above."
    return f"Findings:\n{findings}\n\nImpression:\n{impression}"


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def strip_instruction_leakage(text: str) -> str:
    """Remove obvious control/meta fragments before report parsing."""
    out = text or ""
    for pattern in _META_PATTERNS:
        out = re.sub(pattern, " ", out, flags=re.I)
    # Remove extra generated section labels other than the first outer pair.
    out = re.sub(r"\b(?:Output|Operative item|Operatives?|Recommendations?|Differential Diagnosis|Clinical History)\s*:\s*", " ", out, flags=re.I)
    return re.sub(r"[ \t]+", " ", out).strip()


def ensure_two_section_report(text: str) -> str:
    """Normalise arbitrary generation to exactly one Findings/Impression pair."""
    text = strip_instruction_leakage(text)
    # Work case-insensitively while retaining original content.
    f_matches = list(re.finditer(r"\bFindings\s*:\s*", text, flags=re.I))
    i_matches = list(re.finditer(r"\bImpression\s*:\s*", text, flags=re.I))

    if f_matches:
        start = f_matches[0].end()
        if i_matches and i_matches[0].start() > f_matches[0].start():
            imp = next((m for m in i_matches if m.start() > f_matches[0].start()), None)
            findings = text[start:imp.start()].strip() if imp else text[start:].strip()
            impression = text[imp.end():].strip() if imp else ""
        else:
            findings = text[start:].strip()
            impression = ""
    else:
        if i_matches:
            findings = text[:i_matches[0].start()].strip()
            impression = text[i_matches[0].end():].strip()
        else:
            findings, impression = text.strip(), ""

    # Any nested section labels in content are meta/noise and are removed.
    findings = re.sub(r"\b(?:Findings|Impression)\s*:\s*", " ", findings, flags=re.I)
    impression = re.sub(r"\b(?:Findings|Impression)\s*:\s*", " ", impression, flags=re.I)
    findings = re.sub(r"\s+", " ", findings).strip()
    impression = re.sub(r"\s+", " ", impression).strip()

    if not impression:
        first = _sentences(findings)
        impression = first[0] if first else "Structured brain tumour report."
    return f"Findings:\n{findings}\n\nImpression:\n{impression}".strip()


def _strip_report_headers(text: str) -> str:
    text = re.sub(r"\bFindings\s*:\s*", "", text or "", flags=re.I)
    text = re.split(r"\bImpression\s*:\s*", text, maxsplit=1, flags=re.I)[0]
    return re.sub(r"\s+", " ", text).strip()


def extract_style_cues(reference_report: str, *, max_cues: int = 3) -> List[str]:
    """Extract a tiny, patient-independent vocabulary cue set from a report.

    This is intentionally conservative. RAG is allowed to contribute writing
    style/vocabulary, but never another patient's side, lobe, measurement,
    diagnosis, treatment, recommendation or history.
    """
    text = _strip_report_headers(reference_report).lower()
    cues: List[str] = []
    for pattern, cue in _STYLE_CUE_PATTERNS:
        if re.search(pattern, text, flags=re.I) and cue not in cues:
            cues.append(cue)
        if len(cues) >= int(max_cues):
            break
    return cues


def sanitize_reference_for_conditioning(reference_report: str, slots: Dict[str, Any] | None = None, *, max_chars: int = 220) -> str:
    """Return style cues only; no patient-specific reference prose is retained."""
    del slots  # retained for API compatibility; style cues are patient-independent.
    cues = extract_style_cues(reference_report, max_cues=3)
    out = "; ".join(cues)
    return out[: int(max_chars)].rstrip(" ,;:")


def _descriptor_sentence_from_cues(cues: Iterable[str]) -> str:
    cues = list(dict.fromkeys(str(x).strip() for x in cues if str(x).strip()))
    if not cues:
        return ""
    # One concise sentence keeps targets stable and avoids teaching long copied prose.
    if "heterogeneous signal characteristics" in cues:
        return "The lesion demonstrates heterogeneous signal characteristics."
    if "mixed signal characteristics" in cues:
        return "The lesion demonstrates mixed signal characteristics."
    if "variable signal intensity" in cues:
        return "The lesion demonstrates variable signal intensity."
    if "patchy signal variation" in cues:
        return "Patchy signal variation is present within the lesion."
    if "uneven signal intensity" in cues:
        return "The lesion demonstrates uneven signal intensity."
    return ""


def grounded_vlm_training_target(reference_report: str, slots: Dict[str, Any], *, max_reference_chars: int = 220) -> str:
    """Create a clean predicted-slot-aligned VLM target.

    The target never contains quantitative measurements, precise lobar location,
    patient history, treatment/recommendations or unrelated diagnoses. A single
    generic signal-style sentence may be learned from the training report.
    """
    anchor = structured_report_from_slots(slots)
    anchor_findings = anchor.split("Findings:", 1)[1].split("Impression:", 1)[0].strip()
    anchor_impression = anchor.split("Impression:", 1)[1].strip()
    cue_text = sanitize_reference_for_conditioning(reference_report, slots, max_chars=max_reference_chars)
    descriptor = _descriptor_sentence_from_cues(cue_text.split(";"))
    findings = anchor_findings + (" " + descriptor if descriptor else "")
    return f"Findings:\n{findings}\n\nImpression:\n{anchor_impression}"


def _contains_non_ascii_word(text: str) -> bool:
    # Permit common punctuation but reject obvious foreign-script fragments.
    return bool(re.search(r"[^\x00-\x7F]", text or ""))


def _is_safe_descriptor_sentence(sentence: str, slots: Dict[str, Any]) -> bool:
    """Return True only for a narrow generic MRI signal descriptor sentence."""
    s = sentence.strip()
    low = s.lower()
    if not s or len(s) > 220:
        return False
    if _contains_non_ascii_word(s):
        return False
    if any(re.search(p, low, flags=re.I) for p in _META_PATTERNS):
        return False
    if any(re.search(p, low, flags=re.I) for p in _UNSUPPORTED_PATTERNS):
        return False
    # Anchor already states all structured clinical facts. Do not repeat or alter them.
    if re.search(r"\b(?:left|right|small|moderate|large|edema|oedema|enhanc|necros|necrotic|non[- ]enhancing)\w*\b", low):
        return False
    # Only generic signal-description language is eligible for retention.
    return bool(re.search(r"\b(?:heterogeneous|mixed signal|variable signal|patchy signal|uneven signal|signal intensity|signal characteristics)\b", low))


def safe_generated_descriptors(raw_text: str, slots: Dict[str, Any], *, max_sentences: int = 1) -> List[str]:
    """Extract a tiny safe descriptor set from raw generation for final delivery."""
    two = ensure_two_section_report(raw_text)
    findings = two.split("Findings:", 1)[1].split("Impression:", 1)[0].strip()
    out: List[str] = []
    for sentence in _sentences(findings):
        if _is_safe_descriptor_sentence(sentence, slots):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if not cleaned.endswith((".", "!", "?")):
                cleaned += "."
            if cleaned not in out:
                out.append(cleaned)
        if len(out) >= int(max_sentences):
            break
    return out


def ground_generated_report(raw_text: str, slots: Dict[str, Any]) -> str:
    """Compose the delivered report from the slot anchor plus safe descriptors.

    There is deliberately no global left/right or small/moderate/large string
    replacement. Unsupported clauses are rejected instead of being edited into
    superficially plausible statements.
    """
    anchor = structured_report_from_slots(slots)
    anchor_findings = anchor.split("Findings:", 1)[1].split("Impression:", 1)[0].strip()
    anchor_impression = anchor.split("Impression:", 1)[1].strip()
    descriptors = safe_generated_descriptors(raw_text, slots, max_sentences=1)
    findings = anchor_findings + ((" " + descriptors[0]) if descriptors else "")
    return f"Findings:\n{findings}\n\nImpression:\n{anchor_impression}".strip()

# ---------------------------------------------------------------------------
# V3.2 RAG extension
# ---------------------------------------------------------------------------
# V3.2 stays in the original thesis direction (retrieve similar training
# reports) but converts each retrieved report into a safe report exemplar before
# it reaches the language model.  The current patient's predicted slots remain
# the only source of patient-specific facts.

_V32_EXEMPLAR_REJECT_PATTERNS = (
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?)\b",
    r"\b(?:frontal|parietal|temporal|occipital|frontotemporal|temporoparietal|frontoparietal|parieto[- ]occipital)\b",
    r"\b(?:corpus callosum|thalamus|basal ganglia|cerebellum|brainstem|insula)\b",
    r"\b(?:midline shift|mass effect|hydrocephalus|ventricular compression|ventricular effacement)\b",
    r"\b(?:diffusion restriction|restricted diffusion|haemorrhage|hemorrhage|haemorrhagic|hemorrhagic)\b",
    r"\b(?:glioblastoma|\bgbm\b|glioma|astrocytoma|lymphoma|metast(?:asis|atic))\b",
    r"\b(?:resection cavity|post[- ]?operative|post[- ]?surgical)\b",
    *_UNSUPPORTED_PATTERNS,
)


def _simple_concept_state(text: str, terms: List[str]) -> int | None:
    """Small local negation parser used only to keep retrieved exemplars slot-consistent."""
    sentence = (text or "").lower()
    term_group = r"(?:" + "|".join(re.escape(t.lower()) for t in terms) + r")"
    if not re.search(rf"\b{term_group}\b", sentence):
        return None
    neg = bool(
        re.search(rf"\b(?:no evidence of|absence of|without|no)\b[^.!?]{{0,80}}\b{term_group}\b", sentence)
        or re.search(rf"\b{term_group}\b[^.!?]{{0,50}}\b(?:absent|not present|not seen|not identified|not demonstrated)\b", sentence)
    )
    return 0 if neg else 1


def _v32_sentence_matches_slots(sentence: str, slots: Dict[str, Any]) -> bool:
    """Reject any retrieved sentence that conflicts with the structured slot contract."""
    s = slots_to_plain_dict(slots)
    low = (sentence or "").lower()
    if not low or _contains_non_ascii_word(low):
        return False
    if any(re.search(p, low, flags=re.I) for p in _META_PATTERNS):
        return False
    if any(re.search(p, low, flags=re.I) for p in _V32_EXEMPLAR_REJECT_PATTERNS):
        return False

    hemi = hemisphere_text(s)
    opposite = "left" if hemi == "right" else "right"
    if re.search(rf"\b{opposite}\b", low):
        return False
    expected_extent = extent_text(s)
    for other in ("small", "moderate", "large"):
        if other != expected_extent and re.search(rf"\b{other}\b", low):
            return False

    edema_state = _simple_concept_state(sentence, ["edema", "oedema"])
    if edema_state is not None and edema_state != int(s.get("edema", 0)):
        return False

    # Do not count "non-enhancing core" as an assertion about enhancement.
    enh_text = re.sub(r"\bnon[- ]enhancing\s+core\b", " core ", sentence, flags=re.I)
    enh_state = _simple_concept_state(enh_text, ["enhancing", "enhancement", "enhancing component"])
    if enh_state is not None and enh_state != int(s.get("enhancing", 0)):
        return False

    nec_state = _simple_concept_state(sentence, ["necrotic", "necrosis", "necrotic core", "non-enhancing core"])
    if nec_state is not None and nec_state != int(s.get("necrotic", 0)):
        return False
    return True


def sanitize_reference_as_exemplar(
    reference_report: str,
    slots: Dict[str, Any],
    *,
    max_chars: int = 520,
    max_sentences: int = 4,
) -> str:
    """Create a report-like, slot-consistent exemplar from a retrieved TRAIN report.

    This is deliberately more informative than the V3.1 style-cue string, but it
    never forwards raw patient-specific report text unfiltered.  Sentences that
    contain measurements, precise anatomy, history/treatment, unsupported
    diagnoses, or facts that contradict the retrieved case's five structured
    slots are removed.  A canonical impression is then added so the exemplar
    retains the report form expected by the thesis.
    """
    source = _strip_report_headers(reference_report)
    kept: List[str] = []
    for sentence in _sentences(source):
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if not _v32_sentence_matches_slots(cleaned, slots):
            continue
        # Retain only sentences that contribute report language rather than
        # generic administrative prose.
        if not re.search(
            r"\b(?:lesion|tumou?r|mass|intra[- ]axial|edema|oedema|enhanc|necros|non[- ]enhancing|FLAIR|signal|heterogeneous|mixed)\w*\b",
            cleaned,
            flags=re.I,
        ):
            continue
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        if cleaned not in kept:
            kept.append(cleaned)
        if len(kept) >= int(max_sentences):
            break

    # Always give the model a complete, fact-safe exemplar even when no raw
    # sentence survives sanitisation.  A safe generic signal descriptor from the
    # retrieved report may still provide useful phrasing variation.
    anchor = structured_report_from_slots(slots)
    anchor_findings = anchor.split("Findings:", 1)[1].split("Impression:", 1)[0].strip()
    anchor_impression = anchor.split("Impression:", 1)[1].strip()
    cue_text = sanitize_reference_for_conditioning(reference_report, slots, max_chars=220)
    descriptor = _descriptor_sentence_from_cues(cue_text.split(";"))

    if kept:
        parts = list(kept)
        combined = " ".join(parts)
        anchor_sentences = _sentences(anchor_findings)
        # Add only factual anchor clauses that are missing from the retained
        # exemplar language.  This keeps the exemplar report-like and compact
        # instead of duplicating every slot sentence.
        hemi = hemisphere_text(slots)
        ext = extent_text(slots)
        if not (re.search(rf"\b{hemi}\b", combined, flags=re.I) and re.search(rf"\b{ext}\b", combined, flags=re.I)):
            parts.append(anchor_sentences[0])
        if _simple_concept_state(combined, ["edema", "oedema"]) is None:
            parts.append(anchor_sentences[1])
        enh_combined = re.sub(r"\bnon[- ]enhancing\s+core\b", " core ", combined, flags=re.I)
        if _simple_concept_state(enh_combined, ["enhancing", "enhancement", "enhancing component"]) is None:
            parts.append(anchor_sentences[2])
        if _simple_concept_state(combined, ["necrotic", "necrosis", "necrotic core", "non-enhancing core"]) is None:
            parts.append(anchor_sentences[3])
        findings = " ".join(parts)
    else:
        findings = anchor_findings
    if descriptor and descriptor.lower() not in findings.lower():
        findings += " " + descriptor

    out = f"Findings:\n{findings}\n\nImpression:\n{anchor_impression}".strip()
    if len(out) <= int(max_chars):
        return out
    # Never cut mid-token/mid-sentence.  If the richer exemplar is too long,
    # fall back to the compact fact-safe anchor plus descriptor.
    compact_findings = anchor_findings + ((" " + descriptor) if descriptor else "")
    compact = f"Findings:\n{compact_findings}\n\nImpression:\n{anchor_impression}".strip()
    return compact[: int(max_chars)].rstrip()


def grounded_vlm_training_target_v32(reference_report: str, slots: Dict[str, Any], *, max_reference_chars: int = 520) -> str:
    """RAG-specific target with modest report-language variation and fixed facts.

    The target is built from the current training case's report but filtered by
    the CNN-predicted slot contract.  This lets the RAG-specific fine-tune learn
    report phrasing without learning measurements, lobar locations, treatment,
    history, or other unverified patient-specific assertions.
    """
    return sanitize_reference_as_exemplar(
        reference_report,
        slots,
        max_chars=int(max_reference_chars),
        max_sentences=3,
    )


def preserve_safe_generated_report(raw_text: str) -> str:
    """Normalise a candidate without replacing its wording.

    Callers MUST first pass the raw candidate through the V3.2 strict generation
    diagnostics.  This function performs formatting only; it is not a safety
    check by itself.
    """
    return ensure_two_section_report(raw_text)


# ---------------------------------------------------------------------------
# V3.2.1 RAG refinement
# ---------------------------------------------------------------------------
# V3.2.1 keeps the thesis direction of retrieving two clinically similar
# TRAINING reports, but reduces each retrieved report to a compact, auditable
# signal-wording snippet. This avoids prompt truncation and patient-to-patient
# fact transfer while still letting retrieval contribute report language.

def safe_signal_descriptor_from_report(reference_report: str) -> str:
    """Return one patient-independent signal-description sentence."""
    low = re.sub(r"\s+", " ", reference_report or "").lower()
    has_signal = "signal" in low
    has_high = "high" in low
    has_low = "low" in low
    mixed = bool(re.search(r"\b(?:mixed|mixture|mix of|heterogeneous mix|heterogeneous mixture)\b", low))
    patchy = bool(re.search(r"\b(?:patchy|spotty|spotted|speckled|punctate|spot[- ]like)\b", low))
    uneven = "uneven" in low
    heterogeneous = "heterogeneous" in low

    if has_signal and has_high and has_low and mixed and patchy:
        return "The lesion demonstrates a heterogeneous mixture of high- and low-signal intensities with patchy high-signal areas."
    if has_signal and has_high and has_low and mixed:
        return "The lesion demonstrates a heterogeneous mixture of high- and low-signal intensities."
    if has_signal and patchy:
        return "The lesion demonstrates patchy high-signal areas within a heterogeneous signal pattern."
    if has_signal and uneven:
        return "The lesion demonstrates uneven heterogeneous signal intensity."
    if has_signal and heterogeneous:
        return "The lesion demonstrates heterogeneous signal characteristics."
    if has_signal:
        return "The lesion demonstrates variable signal intensity."
    return "The lesion demonstrates heterogeneous signal characteristics."


def sanitize_reference_as_v321_snippet(reference_report: str) -> str:
    """Convert a retrieved TRAINING report into one safe wording exemplar."""
    return safe_signal_descriptor_from_report(reference_report).strip()


_V321_SAFE_DESCRIPTORS = {
    "The lesion demonstrates a heterogeneous mixture of high- and low-signal intensities with patchy high-signal areas.",
    "The lesion demonstrates a heterogeneous mixture of high- and low-signal intensities.",
    "The lesion demonstrates patchy high-signal areas within a heterogeneous signal pattern.",
    "The lesion demonstrates uneven heterogeneous signal intensity.",
    "The lesion demonstrates heterogeneous signal characteristics.",
    "The lesion demonstrates variable signal intensity.",
}


def descriptor_from_rag_context(rag_context: str) -> str:
    """Extract the first sanitised signal descriptor in retrieval order."""
    for line in (rag_context or "").splitlines():
        cleaned = re.sub(r"^\s*Exemplar\s+\d+\s*:\s*", "", line, flags=re.I).strip()
        if cleaned in _V321_SAFE_DESCRIPTORS:
            return cleaned
    return ""


def grounded_vlm_training_target_v321(slots: Dict[str, Any], rag_context: str) -> str:
    """Create a retrieval-conditioned, slot-faithful two-section target."""
    report = structured_report_from_slots(slots)
    descriptor = descriptor_from_rag_context(rag_context)
    if descriptor:
        findings, impression = report.split("\n\nImpression:\n", 1)
        findings = findings.rstrip()
        if not findings.endswith((".", "!", "?")):
            findings += "."
        findings += " " + descriptor
        report = findings + "\n\nImpression:\n" + impression
    return ensure_two_section_report(report)
