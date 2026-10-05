"""Deterministic structured report baseline."""
from __future__ import annotations
from typing import Dict
from src.report_targets import structured_report_from_slots


class GroundedTemplateBaseline:
    def generate_report(self, slots: Dict) -> Dict[str, str]:
        full = structured_report_from_slots(slots)
        findings = full.split("Findings:", 1)[1].split("Impression:", 1)[0].strip()
        impression = full.split("Impression:", 1)[1].strip()
        return {"findings": findings, "impression": impression, "full_report": full}
