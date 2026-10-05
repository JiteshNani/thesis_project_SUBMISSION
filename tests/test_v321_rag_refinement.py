from src.report_targets import (
    safe_signal_descriptor_from_report,
    sanitize_reference_as_v321_snippet,
    grounded_vlm_training_target_v321,
)
from src.report_metrics import candidate_is_acceptable


def _slots():
    return {"edema": 1, "enhancing": 1, "necrotic": 1, "hemisphere": 0, "extent": 2}


def test_v321_signal_snippet_removes_patient_specific_location_by_construction():
    raw = (
        "The lesion area is in the left frontal and parietal lobes with a heterogeneous mixture "
        "of high and low signals and patchy high-signal areas. Ventricular compression is present."
    )
    out = sanitize_reference_as_v321_snippet(raw)
    assert "frontal" not in out.lower()
    assert "parietal" not in out.lower()
    assert "ventric" not in out.lower()
    assert "left" not in out.lower()
    assert "high-" in out.lower() and "low-signal" in out.lower()


def test_v321_training_target_uses_retrieved_snippet_and_keeps_slots():
    ctx = (
        "Exemplar 1: The lesion demonstrates a heterogeneous mixture of high- and low-signal intensities with patchy high-signal areas.\n"
        "Exemplar 2: The lesion demonstrates variable signal intensity."
    )
    target = grounded_vlm_training_target_v321(_slots(), ctx)
    assert target.count("Findings:") == 1
    assert target.count("Impression:") == 1
    assert "large left-hemisphere" in target.lower()
    assert "surrounding oedema is present" in target.lower()
    assert "enhancing tumour component is present" in target.lower()
    assert "necrotic/non-enhancing core is present" in target.lower()
    assert "heterogeneous mixture of high- and low-signal" in target.lower()


def test_format_only_normalisation_can_restore_missing_findings_label_safely():
    from src.report_targets import ensure_two_section_report
    raw = (
        "There is a large left-hemisphere intra-axial tumour. Surrounding oedema is present. "
        "An enhancing tumour component is present. A necrotic/non-enhancing core is present. "
        "The lesion demonstrates heterogeneous signal characteristics.\n\n"
        "Impression:\nLarge left-sided brain tumour with structured findings as described above."
    )
    norm = ensure_two_section_report(raw)
    assert candidate_is_acceptable(norm, _slots(), min_conditioning=1.0)


def test_signal_descriptor_is_deterministic():
    raw = "Mixed heterogeneous high and low signals with speckled high-signal areas."
    a = safe_signal_descriptor_from_report(raw)
    b = safe_signal_descriptor_from_report(raw)
    assert a == b
