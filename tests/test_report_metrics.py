from src.report_metrics import report_slot_consistency, unsupported_content_flags


def _slots(**overrides):
    base = {"edema": 0, "enhancing": 0, "necrotic": 0, "hemisphere": 0, "extent": 0}
    base.update(overrides)
    return base


def test_negated_presence_slots_are_recognised_as_absent():
    text = """Findings:\nThere is a small left-hemisphere tumour. No oedema, enhancing component, or necrotic core is identified.\n\nImpression:\nSmall left-sided brain tumour."""
    result = report_slot_consistency(text, _slots())
    assert result["edema"] == 1.0
    assert result["enhancing"] == 1.0
    assert result["necrotic"] == 1.0
    assert result["hemisphere"] == 1.0
    assert result["extent"] == 1.0
    assert result["mean"] == 1.0


def test_multiple_negation_phrasings_are_recognised():
    cases = [
        "No oedema is identified.",
        "Without oedema.",
        "Absence of oedema.",
        "No evidence of oedema.",
        "Oedema is not present.",
    ]
    for phrase in cases:
        text = f"Findings:\nThere is a small left-hemisphere tumour. {phrase} No enhancement. No necrotic core.\n\nImpression:\nSmall left-sided tumour."
        result = report_slot_consistency(text, _slots())
        assert result["edema"] == 1.0, phrase


def test_positive_presence_slots_are_recognised():
    text = """Findings:\nThere is a large right-hemisphere tumour with oedema, enhancement and a necrotic core.\n\nImpression:\nLarge right-sided tumour."""
    result = report_slot_consistency(text, _slots(edema=1, enhancing=1, necrotic=1, hemisphere=1, extent=2))
    assert result["mean"] == 1.0


def test_contradictory_negation_is_not_scored_correct():
    text = """Findings:\nThere is a moderate left-hemisphere tumour. Oedema is present. No oedema is identified. No enhancement or necrotic core.\n\nImpression:\nModerate left-sided tumour."""
    result = report_slot_consistency(text, _slots(edema=1, enhancing=0, necrotic=0, extent=1))
    assert result["edema"] == 0.0


def test_negated_anatomy_or_bare_units_are_not_automatically_flagged():
    text = """Findings:\nNo frontal lobe involvement is identified. No numeric measurement is provided in mm.\n\nImpression:\nNo additional unsupported positive claim."""
    flags = unsupported_content_flags(text)
    assert "asserted_precise_lobar_location" not in flags
    assert "asserted_numeric_measurement" not in flags


def test_positive_unsupported_claims_are_conservatively_flagged():
    text = """Findings:\nA 23 mm lesion is centred in the frontal lobe with inflammatory change.\n\nImpression:\nFollow-up MRI is recommended."""
    flags = unsupported_content_flags(text)
    assert "asserted_numeric_measurement" in flags
    assert "asserted_precise_lobar_location" in flags
    assert "asserted_inflammation" in flags
    assert "recommendation" in flags


def test_non_enhancing_core_does_not_count_as_positive_enhancement():
    text = """Findings:
There is a small left-hemisphere tumour with a non-enhancing core and no oedema.

Impression:
Small left-sided tumour with necrotic/non-enhancing core."""
    result = report_slot_consistency(text, _slots(edema=0, enhancing=0, necrotic=1, hemisphere=0, extent=0))
    assert result["enhancing"] == 0.0
    assert result["parsed_states"]["enhancing"] is None
    assert result["necrotic"] == 1.0


def test_enhancing_component_can_coexist_with_non_enhancing_core():
    text = """Findings:
There is a large right-hemisphere tumour with oedema, an enhancing tumour component, and a necrotic/non-enhancing core.

Impression:
Large right-sided tumour with enhancement and necrotic core."""
    result = report_slot_consistency(text, _slots(edema=1, enhancing=1, necrotic=1, hemisphere=1, extent=2))
    assert result["enhancing"] == 1.0
    assert result["necrotic"] == 1.0
    assert result["mean"] == 1.0


def test_non_enhancing_core_alone_does_not_prove_global_absence_of_enhancement():
    text = """Findings:
There is a small left-hemisphere tumour with a necrotic/non-enhancing core and no oedema.

Impression:
Small left-sided tumour."""
    result = report_slot_consistency(text, _slots(edema=0, enhancing=0, necrotic=1, hemisphere=0, extent=0))
    assert result["enhancing"] == 0.0
    assert result["parsed_states"]["enhancing"] is None


def test_v3_structured_template_explicitly_states_negative_slots():
    from src.report_targets import structured_report_from_slots
    text = structured_report_from_slots(_slots())
    result = report_slot_consistency(text, _slots())
    assert result["mean"] == 1.0
    assert "No enhancing tumour component" in text


def test_v3_grounded_final_report_removes_direct_laterality_and_extent_contradictions():
    from src.report_targets import ground_generated_report
    raw = """Findings:\nThere is a large right-hemisphere lesion with oedema and necrosis.\n\nImpression:\nLarge right-sided lesion."""
    slots = _slots(edema=1, enhancing=0, necrotic=1, hemisphere=0, extent=0)
    final = ground_generated_report(raw, slots)
    result = report_slot_consistency(final, slots)
    assert result["mean"] == 1.0
    assert "right" not in final.lower()


def test_v31_finalizer_rejects_unsupported_measurement_and_lobar_location_instead_of_rewriting():
    from src.report_targets import ground_generated_report
    raw = """Findings:\nThere is a small right-sided tumour in the frontal lobe measuring 25 mm. The lesion demonstrates heterogeneous signal characteristics.\n\nImpression:\nSmall right-sided tumour."""
    slots = _slots(edema=1, enhancing=1, necrotic=1, hemisphere=1, extent=0)
    final = ground_generated_report(raw, slots)
    assert "25 mm" not in final
    assert "frontal" not in final.lower()
    assert "heterogeneous signal characteristics" in final.lower()
    assert report_slot_consistency(final, slots)["mean"] == 1.0


def test_v31_finalizer_does_not_globally_replace_unrelated_extent_words():
    from src.report_targets import ground_generated_report
    raw = """Findings:\nTreatment response is moderate. The lesion demonstrates mixed signal characteristics.\n\nImpression:\nModerate response."""
    slots = _slots(edema=1, enhancing=1, necrotic=1, hemisphere=1, extent=0)
    final = ground_generated_report(raw, slots)
    assert "treatment response" not in final.lower()
    assert "mixed signal characteristics" in final.lower()
    assert "response is small" not in final.lower()


def test_v31_finalizer_removes_prompt_leakage_and_exactly_normalises_sections():
    from src.report_targets import ground_generated_report
    from src.report_metrics import structure_compliance, prompt_leakage_flags
    raw = """<state all five structured findings first>\nOutput: Findings:\nTraining-set reference snippets for WRITING STYLE ONLY.\nFindings:\nThe lesion demonstrates heterogeneous signal characteristics.\nImpression:\nOne sentence.\nFindings:\nextra"""
    slots = _slots(edema=1, enhancing=1, necrotic=1, hemisphere=0, extent=1)
    final = ground_generated_report(raw, slots)
    assert structure_compliance(final)
    assert not prompt_leakage_flags(final)
    assert final.count("Findings:") == 1
    assert final.count("Impression:") == 1


def test_v31_generation_diagnostics_catches_known_rag_failure_modes():
    from src.report_metrics import generation_diagnostics
    bad = """Findings:\nThere are three patients with brain injury in the frontal lobe measuring 20 mm.\nOperative item: Surgery is recommended.\nImpression:\n<one concise summary sentence> Хронология"""
    d = generation_diagnostics(bad, _slots(edema=1, enhancing=1, necrotic=1, hemisphere=1, extent=1))
    assert d["unsupported_flag_count"] >= 3
    assert d["prompt_leakage"]
    assert d["non_english_fragment"]
