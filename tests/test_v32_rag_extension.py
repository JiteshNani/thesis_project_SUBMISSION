from pathlib import Path


def _slots(**kw):
    base = {"edema": 1, "enhancing": 1, "necrotic": 1, "hemisphere": 1, "extent": 2}
    base.update(kw)
    return base


def test_v32_sanitized_exemplar_removes_patient_specific_unsupported_content():
    from src.report_targets import sanitize_reference_as_exemplar

    raw = (
        "Findings: There is a large right frontal lesion measuring 42 mm with mass effect and midline shift. "
        "Surrounding oedema is present. An enhancing tumour component is present. "
        "A necrotic core is present. The lesion demonstrates heterogeneous signal characteristics. "
        "Impression: Glioblastoma is suspected; surgical review is recommended."
    )
    out = sanitize_reference_as_exemplar(raw, _slots(), max_chars=800)
    low = out.lower()
    assert "42 mm" not in low
    assert "frontal" not in low
    assert "mass effect" not in low
    assert "midline shift" not in low
    assert "glioblastoma" not in low
    assert "surgical" not in low
    assert "recommended" not in low
    assert "right" in low
    assert "large" in low
    assert "oedema" in low or "edema" in low
    assert "enhanc" in low
    assert "necrot" in low
    assert out.count("Findings:") == 1
    assert out.count("Impression:") == 1


def test_v32_sanitized_exemplar_rejects_opposite_laterality_sentence():
    from src.report_targets import sanitize_reference_as_exemplar

    raw = "Findings: A large left-sided lesion demonstrates heterogeneous signal. Oedema is present. Impression: lesion."
    out = sanitize_reference_as_exemplar(raw, _slots(hemisphere=1), max_chars=800)
    # Canonical safe anchor must be right-sided and the contradictory source sentence removed.
    assert "right" in out.lower()
    assert "left" not in out.lower()


def test_v32_scripts_never_load_test_split_and_keep_v31_checkpoint_separate():
    train = Path("train_rag_v32.py").read_text(encoding="utf-8")
    val = Path("validate_v32_rag.py").read_text(encoding="utf-8")
    assert 'split="test"' not in train
    assert 'split="test"' not in val
    assert "models/hybrid3dvlm_trainable.pth" in train
    assert "models/hybrid3dvlm_rag_v32.pth" in train
    assert '"top_k": 2' in train
    assert '"context_mode": "sanitized_exemplars"' in train


def test_v32_keeps_v31_style_cue_mode_backward_compatible():
    src = Path("src/vlm_model.py").read_text(encoding="utf-8")
    assert '"style_cues"' in src
    assert "Optional style vocabulary" in src
    assert "Retrieved sanitised training-report exemplars" in src
