from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_vlm_training_uses_frozen_cnn_predicted_slots_not_batch_ground_truth_slots():
    src = _text("train_vlm.py")
    assert "include_slots=False" in src
    assert "predict_slots(slot_model, mri)[0]" in src
    assert 'batch["slots"]' not in src
    assert "requires_grad_(False)" in src


def test_vlm_test_inference_uses_cnn_predicted_slots():
    for rel in ("evaluate_vlm.py", "evaluate_vlm_rag.py"):
        src = _text(rel)
        assert "include_slots=False" in src
        assert "predict_slots(slot_model, mri)[0]" in src
        assert "ground_truth_slots" not in src
        assert 'batch["slots"]' not in src
        assert "generate_report_detailed(mri, pred" in src


def test_segmentation_is_separate_from_vlm_inputs():
    vlm = _text("src/vlm_model.py")
    train = _text("train_vlm.py")
    assert "seg_path" not in vlm
    assert "segmentation_model" not in vlm
    assert "segmentation_eval" not in vlm
    assert 'batch["seg"]' not in train
    assert 'batch["seg_brats"]' not in train


def test_laterality_contract_is_world_coordinate_based():
    src = _text("src/data_utils.py")
    assert "_world_x_coordinates" in src
    assert "midline_world_x_mm" in src
    assert "flair_path=row[\"flair_path\"]" in src
    assert "seg.shape[0] // 2" not in src


def test_hd95_contract_uses_monai_not_sampled_kdtree():
    src = _text("src/segmentation_eval.py")
    assert "compute_hausdorff_distance" in src
    assert "percentile=95.0" in src
    assert "spacing=" in src
    assert "cKDTree" not in src
    assert "max_points" not in src


def test_report_diagnostic_is_not_named_hallucination_rate():
    src = _text("compute_report_metrics.py")
    assert "rule_based_unsupported_content_flag_rate" in src
    assert "hallucination_rate" not in src


def test_exact_vlm_configuration_contract():
    cfg = _text("config.yaml")
    assert 'llm_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"' in cfg
    assert "lora_rank: 16" in cfg
    assert "lora_alpha: 32" in cfg
    assert 'lora_target_modules: ["q_proj", "v_proj"]' in cfg
    assert "num_visual_tokens: 8" in cfg


def test_pipeline_order_matches_methodology():
    src = _text("scripts/run_pipeline.ps1")
    ordered = [
        "prepare_data.py", "train_slot_model.py", "evaluate_slot_model.py",
        "train_segmentation_model.py", "evaluate_segmentation_model.py",
        "train_vlm.py", "evaluate_vlm.py", "build_rag_index.py",
        "evaluate_vlm_rag.py", "final_comparison.py",
    ]
    positions = [src.find(x) for x in ordered]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_authoritative_final_runner_exists():
    src = _text("scripts/run_final_experiment.ps1")
    assert "pytest" in src
    assert "run_pipeline.ps1" in src


def test_rag_constructor_keeps_config_and_cpu_embedding_device_contract():
    src = _text("src/rag_retriever.py")
    cfg = _text("config.yaml")
    assert "self.config = config" in src
    assert 'device=self.embedding_device' in src
    assert 'embedding_device: "cpu"' in cfg


def test_no_obsolete_model_names_or_old_result_constants_in_implementation_docs():
    targets = [
        "config.yaml", "README.md", "THESIS_ALIGNMENT.md",
        "train_vlm.py", "evaluate_vlm.py", "evaluate_vlm_rag.py",
        "src/vlm_model.py", "scripts/run_pipeline.ps1",
    ]
    combined = "\n".join(_text(x) for x in targets).lower()
    for banned in ("llama-3", "mistral-7b", "91.1%", "0.95 dice", "zero hallucination"):
        assert banned not in combined


def test_authoritative_runner_requires_imaging_dependencies_and_real_imaging_test():
    src = _text("scripts/run_final_experiment.ps1")
    assert "import nibabel, monai" in src
    assert "tests/test_imaging_core.py" in src


def test_vlm_training_uses_validation_split_for_checkpoint_selection():
    src = _text("train_vlm.py")
    assert 'split="val"' in src
    assert "_validation_loss" in src
    assert "best_validation_loss" in src
    assert "selected_epoch" in src
    assert "test_split_used_during_training" in src


def test_chapter_update_material_generator_is_result_derived():
    src = _text("generate_thesis_update_material.py")
    assert 'thesis_results_bundle.json' in src
    assert "chapter4" in src.lower()
    assert "hallucination rate" in src.lower()


def test_final_result_integrity_gate_exists():
    src = _text("verify_final_results.py")
    assert "RAG index" in src
    assert "held-out test" in src.lower()
    assert "template_gt_oracle" in src
    assert "vlm_rag" in src


def test_vlm_evaluations_preserve_raw_and_postprocessed_reports():
    base = _text("evaluate_vlm.py")
    rag = _text("evaluate_vlm_rag.py")
    model = _text("src/vlm_model.py")
    metrics = _text("compute_report_metrics.py")
    assert "generate_report_detailed" in base
    assert '"vlm_base_raw"' in base
    assert "generate_report_detailed" in rag
    assert '"vlm_rag_raw"' in rag
    assert '"raw_generation"' in model and '"final_report"' in model
    assert "raw_generation_two_section_structure_rate" in metrics
    assert "final_pipeline_two_section_structure_rate" in metrics


def test_smoke_pipeline_stops_before_full_test_integrity_stages():
    src = _text("scripts/run_pipeline.ps1")
    smoke_start = src.index('Write-Host "SMOKE TEST MODE')
    full_start = src.index('} else {', smoke_start)
    smoke_block = src[smoke_start:full_start]
    assert "--max-val-cases 1" in smoke_block
    assert "SMOKE PIPELINE PASS" in smoke_block
    assert "return" in smoke_block


def test_v31_rag_query_does_not_depend_on_generated_draft_by_default():
    src = _text("src/rag_retriever.py")
    cfg = _text("config.yaml")
    assert 'query_mode: "structured_slots"' in cfg
    assert 'if self.query_mode == "structured_slots"' in src
    assert "query_text or structured_report_from_slots" in src
    assert "raw_reports_inserted_into_prompt" in src


def test_v31_prompt_uses_chat_template_and_style_cues_only():
    src = _text("src/vlm_model.py")
    assert "apply_chat_template" in src
    assert "Current structured findings" in src
    assert "Optional style vocabulary" in src
    assert "ground_generated_report" in src
    assert "base_fallback" in src
    assert "<state all five structured findings first" not in src


def test_v31_prompt_tokens_are_masked_from_loss():
    src = _text("src/vlm_model.py")
    assert "ignore_prompt = torch.full_like(prompt_ids, -100)" in src
    assert "ignore_visual" in src
    assert "torch.cat([ignore_visual, ignore_prompt, target_ids]" in src


def test_v31_training_has_rag_aware_train_only_phase():
    src = _text("train_vlm.py")
    assert "rag_aware_epochs" in src
    assert "_ensure_train_only_rag_index" in src
    assert "exclude_case_id" in _text("src/rag_retriever.py")
    assert 'split="test"' not in src


def test_v3_validation_gate_is_validation_only():
    src = _text("validate_v3_report_models.py")
    assert 'split="val"' in src
    assert 'split="test"' not in src
    assert '"test_split_used": False' in src
