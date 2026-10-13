# Brain MRI report generation

Master's thesis by Jitesh Nagaraj Jalla at IU International University of Applied Sciences.

**Thesis title:** Reliable and Grounded Vision Language Analysis for Structured Brain Tumor MRI Report Generation.

This research project studies structured report generation from brain MRI using paired BraTS and TextBraTS data. It combines a 3D CNN for structured findings, tumor segmentation, a TinyLlama-based vision-language model and retrieval from training-set reports. The experiments examine whether generated reports follow the predicted findings and the required structure.

## Implementation

- Python and PyTorch for model training and evaluation.
- MONAI and NiBabel for medical-imaging components.
- Hugging Face Transformers and PEFT/LoRA for adapting TinyLlama 1.1B.
- Sentence Transformers and FAISS for retrieving report context from the training split.
- Separate training, validation and test splits; checks restrict the retrieval index to training reports.
- Scripts for segmentation, structured-slot prediction, report generation and result verification.

## Dataset and evaluation

The saved experiment summary records 368 complete paired cases: 257 training, 55 validation and 56 test cases. Dataset access is handled separately; follow the original dataset providers' access conditions.

| Saved test result | Value |
| --- | --- |
| Structured-slot CNN micro accuracy | 0.9107 |
| Majority baseline micro accuracy | 0.7536 |
| Structured-slot macro balanced accuracy for informative slots | 0.6881 |
| Whole-tumor Dice | 0.8411 |
| Tumor-core Dice | 0.5888 |
| Enhancing-tumor Dice | 0.2730 |

These figures are recorded in `results/thesis_results_summary.txt`. They describe this experiment and dataset split. Performance differs substantially across segmentation targets, and micro accuracy should be interpreted alongside balanced metrics.

Report-generation evaluation separates raw output from final pipeline output. Report-format and unsupported-content diagnostics are not clinical validation, diagnostic accuracy or evidence of zero hallucinations. The project is a research prototype and has not established clinical suitability.

## Navigating the project

| File | Purpose |
| --- | --- |
| `config.yaml` | Dataset paths, model configuration and training settings |
| `requirements.txt` | Python dependencies |
| `prepare_data.py` | Prepare the dataset manifest |
| `train_slot_model.py` | Train the structured-slot model |
| `train_segmentation_model.py` | Train the segmentation model |
| `build_rag_index.py` | Build and check the training-only retrieval index |
| `train_vlm.py` | Train the hybrid vision-language model |
| `evaluate_vlm.py` and `evaluate_vlm_rag.py` | Evaluate report generation |
| `verify_final_results.py` | Check saved experiment results |
| `FINAL_RUN_INSTRUCTIONS.md` | Project execution instructions |

## Reproducing experiments

Read `FINAL_RUN_INSTRUCTIONS.md` and `config.yaml` before running the pipeline. Supply BraTS/TextBraTS data under the configured raw-data directory, install the dependencies in an isolated environment and use the project's final experiment script. The first language-model, retrieval and report-metric run may download model resources.

The saved run used an NVIDIA RTX 4050 laptop GPU with approximately 6 GB of VRAM. Training time and memory requirements vary with hardware and configuration. Smoke-test outputs must be distinguished from full experiment results.

## Author

Jitesh Nagaraj Jalla

- [LinkedIn](https://www.linkedin.com/in/jitesh-nagaraj-jalla-409a0813b/)
- [GitHub](https://github.com/JiteshNani)

## Dataset references

- [BraTS 2020 training-data source linked by this repository](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- [TextBraTS](https://github.com/Jupitern52/TextBraTS)
