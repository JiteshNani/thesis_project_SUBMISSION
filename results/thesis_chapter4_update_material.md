# Chapter 4 — Results update material

## Dataset and split
The authoritative run contained **368 complete paired cases**, split deterministically using seed 42 into 257 training, 55 validation and 56 held-out test cases. The extent thresholds were calculated only from the training split: [66715.33333333333, 122864.0] original-mask voxels.

## Slot CNN
The held-out slot CNN micro-accuracy across all five clinical slots was **0.9107**. The training-split majority baseline was **0.7536**, giving a CNN-minus-baseline difference of **0.1571**.

Per-slot results must be copied from `results/tables/slot_metrics.csv`; do not replace them with previous thesis values.

## Segmentation
Held-out segmentation metrics must be copied from `results/tables/segmentation_metrics.csv`. Dice, IoU, precision and recall are computed on the canonical original held-out BraTS grid. HD95 is reported in millimetres using the original NIfTI physical voxel spacing and MONAI's symmetric 95th-percentile Hausdorff implementation.

## Report generation
The comparison consists of four methods:
1. Ground-truth-slot constrained template (oracle upper bound)
2. CNN-predicted-slot constrained template
3. Hybrid structured-and-visual VLM
4. Hybrid structured-and-visual VLM + train-only RAG

For report-generation results, use `results/tables/report_metrics.csv`. For VLM methods, distinguish **raw generator metrics** from **final slot-grounded pipeline metrics**. The final grounding layer uses only deployment-available CNN-predicted slots and is therefore a reliability guardrail, not evidence of clinical correctness. The unsupported-content diagnostic is a **conservative rule-based proxy**, not a clinical hallucination rate.

## VLM checkpoint selection
The VLM was selected by validation loss using the training/validation split only. Selected epoch: **8**; best validation loss: **0.014045993200587956**.

## Reproducibility
Device: **cuda**; GPU: **NVIDIA GeForce RTX 4050 Laptop GPU**; CUDA build: **12.6**.
