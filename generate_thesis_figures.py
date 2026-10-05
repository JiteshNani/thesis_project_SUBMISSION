"""Generate thesis-ready figures from the authoritative JSON results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close()


def slot_figures(slot):
    if not slot: return
    keys = ["edema", "enhancing", "necrotic", "hemisphere", "extent"]
    cnn = [slot["cnn"]["per_slot"][k]["accuracy"] for k in keys]
    maj = [slot["majority_baseline"]["per_slot"][k]["accuracy"] for k in keys]
    x = np.arange(len(keys)); w = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w/2, cnn, w, label="3D CNN")
    plt.bar(x + w/2, maj, w, label="Training-majority baseline")
    plt.xticks(x, keys, rotation=20)
    plt.ylim(0, 1.05); plt.ylabel("Accuracy"); plt.legend()
    plt.title("Structured slot prediction: CNN vs majority baseline")
    savefig("slot_accuracy_vs_majority.png")

    for key in keys:
        d = slot["cnn"]["per_slot"][key]
        if not d["informative"]: continue
        cm = np.asarray(d["confusion_matrix"])
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm)
        plt.colorbar(label="Cases")
        labels = [str(x) for x in d["labels"]]
        plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix — {key}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        savefig(f"confusion_{key}.png")


def segmentation_figure(seg):
    if not seg: return
    regions = ["whole_tumor", "tumor_core", "enhancing_tumor", "edema", "necrotic_non_enhancing"]
    dice = [seg["summary"][r]["dice"] for r in regions]
    plt.figure(figsize=(8, 4.5))
    plt.bar(np.arange(len(regions)), dice)
    plt.xticks(np.arange(len(regions)), ["WT", "TC", "ET", "Oedema", "Necrotic"], rotation=15)
    plt.ylim(0, 1.05); plt.ylabel("Mean Dice")
    plt.title("Held-out test segmentation performance")
    savefig("segmentation_mean_dice.png")


def report_figure(rep):
    if not rep: return
    methods = ["template_gt_oracle", "cnn_template", "vlm_base", "vlm_rag"]
    labels = ["GT Template\n(oracle)", "CNN +\nTemplate", "Hybrid\nVLM", "VLM +\nRAG"]
    raw_structure = [rep[m]["raw_generation_two_section_structure_rate"] for m in methods]
    final_structure = [rep[m]["final_pipeline_two_section_structure_rate"] for m in methods]
    gt = [rep[m]["ground_truth_slot_consistency_rate"] for m in methods]
    flags = [rep[m]["rule_based_unsupported_content_flag_rate"] for m in methods]
    x = np.arange(len(methods)); w = 0.20
    plt.figure(figsize=(10, 4.8))
    plt.bar(x - 1.5*w, raw_structure, w, label="Raw two-section structure")
    plt.bar(x - 0.5*w, final_structure, w, label="Final pipeline structure")
    plt.bar(x + 0.5*w, gt, w, label="GT slot consistency")
    plt.bar(x + 1.5*w, flags, w, label="Unsupported-content flag rate")
    plt.xticks(x, labels); plt.ylim(0, 1.05); plt.ylabel("Rate"); plt.legend()
    plt.title("Report-generation reliability diagnostics")
    savefig("report_reliability_comparison.png")


def training_curves():
    slot = load("results/slot_training_history.json")
    if slot:
        plt.figure(figsize=(7, 4.2)); plt.plot([x["epoch"] for x in slot], [x["train_loss"] for x in slot], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("3D CNN slot training loss"); savefig("slot_training_loss.png")
    seg = load("results/segmentation_training_history.json")
    if seg:
        plt.figure(figsize=(7, 4.2)); plt.plot([x["epoch"] for x in seg], [x["train_loss"] for x in seg], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("3D U-Net segmentation training loss"); savefig("segmentation_training_loss.png")
    vlm = load("results/vlm_training_history.json")
    if vlm:
        plt.figure(figsize=(7, 4.2)); plt.plot([x["epoch"] for x in vlm], [x["train_loss"] for x in vlm], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Token-level loss"); plt.title("Hybrid 3D VLM LoRA training loss"); savefig("vlm_training_loss.png")


def main():
    slot_figures(load("results/slot_test_metrics.json"))
    segmentation_figure(load("results/segmentation_metrics.json"))
    report_figure(load("results/report_metrics.json"))
    training_curves()
    print("Figures saved under results/figures/")


if __name__ == "__main__":
    main()
