"""Class-balance-aware evaluation for structured slots."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.constants import SLOT_KEYS


def _to_int(v):
    try:
        return int(v.item())
    except Exception:
        return int(v)


def majority_classes(training_slots: List[Dict[str, int]]) -> Dict[str, int]:
    """Learn one constant majority class per slot from TRAINING targets only."""
    return {
        key: Counter(_to_int(x[key]) for x in training_slots).most_common(1)[0][0]
        for key in SLOT_KEYS
    }


def majority_predictions(classes: Dict[str, int], n_cases: int) -> List[Dict[str, int]]:
    return [{k: int(classes[k]) for k in SLOT_KEYS} for _ in range(n_cases)]


def evaluate_slot_predictions(pred_slots: List[Dict[str, int]], true_slots: List[Dict[str, int]]) -> Dict:
    result: Dict = {"n_cases": len(true_slots), "per_slot": {}}
    total_correct, total = 0, 0
    informative_balanced, informative_f1 = [], []

    for key in SLOT_KEYS:
        y_true = np.asarray([_to_int(x[key]) for x in true_slots], dtype=int)
        y_pred = np.asarray([_to_int(x[key]) for x in pred_slots], dtype=int)
        labels = [0, 1, 2] if key == "extent" else [0, 1]
        observed_true = sorted(set(y_true.tolist()))
        support = {str(v): int((y_true == v).sum()) for v in labels}
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        cm_array = confusion_matrix(y_true, y_pred, labels=labels)
        # Balanced accuracy = mean recall over classes actually present in y_true.
        # Computing it explicitly avoids sklearn warnings when a prediction contains
        # a class absent from a small held-out subset.
        per_true_recalls = []
        for class_value in observed_true:
            row = labels.index(class_value)
            denom = cm_array[row, :].sum()
            per_true_recalls.append(float(cm_array[row, row] / denom) if denom else 0.0)
        if len(observed_true) > 1:
            bal = float(np.mean(per_true_recalls))
            informative_balanced.append(bal)
            informative_f1.append(macro_f1)
        else:
            bal = None
        cm = cm_array.tolist()
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        result["per_slot"][key] = {
            "accuracy": acc,
            "balanced_accuracy": bal,
            "macro_f1": macro_f1,
            "support": support,
            "labels": labels,
            "confusion_matrix": cm,
            "per_class_precision": [float(x) for x in p],
            "per_class_recall": [float(x) for x in r],
            "per_class_f1": [float(x) for x in f],
            "per_class_support": [int(x) for x in s],
            "informative": len(observed_true) > 1,
        }
        total_correct += int((y_true == y_pred).sum())
        total += len(y_true)

    result["micro_accuracy_all_slots"] = float(total_correct / max(total, 1))
    result["macro_balanced_accuracy_informative_slots"] = float(np.mean(informative_balanced)) if informative_balanced else None
    result["macro_f1_informative_slots"] = float(np.mean(informative_f1)) if informative_f1 else None
    return result
