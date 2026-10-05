"""Export flat CSV tables for direct transfer into the thesis."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

OUT = Path("results/tables"); OUT.mkdir(parents=True, exist_ok=True)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if Path("results/slot_test_metrics.json").exists():
        d = load("results/slot_test_metrics.json")
        rows = []
        for k, m in d["cnn"]["per_slot"].items():
            rows.append({"slot": k, "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"], "informative": m["informative"], "support": str(m["support"]), "majority_baseline_accuracy": d["majority_baseline"]["per_slot"][k]["accuracy"]})
        pd.DataFrame(rows).to_csv(OUT / "slot_metrics.csv", index=False)
    if Path("results/segmentation_metrics.json").exists():
        d = load("results/segmentation_metrics.json")
        rows = []
        for r, m in d["summary"].items():
            row = {"region": r, **{k: v for k, v in m.items() if not k.endswith("_std") and not k.endswith("_n")}}
            if "hd95" in row:
                row["hd95_mm"] = row.pop("hd95")
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "segmentation_metrics.csv", index=False)
    if Path("results/report_metrics.json").exists():
        d = load("results/report_metrics.json")
        rows = []
        for method, m in d.items():
            row = {"method": method}
            for k, v in m.items():
                if isinstance(v, (int, float, str)) or v is None: row[k] = v
            if isinstance(m.get("rouge"), dict): row["rougeL"] = m["rouge"].get("rougeL")
            if isinstance(m.get("unsupported_content_flag_category_rates"), dict):
                for category, rate in m["unsupported_content_flag_category_rates"].items():
                    row[f"unsupported_{category}_rate"] = rate
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "report_metrics.csv", index=False)
    print("CSV tables saved under results/tables/")


if __name__ == "__main__":
    main()
