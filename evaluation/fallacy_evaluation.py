"""Evaluate the rule-based fallacy detector against validated gold annotations.

Outputs (when validated annotations exist):
 - results/fallacy/fallacy_metrics.json
 - results/fallacy/fallacy_metrics.csv
 - results/fallacy/fallacy_confusion_matrix.png

If no validated entries are present, report that evaluation is pending.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from reasoning.fallacy_detector import detect_fallacies

DATA_PATH = Path("dataset/fallacy/fallacy_gold.jsonl")
OUT_DIR = Path("results/fallacy")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gold() -> List[dict]:
    rows = []
    with DATA_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main():
    gold = load_gold()
    validated = [r for r in gold if r.get("annotation_status") == "validated"]
    if not validated:
        print("No validated gold annotations found. Evaluation pending human validation.")
        return

    y_true = [r["fallacy_label"] for r in validated]
    y_pred = []
    for r in validated:
        out = detect_fallacies(r["text"], r.get("claim", ""), r.get("premises", ""))
        lab = out.get("fallacy_type") or "none"
        # normalize to lowercase keys used in dataset
        lab_norm = lab.lower().replace(" ", "_") if lab else "none"
        # map known display names to dataset labels
        mapping = {
            "contradiction": "contradiction",
            "unstated premise": "unstated_premise",
            "unstated_premise": "unstated_premise",
            "weak inference": "weak_inference",
            "weak_inference": "weak_inference",
            "weak analogy": "weak_analogy",
            "weak_analogy": "weak_analogy",
            "weak authority": "weak_authority",
            "none": "none",
        }
        y_pred.append(mapping.get(lab_norm, "none"))

    labels = sorted(list(set(y_true) | set(y_pred)))
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "labels": labels,
        "per_class": {labels[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])} for i in range(len(labels))},
    }

    (OUT_DIR / "fallacy_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # CSV per-class
    with (OUT_DIR / "fallacy_metrics.csv").open("w", encoding="utf-8") as fh:
        fh.write("label,precision,recall,f1\n")
        for lab in labels:
            pc = metrics["per_class"][lab]
            fh.write(f"{lab},{pc['precision']:.4f},{pc['recall']:.4f},{pc['f1']:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fallacy_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("Fallacy evaluation outputs written to results/fallacy/")


if __name__ == "__main__":
    main()

