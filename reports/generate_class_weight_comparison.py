"""Generate reports/class_weight_comparison.md after balanced retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings

PREV_PATH = _ROOT / "reports" / "class_weight_previous_metrics.json"
REPORT_OUT = _ROOT / "reports" / "class_weight_comparison.md"
FOCUS = ("Shabda", "Upamana", "Pratyaksha")


def eval_test_set() -> dict:
    model = joblib.load(_ROOT / "models" / "nyaya_model.pkl")
    le = joblib.load(_ROOT / "models" / "label_encoder.pkl")
    tdf = pd.read_csv(_ROOT / "dataset" / "test_set.csv")
    X = generate_embeddings(tdf["text"].astype(str).tolist())
    y = le.transform(tdf["pramana_label"].astype(str))
    pred = model.predict(X)
    _, rec, _, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(len(le.classes_))), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall": {le.classes_[i]: float(rec[i]) for i in range(len(le.classes_))},
    }


def main() -> None:
    prev = json.loads(PREV_PATH.read_text(encoding="utf-8"))
    new = eval_test_set()

    lines: list[str] = []
    lines.append('# Class Weight Comparison (`class_weight="balanced"`)\n\n')
    lines.append(
        "Comparison on held-out `dataset/test_set.csv` (n=200) before and after adding "
        '`class_weight="balanced"` to LogisticRegression in `classification/retrain_merged.py`.\n\n'
    )

    lines.append("## Aggregate metrics\n\n")
    lines.append("| Metric | Previous | New | Change |\n")
    lines.append("| --- | ---: | ---: | ---: |\n")
    lines.append(
        f"| Accuracy | {prev['accuracy']:.4f} | {new['accuracy']:.4f} | "
        f"{new['accuracy'] - prev['accuracy']:+.4f} |\n"
    )
    lines.append(
        f"| Macro F1 | {prev['macro_f1']:.4f} | {new['macro_f1']:.4f} | "
        f"{new['macro_f1'] - prev['macro_f1']:+.4f} |\n\n"
    )

    lines.append("## Per-class recall\n\n")
    lines.append("| Class | Previous recall | New recall | Change | Improved? |\n")
    lines.append("| --- | ---: | ---: | ---: | --- |\n")
    for cls in prev["recall"]:
        p = prev["recall"][cls]
        n = new["recall"][cls]
        delta = n - p
        if delta > 0:
            improved = "**Yes**" if cls in FOCUS else "Yes"
        elif delta < 0:
            improved = "**No**" if cls in FOCUS else "No"
        else:
            improved = "Unchanged"
        lines.append(f"| {cls} | {p:.4f} | {n:.4f} | {delta:+.4f} | {improved} |\n")
    lines.append("\n")

    lines.append("## Focus classes (Shabda, Upamana, Pratyaksha)\n\n")
    for cls in FOCUS:
        p = prev["recall"][cls]
        n = new["recall"][cls]
        delta = n - p
        if delta > 0:
            verdict = f"**Improved** by {delta:.4f} ({100 * delta:.1f} pp)"
        elif delta < 0:
            verdict = f"**Decreased** by {abs(delta):.4f} ({100 * abs(delta):.1f} pp)"
        else:
            verdict = "Unchanged"
        lines.append(f"- **{cls}:** {p:.4f} → {n:.4f} — {verdict}\n")
    lines.append("\n")

    lines.append("## Interpretation\n\n")
    acc_delta = new["accuracy"] - prev["accuracy"]
    f1_delta = new["macro_f1"] - prev["macro_f1"]
    lines.append(
        f"Balanced class weights {'increased' if acc_delta >= 0 else 'decreased'} held-out accuracy "
        f"({prev['accuracy']:.1%} → {new['accuracy']:.1%}) and "
        f"{'increased' if f1_delta >= 0 else 'decreased'} macro F1 "
        f"({prev['macro_f1']:.3f} → {new['macro_f1']:.3f}). "
    )
    improved = [c for c in FOCUS if new["recall"][c] > prev["recall"][c]]
    declined = [c for c in FOCUS if new["recall"][c] < prev["recall"][c]]
    if improved:
        lines.append("Recall improved for: **" + ", ".join(improved) + "**. ")
    if declined:
        lines.append("Recall declined for: **" + ", ".join(declined) + "**. ")
    lines.append(
        "Re-weighting penalizes the dominant Anumana class (~80% of merged training data), "
        "shifting the decision boundary toward minority pramana labels.\n"
    )

    REPORT_OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
