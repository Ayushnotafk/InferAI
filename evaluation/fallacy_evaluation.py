"""
Fallacy detector evaluation against a gold annotation set.

Gold format (JSONL):
  {"text": "...", "claim": "...", "premises": "...", "fallacy_type": "Contradiction"|null}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import compute_classification_metrics
from evaluation.publication_figures import plot_categorical_counts
from reasoning.fallacy_detector import detect_fallacies

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = _ROOT / "dataset" / "gold" / "fallacy_gold.jsonl"

# Binary labels for detection quality.
NONE = "None"


def _load_gold(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate_fallacy_detector(
    gold_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute precision/recall and export correct / missed / incorrect examples."""
    path = Path(gold_path) if gold_path else DEFAULT_GOLD
    if not path.is_file():
        return {"error": f"Gold file not found: {path}", "n_samples": 0}

    gold = _load_gold(path)
    y_true: list[str] = []
    y_pred: list[str] = []
    examples: list[dict[str, Any]] = []

    for row in gold:
        text = str(row.get("text", ""))
        claim = str(row.get("claim", ""))
        premises = str(row.get("premises", ""))
        true_type = row.get("fallacy_type") or NONE
        if true_type in ("", "none", "None", None):
            true_type = NONE
        pred = detect_fallacies(text, claim, premises)
        pred_type = pred["fallacy_type"] or NONE
        y_true.append(str(true_type))
        y_pred.append(str(pred_type))

        true_pos = true_type != NONE
        pred_pos = pred_type != NONE
        if true_pos and pred_pos and true_type == pred_type:
            status = "correct"
        elif true_pos and not pred_pos:
            status = "missed"
        elif (not true_pos and pred_pos) or (true_pos and pred_pos and true_type != pred_type):
            status = "incorrect"
        else:
            status = "true_negative"

        examples.append(
            {
                "text": text,
                "claim": claim,
                "premises": premises,
                "y_true": true_type,
                "y_pred": pred_type,
                "status": status,
                "explanation": pred.get("fallacy_explanation"),
            }
        )

    labels = sorted(set(y_true) | set(y_pred))
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

    # Binary detection view.
    bin_true = [0 if t == NONE else 1 for t in y_true]
    bin_pred = [0 if p == NONE else 1 for p in y_pred]
    tp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    status_counts = pd.Series([e["status"] for e in examples]).value_counts().to_dict()
    result: dict[str, Any] = {
        "n_samples": len(gold),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "binary_precision": precision,
        "binary_recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": labels,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        edf = pd.DataFrame(examples)
        edf.to_csv(out / "fallacy_evaluation_examples.csv", index=False)
        pd.DataFrame([result]).to_csv(out / "fallacy_evaluation_metrics.csv", index=False)
        for status in ("correct", "missed", "incorrect"):
            edf[edf["status"] == status].to_csv(
                out / f"fallacy_{status}_examples.csv", index=False
            )
        plot_confusion_matrix(
            y_true,
            y_pred,
            labels,
            str(out / "fallacy_confusion_matrix.png"),
            title="Fallacy detector confusion",
        )
        plot_categorical_counts(
            result["status_counts"],
            out / "fallacy_status_distribution.png",
            title="Fallacy evaluation outcomes",
            xlabel="Status",
        )
        type_counts = edf["y_pred"].value_counts().to_dict()
        plot_categorical_counts(
            {str(k): int(v) for k, v in type_counts.items()},
            out / "fallacy_distribution.png",
            title="Predicted fallacy distribution",
            xlabel="Fallacy type",
        )
    return result
