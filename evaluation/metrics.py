"""Classification and calibration metrics for InferAI benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Return accuracy, precision, recall, macro/weighted F1, and per-class report."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def expected_calibration_error(
    y_true: list[str],
    y_prob: np.ndarray,
    class_names: list[str],
    n_bins: int = 10,
) -> float:
    """
    Multiclass ECE using max-probability binning (one-vs-rest style).

    ``y_prob`` shape: (n_samples, n_classes) aligned with ``class_names``.
    """
    if len(y_true) == 0:
        return 0.0

    probs = np.asarray(y_prob, dtype=np.float64)
    confidences = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred_labels = [class_names[i] for i in pred_idx]
    accuracies = np.array([1.0 if p == t else 0.0 for p, t in zip(pred_labels, y_true)])

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi if i < n_bins - 1 else confidences <= hi)
        if not mask.any():
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
