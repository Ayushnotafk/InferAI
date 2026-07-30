"""Confusion matrix helpers for InferAI evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    out_path: str,
    title: str = "Confusion Matrix",
) -> dict[str, Any]:
    """Save a confusion-matrix heatmap and return raw counts."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {"labels": labels, "matrix": cm.tolist(), "path": out_path}
