"""Evaluate retrained model on dataset/test_set.csv and return full metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings

PRAMANA_LABELS = ("Pratyaksha", "Anumana", "Upamana", "Shabda")


def evaluate_test_set() -> dict:
    model = joblib.load(_ROOT / "models" / "nyaya_model.pkl")
    le = joblib.load(_ROOT / "models" / "label_encoder.pkl")
    tdf = pd.read_csv(_ROOT / "dataset" / "test_set.csv")

    texts = tdf["text"].astype(str).tolist()
    y_true = le.transform(tdf["pramana_label"].astype(str))
    X = generate_embeddings(texts)
    y_pred = model.predict(X)

    classes = list(le.classes_)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classes": classes,
        "per_class": {
            classes[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, digits=4, zero_division=0
        ),
    }
