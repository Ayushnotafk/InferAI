"""
Train the Nyāya-style pramāṇa classifier and persist rich evaluation artifacts.

Artifacts (under ``models/evaluation/``):

* ``metrics.json`` — scalar summaries + optional held-out test metrics
* ``classification_report.txt`` — sklearn text report
* ``confusion_matrix.png`` (and related plots)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings

# ---------------------------------------------------------------------------
# Load training table (``text``, ``label``)
# ---------------------------------------------------------------------------

train_path = Path("dataset/raw/nyaya_dataset_merged.csv")
if not train_path.is_file():
    train_path = Path("dataset/raw/nyaya_dataset.csv")

df = pd.read_csv(train_path)
texts = df["text"].astype(str)
label_col = "pramana_label" if "pramana_label" in df.columns else "label"
labels = df[label_col].astype(str)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X = generate_embeddings(texts.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = float(accuracy_score(y_train, train_pred))
test_acc = float(accuracy_score(y_test, test_pred))

report_txt = classification_report(
    y_test,
    test_pred,
    target_names=label_encoder.classes_,
    digits=4,
)
print("Train accuracy:", round(train_acc, 4))
print("Test accuracy:", round(test_acc, 4))
print(report_txt)

models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)
eval_dir = models_dir / "evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)

bg_size = min(200, len(X_train))
np.save(models_dir / "shap_background.npy", np.asarray(X_train[:bg_size]))

joblib.dump(model, models_dir / "nyaya_model.pkl")
joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion matrix (random 20% holdout)")
plt.tight_layout()
plt.savefig(eval_dir / "confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
counts = df[label_col].value_counts().reindex(label_encoder.classes_).fillna(0)
counts.plot(kind="bar", color="steelblue")
plt.title("Class distribution (training table)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(eval_dir / "class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(5, 4))
plt.bar(["Train accuracy", "Test accuracy"], [train_acc, test_acc], color=["#4c72b0", "#55a868"])
plt.ylim(0, 1.05)
for i, v in enumerate([train_acc, test_acc]):
    plt.text(i, min(v + 0.03, 1.02), f"{v:.3f}", ha="center")
plt.ylabel("Accuracy")
plt.title("Classifier accuracy")
plt.tight_layout()
plt.savefig(eval_dir / "train_test_accuracy.png", dpi=150)
plt.close()

prec, rec, f1, _ = precision_recall_fscore_support(
    y_test,
    test_pred,
    labels=list(range(len(label_encoder.classes_))),
    zero_division=0,
)
x = np.arange(len(label_encoder.classes_))
width = 0.25
plt.figure(figsize=(8, 4))
plt.bar(x - width, prec, width, label="Precision")
plt.bar(x, rec, width, label="Recall")
plt.bar(x + width, f1, width, label="F1")
plt.xticks(x, label_encoder.classes_)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Per-class metrics (holdout)")
plt.legend()
plt.tight_layout()
plt.savefig(eval_dir / "per_class_metrics.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Held-out curated test set (if present)
# ---------------------------------------------------------------------------

test_set_path = Path("dataset/test_set.csv")
held_out: dict | None = None
if test_set_path.is_file():
    tdf = pd.read_csv(test_set_path)
    if {"text", "pramana_label"}.issubset(tdf.columns):
        X_hold = generate_embeddings(tdf["text"].astype(str).tolist())
        y_hold = label_encoder.transform(tdf["pramana_label"].astype(str))
        hold_pred = model.predict(X_hold)
        held_out = {
            "accuracy": float(accuracy_score(y_hold, hold_pred)),
            "classification_report": classification_report(
                y_hold,
                hold_pred,
                target_names=label_encoder.classes_,
                digits=4,
                zero_division=0,
            ),
        }
        hcm = confusion_matrix(y_hold, hold_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            hcm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Confusion matrix (dataset/test_set.csv)")
        plt.tight_layout()
        plt.savefig(eval_dir / "confusion_matrix_test_set.png", dpi=150)
        plt.close()

# ---------------------------------------------------------------------------
# Persist metrics + report
# ---------------------------------------------------------------------------

prec_macro = float(np.mean(prec))
rec_macro = float(np.mean(rec))
f1_macro = float(np.mean(f1))

metrics = {
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "macro_precision": prec_macro,
    "macro_recall": rec_macro,
    "macro_f1": f1_macro,
    "per_class": {
        label_encoder.classes_[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        }
        for i in range(len(label_encoder.classes_))
    },
    "held_out_test_set": held_out,
}

(eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(eval_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

print(f"SHAP background saved ({bg_size} rows)")
print(f"Evaluation artifacts saved under {eval_dir}")
print("Model saved successfully")
