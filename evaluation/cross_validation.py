"""
5-fold stratified cross-validation for the InferAI ML head.

Embeddings are computed once (fixed Sentence-BERT features); logistic
regression is re-fit per fold. This matches the production architecture
without re-encoding on every fold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from classification.embedder import generate_embeddings

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = _ROOT / "dataset" / "raw" / "nyaya_dataset_merged.csv"


def run_cross_validation(
    train_csv: str | Path | None = None,
    *,
    n_folds: int = 5,
    random_state: int = 42,
    out_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Stratified k-fold CV on logistic regression over SBERT embeddings.

    Returns mean/std accuracy and macro F1 across folds.
    """
    path = Path(train_csv) if train_csv else DEFAULT_TRAIN
    if not path.is_file():
        # Fallback to master seed set.
        path = _ROOT / "dataset" / "master_dataset.csv"
    df = pd.read_csv(path)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    if max_samples is not None:
        texts, labels = texts[:max_samples], labels[:max_samples]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = generate_embeddings(texts)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, Any]] = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true = y[te]
        # Inverse for macro F1 with string labels for consistency.
        yt = le.inverse_transform(y_true)
        yp = le.inverse_transform(pred)
        acc = float(accuracy_score(y_true, pred))
        mf1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        fold_rows.append({"fold": fold, "n_test": int(len(te)), "accuracy": acc, "macro_f1": mf1})

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "n_folds": n_folds,
        "n_samples": len(texts),
        "train_csv": str(path),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std(ddof=1)),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "std_macro_f1": float(fold_df["macro_f1"].std(ddof=1)),
        "folds": fold_rows,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(out / "cross_validation_results.csv", index=False)
        pd.DataFrame(
            [
                {
                    "mean_accuracy": summary["mean_accuracy"],
                    "std_accuracy": summary["std_accuracy"],
                    "mean_macro_f1": summary["mean_macro_f1"],
                    "std_macro_f1": summary["std_macro_f1"],
                    "n_folds": n_folds,
                    "n_samples": summary["n_samples"],
                }
            ]
        ).to_csv(out / "cross_validation_summary.csv", index=False)
        (out / "cross_validation_report.md").write_text(
            _cv_markdown(summary), encoding="utf-8"
        )
    return summary


def _cv_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 5-Fold Cross-Validation Report",
        "",
        f"Corpus: `{summary['train_csv']}` (n={summary['n_samples']}).",
        "Protocol: StratifiedKFold on Sentence-BERT embeddings; LogisticRegression re-fit per fold.",
        "",
        f"- Mean accuracy: **{summary['mean_accuracy']:.4f}** ± {summary['std_accuracy']:.4f}",
        f"- Mean macro F1: **{summary['mean_macro_f1']:.4f}** ± {summary['std_macro_f1']:.4f}",
        "",
        "| Fold | n | Accuracy | Macro F1 |",
        "|------|---|----------|----------|",
    ]
    for row in summary["folds"]:
        lines.append(
            f"| {row['fold']} | {row['n_test']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
