"""
Stratified 5-fold cross-validation on merged AAEC + IBM manually reviewed corpora.

Uses frozen Sentence-BERT embeddings and the same LogisticRegression config as
``classification/train_model.py``. Does **not** overwrite production model artifacts.

Outputs:
  - reports/real_world_cross_validation.md
  - reports/figures/real_world_cv_confusion_matrix.png
  - reports/csv/real_world_cv_fold_metrics.csv
  - reports/csv/real_world_cv_per_class_metrics.csv

Usage (from project root):
    python reports/generate_real_world_cross_validation.py
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
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings
from reports._common import apply_plot_style, dataframe_to_markdown, ensure_reports_dir, save_dataframe_csv

AAEC_CSV = _ROOT / "dataset" / "processed" / "aaec_reviewed_dataset.csv"
IBM_CSV = _ROOT / "dataset" / "processed" / "ibm_reviewed_dataset.csv"
TEST_SET = _ROOT / "dataset" / "test_set.csv"
MODEL_PATH = _ROOT / "models" / "nyaya_model.pkl"
LE_PATH = _ROOT / "models" / "label_encoder.pkl"
ORIGINAL_SNAPSHOT = _ROOT / "reports" / "model_eval_snapshots" / "original_test_set.json"

OUT_MD = _ROOT / "reports" / "real_world_cross_validation.md"
OUT_FIG = _ROOT / "reports" / "figures" / "real_world_cv_confusion_matrix.png"
OUT_FOLD_CSV = _ROOT / "reports" / "csv" / "real_world_cv_fold_metrics.csv"
OUT_CLASS_CSV = _ROOT / "reports" / "csv" / "real_world_cv_per_class_metrics.csv"

N_SPLITS = 5
RANDOM_STATE = 42


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def load_real_world_corpus() -> pd.DataFrame:
    aaec = pd.read_csv(AAEC_CSV)
    ibm = pd.read_csv(IBM_CSV)
    aaec["source"] = "AAEC"
    ibm["source"] = "IBM"
    merged = pd.concat([aaec, ibm], ignore_index=True)
    merged["text"] = merged["text"].astype(str).map(_normalize_text)
    merged = merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return merged


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_production_on_test_set(le: LabelEncoder) -> dict:
    model = joblib.load(MODEL_PATH)
    tdf = pd.read_csv(TEST_SET)
    texts = tdf["text"].astype(str).tolist()
    y_true = le.transform(tdf["pramana_label"].astype(str))
    X = generate_embeddings(texts)
    y_pred = model.predict(X)
    m = macro_metrics(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(le.classes_))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(le.classes_))))
    return {
        **m,
        "classes": list(le.classes_),
        "per_class": {
            le.classes_[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(le.classes_))
        },
        "confusion_matrix": cm.tolist(),
        "n": len(tdf),
    }


def run_cross_validation(df: pd.DataFrame) -> dict:
    texts = df["text"].astype(str).tolist()
    le = LabelEncoder()
    le.fit(["Anumana", "Pratyaksha", "Shabda", "Upamana"])
    y = le.transform(df["pramana_label"].astype(str))

    print(f"Embedding {len(texts)} real-world texts...")
    X = generate_embeddings(texts)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict] = []
    oof_pred = np.full(len(y), -1, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        val_pred = clf.predict(X[val_idx])
        oof_pred[val_idx] = val_pred

        m = macro_metrics(y[val_idx], val_pred)
        fold_rows.append({"fold": fold_idx, "n_val": len(val_idx), **m})
        print(
            f"  Fold {fold_idx}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"(n_val={len(val_idx)})"
        )

    fold_df = pd.DataFrame(fold_rows)
    pooled = macro_metrics(y, oof_pred)

    std_row = {
        "fold": "std",
        "n_val": "",
        "accuracy": float(fold_df["accuracy"].std(ddof=1)),
        "macro_precision": float(fold_df["macro_precision"].std(ddof=1)),
        "macro_recall": float(fold_df["macro_recall"].std(ddof=1)),
        "macro_f1": float(fold_df["macro_f1"].std(ddof=1)),
    }
    mean_row = {
        "fold": "mean",
        "n_val": "",
        "accuracy": float(fold_df["accuracy"].mean()),
        "macro_precision": float(fold_df["macro_precision"].mean()),
        "macro_recall": float(fold_df["macro_recall"].mean()),
        "macro_f1": float(fold_df["macro_f1"].mean()),
    }

    prec, rec, f1, support = precision_recall_fscore_support(
        y, oof_pred, labels=list(range(len(le.classes_))), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "Class": list(le.classes_),
            "Precision": np.round(prec, 4),
            "Recall": np.round(rec, 4),
            "F1": np.round(f1, 4),
            "Support": support.astype(int),
        }
    )

    cm = confusion_matrix(y, oof_pred, labels=list(range(len(le.classes_))))

    return {
        "le": le,
        "fold_df": fold_df,
        "mean_row": mean_row,
        "std_row": std_row,
        "pooled": pooled,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": list(le.classes_),
        "n_samples": len(df),
        "class_counts": df["pramana_label"].value_counts().to_dict(),
        "source_counts": df["source"].value_counts().to_dict(),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    apply_plot_style()
    fig_dir = out_path.parent
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def load_original_snapshot() -> dict | None:
    if not ORIGINAL_SNAPSHOT.is_file():
        return None
    return json.loads(ORIGINAL_SNAPSHOT.read_text(encoding="utf-8"))


def write_report(
    cv: dict,
    template_eval: dict,
    original_snapshot: dict | None,
    df: pd.DataFrame,
) -> None:
    fold_display = pd.concat(
        [cv["fold_df"], pd.DataFrame([cv["mean_row"], cv["std_row"]])],
        ignore_index=True,
    )

    lines: list[str] = []
    lines.append("# Real-World Cross-Validation Evaluation\n\n")
    lines.append(
        "Stratified **5-fold cross-validation** on the merged manually reviewed corpora "
        "(`aaec_reviewed_dataset.csv` + `ibm_reviewed_dataset.csv`). "
        "Generated by `reports/generate_real_world_cross_validation.py`.\n\n"
    )
    lines.append(
        "**Evaluation-only:** production artifacts under `models/` are **not** modified. "
        "Each fold trains a fresh Logistic Regression head on Sentence-BERT embeddings "
        "(same settings as `classification/train_model.py`: `max_iter=1000`).\n\n"
    )
    lines.append("---\n\n")

    lines.append("## 1. Dataset\n\n")
    lines.append(f"| Item | Value |\n|------|------:|\n")
    lines.append(f"| AAEC reviewed rows | {cv['source_counts'].get('AAEC', 0)} |\n")
    lines.append(f"| IBM reviewed rows | {cv['source_counts'].get('IBM', 0)} |\n")
    lines.append(f"| Merged (deduped on text) | **{cv['n_samples']}** |\n")
    lines.append(f"| CV folds | {N_SPLITS} (stratified, `random_state={RANDOM_STATE}`) |\n\n")

    lines.append("### Class distribution (merged real-world corpus)\n\n")
    counts = df["pramana_label"].value_counts().reindex(cv["class_names"]).fillna(0).astype(int)
    lines.append("| Pramāṇa | Count | % |\n|--------|------:|--:|\n")
    for cls in cv["class_names"]:
        c = int(counts.get(cls, 0))
        pct = 100.0 * c / cv["n_samples"]
        lines.append(f"| {cls} | {c} | {pct:.1f}% |\n")
    lines.append("\n")

    lines.append("## 2. Cross-validation results (out-of-fold)\n\n")
    lines.append("### Per-fold metrics\n\n")
    lines.append(dataframe_to_markdown(fold_display.round(4)) + "\n\n")

    lines.append("### Pooled out-of-fold aggregate\n\n")
    p = cv["pooled"]
    lines.append("| Metric | Mean (fold) | Std (fold) | Pooled OOF |\n")
    lines.append("|--------|------------:|-----------:|-----------:|\n")
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    ]:
        lines.append(
            f"| {label} | {cv['mean_row'][key]:.4f} | {cv['std_row'][key]:.4f} | {p[key]:.4f} |\n"
        )
    lines.append(
        "\n*Pooled OOF* recomputes metrics on all out-of-fold predictions concatenated "
        "(slightly different from the fold mean when class support per fold is small).\n\n"
    )

    lines.append("### Per-class metrics (pooled OOF)\n\n")
    lines.append(dataframe_to_markdown(cv["per_class"]) + "\n\n")

    lines.append("### Confusion matrix (pooled OOF)\n\n")
    lines.append("Rows = true label, columns = predicted label.\n\n")
    header = "| True \\\\ Pred | " + " | ".join(cv["class_names"]) + " |\n"
    sep = "|" + "|".join(["---"] * (len(cv["class_names"]) + 1)) + "|\n"
    lines.append(header + sep)
    for i, cls in enumerate(cv["class_names"]):
        row = " | ".join(str(int(x)) for x in cv["confusion_matrix"][i])
        lines.append(f"| **{cls}** | {row} |\n")
    lines.append("\n")
    lines.append("![Real-world CV confusion matrix](figures/real_world_cv_confusion_matrix.png)\n\n")

    lines.append("---\n\n")
    lines.append("## 3. Template held-out test (`dataset/test_set.csv`)\n\n")
    lines.append(
        "Evaluation of the **saved production model** (`models/nyaya_model.pkl`) on the "
        "synthetic template test set (n=200), for comparison.\n\n"
    )
    lines.append("| Metric | Production model on test_set.csv |\n|--------|--------------------------------:|\n")
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    ]:
        lines.append(f"| {label} | **{template_eval[key]:.4f}** |\n")
    lines.append("\n")

    if original_snapshot:
        lines.append(
            "Reference snapshot (`reports/model_eval_snapshots/original_test_set.json`, "
            "original merged-training model): "
            f"accuracy **{original_snapshot['accuracy']:.3f}**, "
            f"macro F1 **{original_snapshot['macro_f1']:.3f}**.\n\n"
        )

    lines.append("### Template test confusion matrix (production model)\n\n")
    tcm = np.array(template_eval["confusion_matrix"])
    lines.append(header + sep)
    for i, cls in enumerate(template_eval["classes"]):
        row = " | ".join(str(int(x)) for x in tcm[i])
        lines.append(f"| **{cls}** | {row} |\n")
    lines.append("\n")

    lines.append("---\n\n")
    lines.append("## 4. Comparison: template test vs real-world CV\n\n")
    lines.append("| Metric | Template test (prod model) | Real-world CV (pooled OOF) | Δ (CV − template) |\n")
    lines.append("|--------|---------------------------:|---------------------------:|------------------:|\n")
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    ]:
        t_val = template_eval[key]
        cv_val = p[key]
        lines.append(f"| {label} | {t_val:.4f} | {cv_val:.4f} | {cv_val - t_val:+.4f} |\n")
    lines.append(
        "\n**Reading the comparison:** Real-world CV shows **higher accuracy** but **lower macro F1** "
        "than the template test. Accuracy is inflated on the imbalanced real-world corpus because fold "
        "models predict **Anumana** for most sentences. Macro F1 is the fairer headline metric across "
        "pramāṇas; on that measure, both evaluations are moderate, with the template test appearing "
        "stronger partly due to balanced classes and explicit cue templates.\n\n"
    )

    lines.append("## 5. Why the two evaluations differ\n\n")
    lines.append(
        "### Different data provenance\n\n"
        "- **Template test** (`test_set.csv`) is **synthetically generated** from English templates "
        "with domain parentheticals (`Bench notes (science): …`). Roughly **56%** of rows are generic "
        "padding vignettes (`test augmentation`). Classes are **deliberately balanced** (~22–29% each).\n"
        "- **Real-world CV** uses **850 manually Nyāya-annotated** argumentative sentences from "
        "**AAEC essays** and **IBM ArgKP claims**. Text is natural, messy, and **heavily skewed** "
        "toward Anumana (~90% of merged real-world rows).\n\n"
    )
    lines.append(
        "### Different evaluation protocols\n\n"
        "- **Template test:** single held-out pass with the **fixed production model** trained on "
        "synthetic + AAEC + IBM merged corpus (1,461 rows after dedup).\n"
        "- **Real-world CV:** **5-fold stratified** training within the real-world pool only; each "
        "sentence is predicted by a fold model that did **not** see it during that fold's training. "
        "No synthetic nyaya rows participate.\n\n"
    )
    lines.append(
        "### Distribution shift (OOD template test)\n\n"
        "As documented in `reports/test_set_distribution_analysis.md`, the template test is "
        "**out-of-distribution** relative to AAEC/IBM training text (Jensen–Shannon divergence ≈ 0.51, "
        "class TV distance ≈ 0.58). Template accuracy can **overstate** or **understate** real-world "
        "performance depending on whether cue patterns in templates match model biases.\n\n"
    )
    lines.append(
        "### Class imbalance effects\n\n"
        f"- Real-world merged corpus: **~{100 * counts.get('Anumana', 0) / cv['n_samples']:.0f}% Anumana** "
        f"vs template test **~22% Anumana**.\n"
        "- Macro metrics on real-world data penalize minority classes (Pratyaksha, Upamana, Shabda) "
        "more severely; high template accuracy partly reflects balanced classes and repetitive cue templates.\n\n"
    )
    lines.append(
        "### Practical takeaway\n\n"
        "**Real-world cross-validation is the more trustworthy estimate** of performance on manually "
        "reviewed argumentative English. The template test remains useful as a **leakage-free regression "
        "check** on synthetic cue patterns, but should not be treated as i.i.d. with deployment data.\n"
    )

    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ensure_reports_dir()

    for p in (AAEC_CSV, IBM_CSV, TEST_SET, MODEL_PATH, LE_PATH):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = load_real_world_corpus()
    print(f"Merged real-world corpus: {len(df)} rows")
    print(df["pramana_label"].value_counts())

    cv = run_cross_validation(df)

    save_dataframe_csv(cv["fold_df"], OUT_FOLD_CSV)
    save_dataframe_csv(cv["per_class"], OUT_CLASS_CSV)

    plot_confusion_matrix(
        cv["confusion_matrix"],
        cv["class_names"],
        OUT_FIG,
        "Real-world 5-fold CV — pooled confusion matrix (OOF)",
    )

    le_prod = joblib.load(LE_PATH)
    print("Evaluating production model on template test_set.csv...")
    template_eval = evaluate_production_on_test_set(le_prod)

    original_snapshot = load_original_snapshot()
    write_report(cv, template_eval, original_snapshot, df)

    print(f"\nWrote {OUT_MD}")
    print(
        f"Pooled OOF: acc={cv['pooled']['accuracy']:.4f} "
        f"macro_f1={cv['pooled']['macro_f1']:.4f}"
    )
    print(
        f"Template test (prod): acc={template_eval['accuracy']:.4f} "
        f"macro_f1={template_eval['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
