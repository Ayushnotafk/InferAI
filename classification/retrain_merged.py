"""
Complete retraining pipeline: merge synthetic + AAEC + IBM reviewed corpora,
train Sentence-BERT + Logistic Regression, overwrite model artifacts, and
write ``reports/final_retraining_report.md``.

Usage (from project root):
    python classification/retrain_merged.py

Does not modify API or frontend code. Artifacts are saved to the same paths
consumed by ``classification/predictor.py`` and ``api/app.py``.
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
# Paths
# ---------------------------------------------------------------------------

SYNTHETIC_CANDIDATES = (
    _ROOT / "dataset" / "nyaya_dataset.csv",
    _ROOT / "dataset" / "raw" / "nyaya_dataset.csv",
)
AAEC_CSV = _ROOT / "dataset" / "processed" / "aaec_reviewed_dataset.csv"
IBM_CSV = _ROOT / "dataset" / "processed" / "ibm_reviewed_dataset.csv"
MERGED_CSV = _ROOT / "dataset" / "processed" / "merged_retraining_corpus.csv"
REPORT_OUT = _ROOT / "reports" / "final_retraining_report.md"

MODELS_DIR = _ROOT / "models"
EVAL_DIR = MODELS_DIR / "evaluation"

PRAMANA_LABELS = ("Pratyaksha", "Anumana", "Upamana", "Shabda")
STRENGTH_LABELS = ("weak", "moderate", "strong")
RANDOM_STATE = 42
MINORITY_OVERSAMPLE_TARGET = 500
MINORITY_CLASSES = ("Pratyaksha", "Upamana", "Shabda")
OVERSAMPLED_CSV = _ROOT / "dataset" / "processed" / "merged_retraining_corpus_oversampled.csv"


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _normalize_pramana(value: str) -> str:
    label = str(value).strip()
    aliases = {
        "pratyaksha": "Pratyaksha",
        "anumana": "Anumana",
        "upamana": "Upamana",
        "shabda": "Shabda",
    }
    if label in PRAMANA_LABELS:
        return label
    key = label.lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown pramana label: {value!r}")


def _normalize_strength(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    s = str(value).strip().lower()
    if s not in STRENGTH_LABELS:
        return s
    return s


def _resolve_synthetic_path() -> Path:
    for path in SYNTHETIC_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Synthetic dataset not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in SYNTHETIC_CANDIDATES)
    )


def _load_standardized(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"{path}: missing 'text' column")

    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"{path}: missing label column")

    out = pd.DataFrame()
    out["text"] = df["text"].astype(str).map(_normalize_text)
    out["pramana_label"] = df[label_col].astype(str).map(_normalize_pramana)
    out["strength_label"] = (
        df["strength_label"].map(_normalize_strength)
        if "strength_label" in df.columns
        else ""
    )
    out["source"] = source
    return out


def merge_datasets() -> tuple[pd.DataFrame, dict]:
    synthetic_path = _resolve_synthetic_path()
    for path in (AAEC_CSV, IBM_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required dataset: {path}")

    parts = [
        _load_standardized(synthetic_path, "nyaya_synthetic"),
        _load_standardized(AAEC_CSV, "aaec_reviewed"),
        _load_standardized(IBM_CSV, "ibm_reviewed"),
    ]
    combined = pd.concat(parts, ignore_index=True)
    rows_before_dedup = len(combined)

    combined = combined[combined["text"].str.len() > 0].copy()
    combined = combined.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicates_removed = rows_before_dedup - len(combined)

    combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    stats = {
        "synthetic_path": str(synthetic_path.relative_to(_ROOT)),
        "synthetic_rows": int((combined["source"] == "nyaya_synthetic").sum()),
        "aaec_rows": int((combined["source"] == "aaec_reviewed").sum()),
        "ibm_rows": int((combined["source"] == "ibm_reviewed").sum()),
        "rows_before_dedup": rows_before_dedup,
        "duplicates_removed": duplicates_removed,
        "total_rows": len(combined),
    }
    return combined, stats


def oversample_minority_classes(df: pd.DataFrame, target: int = MINORITY_OVERSAMPLE_TARGET) -> tuple[pd.DataFrame, dict]:
    """
    Keep all Anumana rows; upsample Pratyaksha, Upamana, Shabda to ``target``
    via random sampling with replacement, then shuffle.
    """
    counts_before = (
        df["pramana_label"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).astype(int).to_dict()
    )

    parts: list[pd.DataFrame] = [df[df["pramana_label"] == "Anumana"].copy()]

    for label in MINORITY_CLASSES:
        subset = df[df["pramana_label"] == label]
        if subset.empty:
            raise ValueError(f"No training rows found for minority class {label}")
        if len(subset) >= target:
            parts.append(subset.copy())
        else:
            parts.append(
                subset.sample(n=target, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)
            )

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    counts_after = (
        out["pramana_label"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).astype(int).to_dict()
    )
    return out, {"before": counts_before, "after": counts_after, "target": target}


def print_oversample_stats(oversample_stats: dict) -> None:
    print("\nMinority oversampling (Anumana unchanged):")
    print(f"  Target per minority class: {oversample_stats['target']}")
    print("  Class counts BEFORE oversampling:")
    for label in PRAMANA_LABELS:
        print(f"    {label:12s} {oversample_stats['before'][label]}")
    print("  Class counts AFTER oversampling:")
    for label in PRAMANA_LABELS:
        print(f"    {label:12s} {oversample_stats['after'][label]}")


def print_merge_stats(df: pd.DataFrame, stats: dict) -> None:
    print("=" * 72)
    print("Merged retraining corpus")
    print("=" * 72)
    print(f"Synthetic source: {stats['synthetic_path']}")
    print(f"Total rows (after dedup + shuffle): {stats['total_rows']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Synthetic retained: {stats['synthetic_rows']}")
    print(f"  AAEC reviewed:      {stats['aaec_rows']}")
    print(f"  IBM reviewed:       {stats['ibm_rows']}")
    print("\nClass distribution (pramana_label):")
    print(df["pramana_label"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).to_string())
    print("\nStrength distribution (labeled rows only):")
    strength = df[df["strength_label"].isin(STRENGTH_LABELS)]["strength_label"]
    if len(strength):
        print(strength.value_counts().reindex(STRENGTH_LABELS, fill_value=0).to_string())
    else:
        print("  (no strength labels)")


def train_and_evaluate(df: pd.DataFrame) -> dict:
    texts = df["text"].astype(str).tolist()
    labels = df["pramana_label"].astype(str).tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("\nGenerating Sentence-BERT embeddings...")
    X = generate_embeddings(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, train_pred))
    test_acc = float(accuracy_score(y_test, test_pred))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, labels=list(range(len(label_encoder.classes_))), zero_division=0
    )
    macro_precision = float(np.mean(prec))
    macro_recall = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))

    report_txt = classification_report(
        y_test,
        test_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, test_pred)

    # Persist model artifacts (same paths as train_model.py / predictor.py)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    bg_size = min(200, len(X_train))
    np.save(MODELS_DIR / "shap_background.npy", np.asarray(X_train[:bg_size]))
    joblib.dump(model, MODELS_DIR / "nyaya_model.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    # Plots
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
    plt.title("Confusion matrix (20% holdout, merged retraining)")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    counts = df["pramana_label"].value_counts().reindex(label_encoder.classes_).fillna(0)
    counts.plot(kind="bar", color="steelblue")
    plt.title("Class distribution (merged training corpus)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["Train accuracy", "Test accuracy"], [train_acc, test_acc], color=["#4c72b0", "#55a868"])
    plt.ylim(0, 1.05)
    for i, v in enumerate([train_acc, test_acc]):
        plt.text(i, min(v + 0.03, 1.02), f"{v:.3f}", ha="center")
    plt.ylabel("Accuracy")
    plt.title("Classifier accuracy (merged retraining)")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "train_test_accuracy.png", dpi=150)
    plt.close()

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
    plt.savefig(EVAL_DIR / "per_class_metrics.png", dpi=150)
    plt.close()

    # Optional held-out curated test set
    held_out = None
    test_set_path = _ROOT / "dataset" / "test_set.csv"
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
            plt.savefig(EVAL_DIR / "confusion_matrix_test_set.png", dpi=150)
            plt.close()

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            label_encoder.classes_[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
            }
            for i in range(len(label_encoder.classes_))
        },
        "held_out_test_set": held_out,
        "training_sources": {
            "merged_corpus": str(MERGED_CSV.relative_to(_ROOT)),
            "random_state": RANDOM_STATE,
        },
    }
    (EVAL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (EVAL_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "classification_report": report_txt,
        "confusion_matrix": cm,
        "classes": list(label_encoder.classes_),
        "metrics": metrics,
        "shap_background_size": bg_size,
    }


def verify_api_model_paths() -> dict:
    """Confirm predictor artifact paths exist and load without API changes."""
    model_path = MODELS_DIR / "nyaya_model.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"
    shap_bg_path = MODELS_DIR / "shap_background.npy"

    checks = {
        "nyaya_model.pkl exists": model_path.is_file(),
        "label_encoder.pkl exists": encoder_path.is_file(),
        "shap_background.npy exists": shap_bg_path.is_file(),
    }

    # Load via same paths as classification/predictor.py
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    sample = "Smoke is rising from the hill, therefore there must be fire."
    emb = generate_embeddings([sample])
    pred = model.predict(emb)[0]
    label = encoder.inverse_transform([pred])[0]
    checks["predictor-style inference OK"] = label in PRAMANA_LABELS

    return {"checks": checks, "sample_prediction": label}


def write_report(
    merge_stats: dict,
    df: pd.DataFrame,
    eval_results: dict,
    api_verify: dict,
) -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    cm = eval_results["confusion_matrix"]
    classes = eval_results["classes"]

    lines: list[str] = []
    lines.append("# InferAI — Final Retraining Report\n\n")
    lines.append(
        "Model retrained on merged synthetic + AAEC reviewed + IBM reviewed corpora. "
        "Architecture unchanged: **Sentence-BERT (`all-MiniLM-L6-v2`) + Logistic Regression**.\n\n"
    )

    lines.append("## 1. Merged corpus\n\n")
    lines.append(f"- Synthetic source: `{merge_stats['synthetic_path']}`\n")
    lines.append(f"- AAEC reviewed: `dataset/processed/aaec_reviewed_dataset.csv`\n")
    lines.append(f"- IBM reviewed: `dataset/processed/ibm_reviewed_dataset.csv`\n")
    lines.append(f"- Merged export: `{MERGED_CSV.relative_to(_ROOT).as_posix()}`\n")
    lines.append(f"- **Total rows:** {merge_stats['total_rows']}\n")
    lines.append(f"- **Duplicates removed:** {merge_stats['duplicates_removed']}\n")
    lines.append(f"- Shuffle seed: `{RANDOM_STATE}`\n\n")

    lines.append("### Source contribution (after dedup)\n\n")
    lines.append(f"- Synthetic: {merge_stats['synthetic_rows']}\n")
    lines.append(f"- AAEC: {merge_stats['aaec_rows']}\n")
    lines.append(f"- IBM: {merge_stats['ibm_rows']}\n\n")

    lines.append("### Class distribution\n\n")
    for label in PRAMANA_LABELS:
        n = int((df["pramana_label"] == label).sum())
        pct = 100.0 * n / len(df) if len(df) else 0.0
        lines.append(f"- {label}: {n} ({pct:.1f}%)\n")
    lines.append("\n")

    lines.append("### Strength distribution (real-world rows)\n\n")
    strength = df[df["strength_label"].isin(STRENGTH_LABELS)]["strength_label"]
    for s in STRENGTH_LABELS:
        n = int((strength == s).sum())
        lines.append(f"- {s}: {n}\n")
    lines.append("\n")

    lines.append("## 2. Evaluation (20% random holdout)\n\n")
    lines.append(f"- **Accuracy:** {eval_results['test_accuracy']:.4f}\n")
    lines.append(f"- **Macro precision:** {eval_results['macro_precision']:.4f}\n")
    lines.append(f"- **Macro recall:** {eval_results['macro_recall']:.4f}\n")
    lines.append(f"- **Macro F1:** {eval_results['macro_f1']:.4f}\n")
    lines.append(f"- Train accuracy: {eval_results['train_accuracy']:.4f}\n\n")

    lines.append("### Classification report\n\n")
    lines.append("```text\n")
    lines.append(eval_results["classification_report"].strip() + "\n")
    lines.append("```\n\n")

    lines.append("### Confusion matrix\n\n")
    lines.append("| True \\ Pred | " + " | ".join(classes) + " |\n")
    lines.append("| --- | " + " | ".join("---" for _ in classes) + " |\n")
    for i, true_lbl in enumerate(classes):
        row = " | ".join(str(int(cm[i, j])) for j in range(len(classes)))
        lines.append(f"| {true_lbl} | {row} |\n")
    lines.append("\n")

    held = eval_results["metrics"].get("held_out_test_set")
    if isinstance(held, dict):
        lines.append("## 3. Held-out curated test set (`dataset/test_set.csv`)\n\n")
        lines.append(f"- Accuracy: **{held.get('accuracy')}**\n\n")
        lines.append("```text\n")
        lines.append(str(held.get("classification_report", "")).strip() + "\n")
        lines.append("```\n\n")

    lines.append("## 4. Saved artifacts\n\n")
    lines.append("- `models/nyaya_model.pkl`\n")
    lines.append("- `models/label_encoder.pkl`\n")
    lines.append(f"- `models/shap_background.npy` ({eval_results['shap_background_size']} rows)\n")
    lines.append("- `models/evaluation/metrics.json`\n")
    lines.append("- `models/evaluation/classification_report.txt`\n")
    lines.append("- `models/evaluation/confusion_matrix.png`\n\n")

    lines.append("## 5. API compatibility check\n\n")
    lines.append(
        "`api/app.py` imports `predict_pramana_detailed` from `classification/predictor.py`, "
        "which loads `models/nyaya_model.pkl` and `models/label_encoder.pkl`. "
        "No API code changes required.\n\n"
    )
    for name, ok in api_verify["checks"].items():
        lines.append(f"- {name}: **{'PASS' if ok else 'FAIL'}**\n")
    lines.append(f"\nSample inference: \"Smoke is rising…\" → **{api_verify['sample_prediction']}**\n")

    REPORT_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    merged, merge_stats = merge_datasets()
    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8")

    print_merge_stats(merged, merge_stats)

    training_df, oversample_stats = oversample_minority_classes(merged)
    training_df.to_csv(OVERSAMPLED_CSV, index=False, encoding="utf-8")
    (_ROOT / "dataset" / "processed" / "oversample_stats.json").write_text(
        json.dumps(oversample_stats, indent=2), encoding="utf-8"
    )
    print_oversample_stats(oversample_stats)

    eval_results = train_and_evaluate(training_df)
    api_verify = verify_api_model_paths()
    write_report(merge_stats, training_df, eval_results, api_verify)

    print("\n" + "=" * 72)
    print("Retraining complete")
    print("=" * 72)
    print(f"Test accuracy:     {eval_results['test_accuracy']:.4f}")
    print(f"Macro precision:   {eval_results['macro_precision']:.4f}")
    print(f"Macro recall:      {eval_results['macro_recall']:.4f}")
    print(f"Macro F1:          {eval_results['macro_f1']:.4f}")
    print("\nAPI artifact checks:")
    for name, ok in api_verify["checks"].items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"\nSaved merged corpus: {MERGED_CSV}")
    print(f"Saved report:        {REPORT_OUT}")


if __name__ == "__main__":
    main()
