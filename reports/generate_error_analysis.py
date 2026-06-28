"""
Diagnostic error analysis for the retrained model on dataset/test_set.csv.

Outputs:
  - reports/error_analysis.md
  - reports/figures/error_analysis_confusion_matrix.png

Does not retrain the model.

Usage:
    python reports/generate_error_analysis.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings
from reports._common import apply_plot_style, ensure_reports_dir

MODEL_PATH = _ROOT / "models" / "nyaya_model.pkl"
ENCODER_PATH = _ROOT / "models" / "label_encoder.pkl"
TEST_PATH = _ROOT / "dataset" / "test_set.csv"
MERGED_PATH = _ROOT / "dataset" / "processed" / "merged_retraining_corpus.csv"
REPORT_OUT = _ROOT / "reports" / "error_analysis.md"
FIG_OUT = _ROOT / "reports" / "figures" / "error_analysis_confusion_matrix.png"

PRAMANA_LABELS = ("Pratyaksha", "Anumana", "Upamana", "Shabda")
MAX_MISCLASSIFIED = 50

# Cue families for diagnostic tagging
CUE_FAMILIES: dict[str, list[str]] = {
    "Pratyaksha": [
        "saw", "heard", "observed", "measured", "reading", "display", "log", "sensor",
        "fluoresced", "bench notes", "instrument", "record", "data", "survey",
    ],
    "Shabda": [
        "according to", "expert", "researcher", "report", "study", "manual", "handbook",
        "guideline", "authority", "encyclopedia", "textbook", "constitution",
    ],
    "Upamana": [
        "like", "similar to", "analogy", "analogous", "compared", "as if", "resembles", "just as",
    ],
    "Anumana": [
        "because", "therefore", "thus", "hence", "implies", "suggests", "likely", "so the",
        "which makes", "plausible", "consistent with",
    ],
}

PAIR_DIAGNOSES: dict[tuple[str, str], str] = {
    ("Pratyaksha", "Anumana"): (
        "Perceptual or measurement language co-occurs with inferential templates "
        "(therefore/suggests), so the linear head follows dominant Anumana training patterns."
    ),
    ("Pratyaksha", "Shabda"): (
        "Weak perceptual cues without strong sensory verbs; testimonial or authority "
        "framing may be absent but Anumana/Shabda overlap in short curated spans."
    ),
    ("Shabda", "Anumana"): (
        "Authority or source cues are implicit; the span reads as causal or inferential "
        "reason-giving rather than explicit testimony."
    ),
    ("Shabda", "Pratyaksha"): (
        "Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha."
    ),
    ("Upamana", "Anumana"): (
        "Comparative structure is subtle or missing explicit 'like/similar to'; "
        "argument functions as general inference."
    ),
    ("Upamana", "Shabda"): (
        "Analogy references authoritative sources, mixing comparison with testimony."
    ),
    ("Anumana", "Upamana"): (
        "Explicit analogy markers (like, similar to) trigger Upamana despite inferential intent."
    ),
    ("Anumana", "Shabda"): (
        "Cites experts, manuals, or reports while also stating an inferential conclusion."
    ),
    ("Anumana", "Pratyaksha"): (
        "Direct sensory reporting with minimal inference markers classified as perception."
    ),
}


def _detect_cues(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    hits = {label: [] for label in PRAMANA_LABELS}
    for label, cues in CUE_FAMILIES.items():
        for cue in cues:
            if cue in lowered:
                hits[label].append(cue)
    return hits


def _diagnose_error(true_lbl: str, pred_lbl: str, text: str, train_counts: dict[str, int]) -> dict[str, str]:
    hits = _detect_cues(text)
    true_hits = hits.get(true_lbl, [])
    pred_hits = hits.get(pred_lbl, [])

    causes: list[str] = []

    # Class imbalance
    total_train = sum(train_counts.values()) or 1
    true_share = train_counts.get(true_lbl, 0) / total_train
    if true_share < 0.10:
        causes.append(
            f"class imbalance (only {train_counts.get(true_lbl, 0)} {true_lbl} training examples, "
            f"{100 * true_share:.1f}% of corpus)"
        )

    # Missing heuristics
    if true_hits and not pred_hits:
        causes.append(
            f"missing or weak heuristics for true class (found {true_lbl} cues: {', '.join(true_hits[:4])})"
        )
    elif not true_hits:
        causes.append(
            f"missing heuristics for true class (no strong {true_lbl} lexical cues in span)"
        )

    # Ambiguous wording
    active_families = sum(1 for lbl in PRAMANA_LABELS if hits[lbl])
    if active_families >= 2:
        causes.append(
            "ambiguous wording (multiple pramana cue families present in the same span)"
        )

    # Insufficient training examples
    if train_counts.get(true_lbl, 0) < 100:
        causes.append(
            f"insufficient training examples for {true_lbl} ({train_counts.get(true_lbl, 0)} rows)"
        )

    pair_key = (true_lbl, pred_lbl)
    pair_note = PAIR_DIAGNOSES.get(
        pair_key,
        f"the model confuses rhetorical patterns associated with {true_lbl} and {pred_lbl}.",
    )

    if not causes:
        causes.append("dominant Anumana decision boundary despite superficial cue overlap")

    return {
        "likely_reason": pair_note,
        "contributing_factors": "; ".join(causes),
    }


def _load_training_counts() -> dict[str, int]:
    if MERGED_PATH.is_file():
        df = pd.read_csv(MERGED_PATH)
        col = "pramana_label" if "pramana_label" in df.columns else "label"
        return df[col].value_counts().to_dict()
    return {}


def evaluate() -> dict:
    if not MODEL_PATH.is_file() or not ENCODER_PATH.is_file():
        raise FileNotFoundError("Trained model not found. Run classification/retrain_merged.py first.")
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Missing {TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    tdf = pd.read_csv(TEST_PATH)

    texts = tdf["text"].astype(str).tolist()
    y_true_str = tdf["pramana_label"].astype(str).tolist()
    y_true = le.transform(y_true_str)

    X = generate_embeddings(texts)
    proba = model.predict_proba(X)
    y_pred = model.predict(X)
    y_pred_str = le.inverse_transform(y_pred)
    classes = list(le.classes_)

    rows: list[dict] = []
    for i, text in enumerate(texts):
        dist = {classes[j]: float(proba[i, j]) for j in range(len(classes))}
        conf = float(max(proba[i]) * 100.0)
        rows.append(
            {
                "text": text,
                "true_label": y_true_str[i],
                "predicted_label": y_pred_str[i],
                "confidence": round(conf, 2),
                "probabilities": dist,
                "correct": y_true_str[i] == y_pred_str[i],
            }
        )

    results_df = pd.DataFrame(rows)
    accuracy = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    return {
        "results_df": results_df,
        "accuracy": accuracy,
        "classes": classes,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
        "cm": cm,
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, digits=4, zero_division=0
        ),
    }


def _confusion_pairs(cm: np.ndarray, classes: list[str]) -> list[dict]:
    pairs: list[dict] = []
    for i, true_lbl in enumerate(classes):
        row_sum = int(cm[i].sum())
        for j, pred_lbl in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append(
                    {
                        "true": true_lbl,
                        "predicted": pred_lbl,
                        "count": count,
                        "rate": count / row_sum if row_sum else 0.0,
                    }
                )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs


def write_report(eval_data: dict, train_counts: dict[str, int]) -> None:
    ensure_reports_dir()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    classes = eval_data["classes"]
    cm = eval_data["cm"]
    prec, rec, f1, support = (
        eval_data["precision"],
        eval_data["recall"],
        eval_data["f1"],
        eval_data["support"],
    )

    # Confusion matrix figure
    apply_plot_style()
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"Test-set confusion matrix (accuracy = {eval_data['accuracy']:.1%})")
    plt.tight_layout()
    plt.savefig(FIG_OUT, bbox_inches="tight")
    plt.close()

    per_class = pd.DataFrame(
        {
            "Class": classes,
            "Precision": np.round(prec, 4),
            "Recall": np.round(rec, 4),
            "F1": np.round(f1, 4),
            "Support": support.astype(int),
        }
    )

    lowest_recall_idx = int(np.argmin(rec))
    lowest_prec_idx = int(np.argmin(prec))
    confusion_pairs = _confusion_pairs(cm, classes)

    misclassified = eval_data["results_df"][~eval_data["results_df"]["correct"]].head(MAX_MISCLASSIFIED)

    lines: list[str] = []
    lines.append("# InferAI — Error Analysis (Retrained Model)\n\n")
    lines.append(
        "Diagnostic evaluation on the held-out curated set `dataset/test_set.csv` "
        f"(n={len(eval_data['results_df'])}). **No retraining performed.**\n\n"
    )

    lines.append("## 1. Overall performance\n\n")
    lines.append(f"- **Accuracy:** {eval_data['accuracy']:.4f} ({eval_data['accuracy']:.1%})\n")
    lines.append(f"- **Misclassified:** {int((~eval_data['results_df']['correct']).sum())}\n\n")

    lines.append("## 2. Per-class metrics\n\n")
    lines.append("| Class | Precision | Recall | F1 | Support |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for _, row in per_class.iterrows():
        lines.append(
            f"| {row['Class']} | {row['Precision']} | {row['Recall']} | {row['F1']} | {row['Support']} |\n"
        )
    lines.append("\n")

    lines.append("## 3. Key findings\n\n")
    lines.append(
        f"- **Lowest recall class:** {classes[lowest_recall_idx]} "
        f"({rec[lowest_recall_idx]:.4f})\n"
    )
    lines.append(
        f"- **Lowest precision class:** {classes[lowest_prec_idx]} "
        f"({prec[lowest_prec_idx]:.4f})\n\n"
    )

    lines.append("### Most common confusion pairs\n\n")
    lines.append("| True class | Predicted class | Count | Rate (of true class) |\n")
    lines.append("| --- | --- | ---: | ---: |\n")
    for pair in confusion_pairs[:8]:
        lines.append(
            f"| {pair['true']} | {pair['predicted']} | {pair['count']} | {pair['rate']:.1%} |\n"
        )
    lines.append("\n")

    lines.append("## 4. Confusion matrix\n\n")
    lines.append(f"![Confusion matrix](figures/error_analysis_confusion_matrix.png)\n\n")
    lines.append("| True \\ Pred | " + " | ".join(classes) + " |\n")
    lines.append("| --- | " + " | ".join("---" for _ in classes) + " |\n")
    for i, true_lbl in enumerate(classes):
        row = " | ".join(str(int(cm[i, j])) for j in range(len(classes)))
        lines.append(f"| {true_lbl} | {row} |\n")
    lines.append("\n")

    lines.append("## 5. Classification report\n\n")
    lines.append("```text\n")
    lines.append(eval_data["classification_report"].strip() + "\n")
    lines.append("```\n\n")

    lines.append(f"## 6. Misclassified examples (up to {MAX_MISCLASSIFIED})\n\n")
    for i, row in enumerate(misclassified.itertuples(index=False), 1):
        diag = _diagnose_error(row.true_label, row.predicted_label, row.text, train_counts)
        prob_parts = ", ".join(
            f"{k}={row.probabilities[k]:.3f}" for k in classes
        )
        lines.append(f"### {i}. True: **{row.true_label}** → Predicted: **{row.predicted_label}** "
                     f"(confidence {row.confidence:.1f}%)\n\n")
        lines.append(f"**Text:** \"{row.text}\"\n\n")
        lines.append(f"**Probability distribution:** {prob_parts}\n\n")
        lines.append(f"**Likely confusion:** {diag['likely_reason']}\n\n")
        lines.append(f"**Contributing factors:** {diag['contributing_factors']}\n\n")

    lines.append("## 7. Summary diagnosis\n\n")
    lines.append("### Training corpus skew (merged retraining set)\n\n")
    total_tr = sum(train_counts.values()) or 1
    for lbl in PRAMANA_LABELS:
        n = train_counts.get(lbl, 0)
        lines.append(f"- {lbl}: {n} ({100 * n / total_tr:.1f}%)\n")
    lines.append("\n")

    top_pair = confusion_pairs[0] if confusion_pairs else None
    lines.append("### Root causes\n\n")
    lines.append(
        "1. **Class imbalance:** Anumana dominates the merged training corpus (~80%), "
        "biasing the logistic head toward inferential labels even when test spans carry "
        "perception, testimony, or analogy cues.\n"
    )
    lines.append(
        "2. **Missing heuristics:** The hybrid rule engine is not used at training time; "
        "embedding-only classification misses explicit Nyaya cue words unless they align "
        "with Sentence-BERT geometry learned from skewed data.\n"
    )
    lines.append(
        "3. **Ambiguous wording:** Curated test seeds reuse template frames across domains "
        "(e.g. *Bench notes*, *Test augmentation*) where multiple pramana readings are plausible.\n"
    )
    lines.append(
        "4. **Insufficient training examples:** Pratyaksha, Upamana, and Shabda each have "
        "fewer than 150 merged rows vs. >1,100 Anumana rows, limiting recall on minority classes.\n"
    )
    if top_pair:
        lines.append(
            f"\nThe single most frequent error is **{top_pair['true']} → {top_pair['predicted']}** "
            f"({top_pair['count']} cases), consistent with the imbalance and template overlap described above.\n"
        )

    REPORT_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    train_counts = _load_training_counts()
    eval_data = evaluate()
    write_report(eval_data, train_counts)

    classes = eval_data["classes"]
    prec, rec = eval_data["precision"], eval_data["recall"]
    n_err = int((~eval_data["results_df"]["correct"]).sum())

    print("Error analysis complete")
    print("=" * 50)
    print(f"Test accuracy: {eval_data['accuracy']:.4f}")
    print(f"Misclassified: {n_err}")
    print(f"Lowest recall:  {classes[int(np.argmin(rec))]} ({np.min(rec):.4f})")
    print(f"Lowest precision: {classes[int(np.argmin(prec))]} ({np.min(prec):.4f})")
    print(f"\nSaved: {REPORT_OUT}")
    print(f"Saved: {FIG_OUT}")


if __name__ == "__main__":
    main()
