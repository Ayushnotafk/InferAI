"""
Strong baseline comparison suite.

Baselines (reviewer issue M-09)
--------------------------------
1. Majority-class classifier
2. TF-IDF + LinearSVC
3. TF-IDF + Logistic Regression
4. Sentence-BERT + Logistic Regression (current system, ML-only)
5. Symbolic rule engine only
6. Hybrid (ML + rules, fixed alpha from alpha_selection_report)

All baselines are trained on the TRAIN partition and evaluated on the VAL
partition using group-disjoint splits from build_clean_splits.py.
Final numbers on TEST come from scripts/evaluate_final_model.py (run once only).

Usage (from project root):
    python scripts/run_baseline_comparison.py

Outputs
-------
reports/baseline_comparison.md
reports/baseline_comparison.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _df_to_md(df, index=False):
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    h = "| " + " | ".join(str(c) for c in cols) + " |"
    s = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([h, s] + rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.embedder import generate_embeddings
from classification.hybrid_reasoning import hybrid_fuse, rule_distribution, DEFAULT_CLASS_ORDER

PROCESSED = ROOT / "dataset" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MANIFESTS = ROOT / "dataset" / "split_manifests"

TRAIN_CSV = PROCESSED / "split_train.csv"
VAL_CSV = PROCESSED / "split_val.csv"

PRAMANA_LABELS = ["Anumana", "Pratyaksha", "Shabda", "Upamana"]


def load_split(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    df = df[df["pramana_label"].isin(PRAMANA_LABELS)].dropna(subset=["text", "pramana_label"])
    return df["text"].astype(str).tolist(), df["pramana_label"].astype(str).tolist()


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=PRAMANA_LABELS, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=PRAMANA_LABELS, digits=4, zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "classification_report": report,
    }


def run_majority(X_train, y_train, X_val, y_val) -> dict:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    return evaluate(y_val, clf.predict(X_val).tolist())


def run_tfidf_svm(X_train, y_train, X_val, y_val) -> dict:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ("svm", LinearSVC(max_iter=5000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return evaluate(y_val, pipe.predict(X_val).tolist())


def run_tfidf_lr(X_train, y_train, X_val, y_val) -> dict:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return evaluate(y_val, pipe.predict(X_val).tolist())


def run_sbert_lr(X_train_text, y_train, X_val_text, y_val) -> dict:
    import joblib
    model = joblib.load(MODELS_DIR / "nyaya_model.pkl")
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    X_val = generate_embeddings(X_val_text)
    preds_enc = model.predict(X_val)
    preds = encoder.inverse_transform(preds_enc).tolist()
    return evaluate(y_val, preds)


def run_rules_only(X_val_text, y_val) -> dict:
    preds = []
    class_order = list(DEFAULT_CLASS_ORDER)
    for text in X_val_text:
        p = rule_distribution(text, class_order=tuple(class_order))
        preds.append(class_order[int(np.argmax(p))])
    return evaluate(y_val, preds)


def run_hybrid(X_val_text, y_val, alpha: float) -> dict:
    import joblib
    model = joblib.load(MODELS_DIR / "nyaya_model.pkl")
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    X_val_emb = generate_embeddings(X_val_text)
    ml_probs = model.predict_proba(X_val_emb)
    class_order = list(encoder.classes_)
    preds = []
    for i, text in enumerate(X_val_text):
        result = hybrid_fuse(
            ml_probs[i], text, class_order=class_order,
            ml_weight=alpha, rule_weight=1.0 - alpha
        )
        preds.append(result["final_label"])
    return evaluate(y_val, preds)


def main() -> None:
    if not TRAIN_CSV.is_file() or not VAL_CSV.is_file():
        raise FileNotFoundError(
            "Missing split CSVs. Run scripts/build_clean_splits.py first."
        )

    # Load alpha from registry
    reg_path = MANIFESTS / "experiment_registry.json"
    alpha = 0.8  # default
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text())
        alpha = reg.get("alpha_selection", {}).get("selected_alpha", 0.8)
    print(f"Using alpha={alpha} (from experiment_registry.json)")

    print("Loading splits...")
    X_train, y_train = load_split(TRAIN_CSV)
    X_val, y_val = load_split(VAL_CSV)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    rows = []

    # --- 1. Majority class ---
    print("\n[1/6] Majority class...")
    t0 = time.time()
    r = run_majority(X_train, y_train, X_val, y_val)
    rows.append({"Baseline": "Majority class", **{k: v for k, v in r.items() if k != "classification_report"}, "latency_s": round(time.time() - t0, 2)})
    print(f"  Macro-F1: {r['macro_f1']}")

    # --- 2. TF-IDF + LinearSVC ---
    print("[2/6] TF-IDF + LinearSVC...")
    t0 = time.time()
    r_svm = run_tfidf_svm(X_train, y_train, X_val, y_val)
    rows.append({"Baseline": "TF-IDF + LinearSVC", **{k: v for k, v in r_svm.items() if k != "classification_report"}, "latency_s": round(time.time() - t0, 2)})
    print(f"  Macro-F1: {r_svm['macro_f1']}")

    # --- 3. TF-IDF + LR ---
    print("[3/6] TF-IDF + Logistic Regression...")
    t0 = time.time()
    r_tflr = run_tfidf_lr(X_train, y_train, X_val, y_val)
    rows.append({"Baseline": "TF-IDF + LogReg", **{k: v for k, v in r_tflr.items() if k != "classification_report"}, "latency_s": round(time.time() - t0, 2)})
    print(f"  Macro-F1: {r_tflr['macro_f1']}")

    # --- 4. SBERT + LR ---
    print("[4/6] Sentence-BERT + LogReg (current system, ML-only)...")
    t0 = time.time()
    r_sbert = run_sbert_lr(X_train, y_train, X_val, y_val)
    rows.append({"Baseline": "SBERT + LogReg (ML-only)", **{k: v for k, v in r_sbert.items() if k != "classification_report"}, "latency_s": round(time.time() - t0, 2)})
    print(f"  Macro-F1: {r_sbert['macro_f1']}")

    # --- 5. Rules only ---
    print("[5/6] Symbolic rules only...")
    t0 = time.time()
    r_rules = run_rules_only(X_val, y_val)
    rows.append({"Baseline": "Symbolic rules only", **{k: v for k, v in r_rules.items() if k != "classification_report"}, "latency_s": round(time.time() - t0, 2)})
    print(f"  Macro-F1: {r_rules['macro_f1']}")

    # --- 6. Hybrid ---
    print(f"[6/6] Hybrid (α={alpha})...")
    t0 = time.time()
    r_hyb = run_hybrid(X_val, y_val, alpha)
    rows.append({"Baseline": f"Hybrid (α={alpha})", **{k: v for k, v in r_hyb.items() if k != "classification_report"}, "latency_s": round(time.time() - t0, 2)})
    print(f"  Macro-F1: {r_hyb['macro_f1']}")

    # --- Summary table ---
    result_df = pd.DataFrame(rows)
    result_df.to_csv(REPORTS_DIR / "baseline_comparison.csv", index=False)
    print("\n" + "=" * 70)
    print("Baseline comparison summary (VAL set)")
    print("=" * 70)
    print(result_df[["Baseline", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].to_string(index=False))

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = result_df["Baseline"].tolist()
    f1s = result_df["macro_f1"].tolist()
    colors = ["#aaaaaa", "#4c72b0", "#4c72b0", "#55a868", "#dd8452", "#c44e52"]
    bars = ax.barh(methods, f1s, color=colors[:len(methods)])
    ax.set_xlabel("Macro-F1 (VAL)")
    ax.set_title("Baseline Comparison — InferAI (Group-Disjoint VAL)")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, f1s):
        ax.text(min(val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig_path = REPORTS_DIR / "figures" / "baseline_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Figure: {fig_path.relative_to(ROOT)}")

    # --- Markdown report ---
    lines = [
        "# Baseline Comparison (VAL Set — Group-Disjoint)\n\n",
        "All baselines trained on `dataset/processed/split_train.csv`, "
        "evaluated on `dataset/processed/split_val.csv`. "
        "Group-disjoint splits (essay_id / topic_group / template_family) ensure "
        "no document or topic leakage between partitions.\n\n",
        "## Results\n\n",
        _df_to_md(result_df) + "\n\n",
        "## Per-baseline classification reports\n\n",
    ]
    for i, (row, r_full) in enumerate(zip(rows, [r, r_svm, r_tflr, r_sbert, r_rules, r_hyb])):
        lines.append(f"### {row['Baseline']}\n\n```text\n{r_full['classification_report']}```\n\n")

    lines.append("## Notes\n\n")
    lines.append(
        f"- Fusion alpha α={alpha} was selected on this VAL set by `scripts/select_alpha_and_calibrate.py`.\n"
        "- Final test evaluation (on `dataset/clean_final_test.csv`) is run exactly once by `scripts/evaluate_final_model.py`.\n"
        "- A fine-tuned transformer baseline (RoBERTa/DeBERTa) is planned as P1 work; this requires GPU training.\n"
    )
    (REPORTS_DIR / "baseline_comparison.md").write_text("".join(lines), encoding="utf-8")
    print(f"  Report: {(REPORTS_DIR / 'baseline_comparison.md').relative_to(ROOT)}")
    print("\n✅ Baseline comparison complete.")


if __name__ == "__main__":
    main()
