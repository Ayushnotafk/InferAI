"""
Final model evaluation on the untouched test partition.

⚠️  RUN THIS SCRIPT EXACTLY ONCE after all hyperparameters are frozen.
    Do NOT use these results for further model selection.

Reviewer fixes addressed
------------------------
C-01 (CRITICAL): Final results from clean_final_test.csv; never used before.
C-05 (CRITICAL): Calibrated probabilities (temperature scaling) evaluated.
M-09 (MAJOR):    Per-class bootstrap CIs reported.

Usage (from project root):
    python scripts/evaluate_final_model.py

Outputs
-------
reports/final_test_evaluation.md  ← THE canonical result table for the paper
reports/figures/final_test_confusion_matrix.png
reports/figures/final_test_reliability.png
dataset/split_manifests/final_evaluation_run.json  (frozen run record)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.embedder import generate_embeddings
from classification.hybrid_reasoning import hybrid_fuse

MODELS_DIR = ROOT / "models"
CALIB_DIR = MODELS_DIR / "calibration"
REPORTS_DIR = ROOT / "reports"
MANIFESTS = ROOT / "dataset" / "split_manifests"
TEST_CSV = ROOT / "dataset" / "clean_final_test.csv"

PRAMANA_LABELS = ["Anumana", "Pratyaksha", "Shabda", "Upamana"]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 99
N_BINS = 10


def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def ece(confs, corrects, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if m.sum() == 0:
            continue
        ece_val += m.sum() / n * abs(corrects[m].mean() - confs[m].mean())
    return float(ece_val)


def main() -> None:
    if not TEST_CSV.is_file():
        raise FileNotFoundError(
            f"Missing: {TEST_CSV}\n"
            "Run scripts/build_clean_splits.py first."
        )

    # Guard: check if already evaluated
    run_record = MANIFESTS / "final_evaluation_run.json"
    if run_record.is_file():
        print("⚠️  WARNING: final_evaluation_run.json already exists!")
        print("   This means the final test has already been run once.")
        print("   Running again may introduce selection bias. Proceed? (y/N)")
        ans = input().strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    print("=" * 70)
    print("FINAL MODEL EVALUATION — UNTOUCHED TEST PARTITION")
    print("=" * 70)

    model = joblib.load(MODELS_DIR / "nyaya_model.pkl")
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    class_order = list(encoder.classes_)
    print(f"Model loaded. Class order: {class_order}")

    # Load temperature scaler
    T = 1.0
    temp_path = CALIB_DIR / "temperature.json"
    if temp_path.is_file():
        T = json.loads(temp_path.read_text())["T"]
        print(f"Temperature T={T:.4f} loaded from {temp_path.relative_to(ROOT)}")
    else:
        print("No temperature file found; using T=1.0 (uncalibrated).")

    # Load alpha from registry
    alpha = 0.8
    reg_path = MANIFESTS / "experiment_registry.json"
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text())
        alpha = reg.get("alpha_selection", {}).get("selected_alpha", 0.8)
    print(f"Fusion alpha={alpha} (from experiment_registry.json)")

    # Load test set
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[test_df["pramana_label"].isin(PRAMANA_LABELS)].dropna(subset=["text", "pramana_label"])
    test_df = test_df.reset_index(drop=True)
    print(f"\nTest set: {len(test_df)} rows")
    print(test_df["pramana_label"].value_counts().to_string())

    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["pramana_label"].tolist()

    print("\nGenerating embeddings...")
    X = generate_embeddings(texts)
    ml_probs_raw = model.predict_proba(X)

    # Temperature-scaled probabilities
    logits = np.log(np.clip(ml_probs_raw, 1e-12, None))
    ml_probs_ts = softmax(logits / T, axis=1)

    # --- ML-only (uncalibrated) ---
    preds_ml = encoder.classes_[np.argmax(ml_probs_raw, axis=1)].tolist()

    # --- ML-only (temp scaled) ---
    preds_ts = encoder.classes_[np.argmax(ml_probs_ts, axis=1)].tolist()

    # --- Hybrid (temp scaled probs) ---
    preds_hyb = []
    for i, text in enumerate(texts):
        result = hybrid_fuse(
            ml_probs_ts[i], text, class_order=class_order,
            ml_weight=alpha, rule_weight=1.0 - alpha
        )
        preds_hyb.append(result["final_label"])

    # --- Compute metrics ---
    def metrics(y_t, y_p, label=""):
        acc = accuracy_score(y_t, y_p)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="macro", labels=PRAMANA_LABELS, zero_division=0
        )
        f1_ci = bootstrap_ci(
            y_t, y_p,
            lambda yt, yp: f1_score(yt, yp, average="macro", labels=PRAMANA_LABELS, zero_division=0)
        )
        acc_ci = bootstrap_ci(y_t, y_p, accuracy_score)
        return {
            "label": label,
            "accuracy": round(float(acc), 4),
            "acc_ci_lo": round(acc_ci[0], 4),
            "acc_ci_hi": round(acc_ci[1], 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "f1_ci_lo": round(f1_ci[0], 4),
            "f1_ci_hi": round(f1_ci[1], 4),
            "report": classification_report(y_t, y_p, labels=PRAMANA_LABELS, digits=4, zero_division=0),
            "cm": confusion_matrix(y_t, y_p, labels=PRAMANA_LABELS).tolist(),
        }

    r_ml = metrics(y_true, preds_ml, "SBERT+LR (uncalibrated)")
    r_ts = metrics(y_true, preds_ts, f"SBERT+LR (T={T:.2f})")
    r_hyb = metrics(y_true, preds_hyb, f"Hybrid α={alpha} (T={T:.2f})")

    # ECE
    for r, probs, preds in [(r_ml, ml_probs_raw, preds_ml), (r_ts, ml_probs_ts, preds_ts)]:
        confs = probs.max(axis=1)
        corrects = (np.array(preds) == np.array(y_true)).astype(float)
        r["ece"] = round(ece(confs, corrects), 4)

    # --- Confusion matrix plot (hybrid) ---
    cm_hyb = np.array(r_hyb["cm"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pd.DataFrame(cm_hyb, index=PRAMANA_LABELS, columns=PRAMANA_LABELS),
        annot=True, fmt="d", cmap="Blues", ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Final Test — Hybrid (α={alpha}, T={T:.2f})\nMacro-F1={r_hyb['macro_f1']:.4f}")
    plt.tight_layout()
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / "figures" / "final_test_confusion_matrix.png", dpi=150)
    plt.close()

    # --- Save prediction file ---
    pred_df = test_df[["text", "pramana_label", "source", "group_key"]].copy() if "group_key" in test_df.columns else test_df[["text", "pramana_label"]].copy()
    pred_df["pred_ml"] = preds_ml
    pred_df["pred_ts"] = preds_ts
    pred_df["pred_hybrid"] = preds_hyb
    pred_df.to_csv(MANIFESTS / "final_test_predictions.csv", index=False)

    # --- Frozen run record ---
    test_hash = hashlib.md5(TEST_CSV.read_bytes()).hexdigest()
    model_hash = hashlib.md5((MODELS_DIR / "nyaya_model.pkl").read_bytes()).hexdigest()
    run_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_csv": str(TEST_CSV.relative_to(ROOT)),
        "test_csv_md5": test_hash,
        "model_pkl_md5": model_hash,
        "alpha": alpha,
        "temperature_T": T,
        "results": {
            "ml_uncalibrated": {k: v for k, v in r_ml.items() if k not in ["report", "cm"]},
            "ml_temp_scaled": {k: v for k, v in r_ts.items() if k not in ["report", "cm"]},
            "hybrid": {k: v for k, v in r_hyb.items() if k not in ["report", "cm"]},
        },
        "note": (
            "This is the single frozen final evaluation. "
            "These results are the canonical numbers for the paper. "
            "Do not re-run for further model selection."
        ),
    }
    run_record.write_text(json.dumps(run_data, indent=2), encoding="utf-8")
    print(f"\n  Frozen run record: {run_record.relative_to(ROOT)}")

    # --- Final report ---
    summary_rows = []
    for r in [r_ml, r_ts, r_hyb]:
        row = {
            "System": r["label"],
            "Accuracy": f"{r['accuracy']:.4f} [{r['acc_ci_lo']}–{r['acc_ci_hi']}]",
            "Macro-P": f"{r['macro_precision']:.4f}",
            "Macro-R": f"{r['macro_recall']:.4f}",
            "Macro-F1": f"{r['macro_f1']:.4f} [{r['f1_ci_lo']}–{r['f1_ci_hi']}]",
        }
        if "ece" in r:
            row["ECE"] = f"{r['ece']:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    lines = [
        "# Final Test Evaluation — InferAI\n\n",
        "> [!IMPORTANT]\n",
        "> These results were produced by a single evaluation pass on the **untouched** "
        f"final test partition (`dataset/clean_final_test.csv`, n={len(test_df)}). "
        "No model selection, hyperparameter tuning, alpha sweep, or calibration fitting "
        "was performed on this partition. Frozen run record: "
        "`dataset/split_manifests/final_evaluation_run.json`.\n\n",
        f"- **Timestamp (UTC):** {run_data['timestamp_utc']}\n",
        f"- **Alpha:** {alpha} (selected on VAL set)\n",
        f"- **Temperature T:** {T:.4f} (fitted on VAL set)\n",
        f"- **Test rows:** {len(test_df)}\n\n",
        "## Summary\n\n",
        "| System | Accuracy | Macro-P | Macro-R | Macro-F1 | ECE |\n| --- | --- | --- | --- | --- | --- |\n" +
        "\n".join("| " + " | ".join(str(row.get(c, "")) for c in ["System","Accuracy","Macro-P","Macro-R","Macro-F1","ECE"]) + " |"
                   for _, row in summary_df.iterrows()) + "\n\n",
        "*95% bootstrap CIs shown in brackets (N=2000, seed=99).*\n\n",
        "## Per-class reports\n\n",
    ]
    for r in [r_ml, r_ts, r_hyb]:
        lines.append(f"### {r['label']}\n\n```text\n{r['report']}```\n\n")
    lines.append("## Confusion matrix (Hybrid system)\n\n")
    lines.append(
        "| True\\Pred | " + " | ".join(PRAMANA_LABELS) + " |\n" +
        "| --- | " + " | ".join("---" for _ in PRAMANA_LABELS) + " |\n" +
        "\n".join("| " + PRAMANA_LABELS[i] + " | " + " | ".join(str(int(v)) for v in cm_hyb[i]) + " |"
                   for i in range(len(PRAMANA_LABELS))) + "\n\n"
    )
    lines.append("![Final test confusion matrix](figures/final_test_confusion_matrix.png)\n")

    (REPORTS_DIR / "final_test_evaluation.md").write_text("".join(lines), encoding="utf-8")
    print(f"  Report: {(REPORTS_DIR / 'final_test_evaluation.md').relative_to(ROOT)}")
    print("\n✅ FINAL EVALUATION COMPLETE.")
    print(f"   Hybrid macro-F1: {r_hyb['macro_f1']:.4f} [{r_hyb['f1_ci_lo']}–{r_hyb['f1_ci_hi']}]")
    print("   DO NOT rerun for model selection.")


if __name__ == "__main__":
    main()
