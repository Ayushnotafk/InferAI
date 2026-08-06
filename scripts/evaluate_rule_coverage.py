"""
Rule-evidence coverage and explainability evaluation.

Reviewer fix: MISS-4 (MAJOR)
-------------------------------
Measures:
1. Rule coverage: % of predictions where ≥1 symbolic rule matched (per-class)
2. Coverage gap: predictions where explanation is embedding-only (no rules)
3. Generates an annotation sheet for human explainability rating

Usage (from project root):
    python scripts/evaluate_rule_coverage.py

Outputs
-------
reports/rule_coverage_report.md
reports/explainability_rating_sheet.csv  ← give to human raters
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.embedder import generate_embeddings
from classification.hybrid_reasoning import hybrid_fuse, score_weighted_rules, DEFAULT_CLASS_ORDER

MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "dataset" / "processed"
REPORTS_DIR = ROOT / "reports"
MANIFESTS = ROOT / "dataset" / "split_manifests"

VAL_CSV = PROCESSED / "split_val.csv"
PRAMANA_LABELS = ["Anumana", "Pratyaksha", "Shabda", "Upamana"]


def main() -> None:
    if not VAL_CSV.is_file():
        raise FileNotFoundError(
            f"Missing: {VAL_CSV}\nRun scripts/build_clean_splits.py first."
        )

    model = joblib.load(MODELS_DIR / "nyaya_model.pkl")
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    class_order = list(encoder.classes_)

    reg_path = MANIFESTS / "experiment_registry.json"
    alpha = 0.8
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text())
        alpha = reg.get("alpha_selection", {}).get("selected_alpha", 0.8)

    df = pd.read_csv(VAL_CSV)
    df = df[df["pramana_label"].isin(PRAMANA_LABELS)].dropna(subset=["text", "pramana_label"])
    df = df.reset_index(drop=True)
    texts = df["text"].astype(str).tolist()
    y_true = df["pramana_label"].tolist()

    print(f"Evaluating rule coverage on {len(df)} val rows...")
    X = generate_embeddings(texts)
    ml_probs = model.predict_proba(X)

    records = []
    for i, (text, true_label) in enumerate(zip(texts, y_true)):
        raw_scores, matched_cues = score_weighted_rules(text)
        any_rule_hit = any(len(cues) > 0 for cues in matched_cues.values())
        total_rules_fired = sum(len(cues) for cues in matched_cues.values())
        top_rule_pramana = max(raw_scores, key=raw_scores.get) if any(raw_scores.values()) else None
        top_rule_score = max(raw_scores.values()) if any(raw_scores.values()) else 0.0
        all_cues = []
        for p, cues in matched_cues.items():
            for c in cues:
                all_cues.append(f"{p}:{c['cue']}({c['span']})")

        result = hybrid_fuse(
            ml_probs[i], text, class_order=class_order,
            ml_weight=alpha, rule_weight=1.0 - alpha
        )

        records.append({
            "text": text[:150],
            "true_label": true_label,
            "ml_label": result["ml_label"],
            "rule_label": result["rule_label"] or "none",
            "final_label": result["final_label"],
            "correct": true_label == result["final_label"],
            "any_rule_hit": any_rule_hit,
            "total_rules_fired": total_rules_fired,
            "top_rule_pramana": top_rule_pramana or "none",
            "top_rule_score": round(top_rule_score, 3),
            "matched_cues_summary": "; ".join(all_cues[:5]),
            "ml_rule_agree": result["agreement"].get("agreement", None),
            "adjusted_confidence": round(result["adjusted_confidence"], 2),
        })

    result_df = pd.DataFrame(records)

    # --- Coverage statistics ---
    total = len(result_df)
    covered = result_df["any_rule_hit"].sum()
    coverage_pct = 100.0 * covered / total

    print(f"\nOverall rule coverage: {covered}/{total} ({coverage_pct:.1f}%)")

    per_class_cov = result_df.groupby("true_label").agg(
        total=("any_rule_hit", "count"),
        covered=("any_rule_hit", "sum"),
    ).reset_index()
    per_class_cov["coverage_pct"] = (100.0 * per_class_cov["covered"] / per_class_cov["total"]).round(1)
    print("\nPer-class rule coverage:")
    print(per_class_cov.to_string(index=False))

    # Accuracy when rule covered vs not covered
    acc_covered = result_df[result_df["any_rule_hit"]]["correct"].mean()
    acc_not_covered = result_df[~result_df["any_rule_hit"]]["correct"].mean()
    print(f"\nAccuracy when rule covered:     {acc_covered:.4f}")
    print(f"Accuracy when NO rule covered:  {acc_not_covered:.4f}")

    # Agreement boost
    agree_df = result_df[result_df["any_rule_hit"]]
    if len(agree_df):
        acc_agree = agree_df[agree_df["ml_rule_agree"] == True]["correct"].mean()
        acc_disagree = agree_df[agree_df["ml_rule_agree"] == False]["correct"].mean()
        print(f"\nAccuracy when ML+Rule AGREE:    {acc_agree:.4f} (n={int((agree_df['ml_rule_agree']==True).sum())})")
        print(f"Accuracy when ML+Rule DISAGREE: {acc_disagree:.4f} (n={int((agree_df['ml_rule_agree']==False).sum())})")

    # --- Coverage bar chart ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = per_class_cov["true_label"].tolist()
    vals = per_class_cov["coverage_pct"].tolist()
    ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#dd8452", "#c44e52"])
    ax.axhline(coverage_pct, color="black", linestyle="--", label=f"Overall {coverage_pct:.1f}%")
    ax.set_ylabel("Rule coverage (%)")
    ax.set_title("Symbolic rule coverage by true pramana class (VAL set)")
    ax.legend()
    plt.tight_layout()
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / "figures" / "rule_coverage.png", dpi=150)
    plt.close()

    # --- Explainability rating sheet (sample 100, balanced per class) ---
    sample_rows = []
    for lbl in PRAMANA_LABELS:
        subset = result_df[result_df["true_label"] == lbl].head(25)
        sample_rows.append(subset)
    rating_df = pd.concat(sample_rows, ignore_index=True).sample(frac=1.0, random_state=42).head(100)

    rating_sheet = pd.DataFrame({
        "row_id": range(1, len(rating_df) + 1),
        "text": rating_df["text"].values,
        "system_prediction": rating_df["final_label"].values,
        "true_label": rating_df["true_label"].values,
        "matched_rules": rating_df["matched_cues_summary"].values,
        "ml_rule_agreement": rating_df["ml_rule_agree"].values,
        "confidence_pct": rating_df["adjusted_confidence"].values,
        # Rater columns (to be filled)
        "rater_correctness_1_5": "",   # Is the explanation correct? (1=wrong, 5=fully correct)
        "rater_usefulness_1_5": "",    # Is the explanation useful? (1=useless, 5=very helpful)
        "rater_comments": "",
    })
    rating_path = REPORTS_DIR / "explainability_rating_sheet.csv"
    rating_sheet.to_csv(rating_path, index=False)
    print(f"\n  Rating sheet: {rating_path.relative_to(ROOT)} (100 examples for human raters)")

    # --- Write report ---
    lines = [
        "# Rule Coverage and Explainability Report\n\n",
        "**Reviewer fix MISS-4**: Quantitative rule coverage analysis. "
        "An explainability rating sheet has been prepared for human evaluation.\n\n",
        "## Overall coverage\n\n",
        f"| Metric | Value |\n|---|---:|\n",
        f"| Total predictions | {total} |\n",
        f"| Predictions with ≥1 rule match | {int(covered)} ({coverage_pct:.1f}%) |\n",
        f"| Predictions with NO rule match | {total - int(covered)} ({100 - coverage_pct:.1f}%) |\n",
        f"| Accuracy (rule covered) | {acc_covered:.4f} |\n",
        f"| Accuracy (no rule) | {acc_not_covered:.4f} |\n\n",
        "**Interpretation:** For predictions with no rule match, the explanation "
        "falls back to embedding-space SHAP values, which are not human-readable "
        "token-level justifications. This must be disclosed in the paper.\n\n",
        "## Per-class coverage\n\n",
        "| pramana_label | total | covered | coverage_pct |\n| --- | --- | --- | --- |\n" +
        "\n".join(f"| {row['true_label']} | {row['total']} | {row['covered']} | {row['coverage_pct']}% |" for _, row in per_class_cov.iterrows()) + "\n\n",
        "![Rule coverage by class](figures/rule_coverage.png)\n\n",
        "## Agreement analysis\n\n",
        f"| Condition | Accuracy |\n|---|---:|\n",
        f"| ML and rule agree | {acc_agree:.4f} |\n",
        f"| ML and rule disagree | {acc_disagree:.4f} |\n\n",
        "## Human evaluation (pending)\n\n",
        "The rating sheet `reports/explainability_rating_sheet.csv` contains 100 "
        "sampled predictions (25 per class) with system explanations. "
        "Two raters should independently score each explanation on:\n\n",
        "- **Correctness** (1–5): Does the matched rule/cue actually justify the predicted label?\n",
        "- **Usefulness** (1–5): Does the explanation help a user understand the system's reasoning?\n\n",
        "Inter-rater agreement (Krippendorff's α or Cohen's κ) must be reported. "
        "Results will be added to this section after rating.\n\n",
        "## Explainability coverage claim\n\n",
        "The paper should state:\n\n",
        f"> Symbolic rule evidence was available for {coverage_pct:.1f}% of predictions. "
        f"For the remaining {100 - coverage_pct:.1f}%, the explanation relies on "
        "embedding-space attributions (SHAP), which provide feature-level "
        "but not human-readable textual justifications.\n",
    ]
    (REPORTS_DIR / "rule_coverage_report.md").write_text("".join(lines), encoding="utf-8")
    print(f"  Report: {(REPORTS_DIR / 'rule_coverage_report.md').relative_to(ROOT)}")
    print("✅ Rule coverage evaluation complete.")


if __name__ == "__main__":
    main()
