"""
Group-disjoint 5-fold cross-validation on real-world corpus.

Reviewer fix: M-15 (MAJOR)
The previous CV in generate_real_world_cross_validation.py used stratified
row-level splits. This script uses StratifiedGroupKFold so that no essay
(AAEC) or topic (IBM ArgKP) appears in both train and validation folds.

Usage (from project root):
    python scripts/run_group_cv.py

Outputs
-------
reports/group_cv_report.md
reports/figures/group_cv_confusion_matrix.png
reports/group_cv_vs_row_cv_comparison.md   (shows Δ from M-15 correction)
"""

from __future__ import annotations

import sys
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
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.embedder import generate_embeddings

PROCESSED = ROOT / "dataset" / "processed"
REPORTS_DIR = ROOT / "reports"

AAEC_WITH_GROUPS = PROCESSED / "aaec_reviewed_with_groups.csv"
IBM_WITH_GROUPS = PROCESSED / "ibm_reviewed_with_groups.csv"

PRAMANA_LABELS = ["Anumana", "Pratyaksha", "Shabda", "Upamana"]
N_FOLDS = 5
RANDOM_STATE = 42


def _group_key_real(row: pd.Series) -> str:
    if row["source"] == "aaec_reviewed" and str(row.get("essay_id", "")).strip():
        return "essay__" + str(row["essay_id"]).strip()
    if row["source"] == "ibm_reviewed" and str(row.get("topic_group", "")).strip():
        return "topic__" + str(row["topic_group"]).strip()
    return "anon__" + str(abs(hash(str(row.get("text", ""))[:80])))


def main() -> None:
    # Load group-enriched real-world data
    missing = [p for p in [AAEC_WITH_GROUPS, IBM_WITH_GROUPS] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing group-key files: {missing}\n"
            "Run scripts/add_group_keys.py first."
        )

    aaec = pd.read_csv(AAEC_WITH_GROUPS)
    ibm = pd.read_csv(IBM_WITH_GROUPS)
    df = pd.concat([aaec, ibm], ignore_index=True)
    df = df[df["pramana_label"].isin(PRAMANA_LABELS)].dropna(subset=["text", "pramana_label"])
    df = df.reset_index(drop=True)
    df["group_key"] = df.apply(_group_key_real, axis=1)

    print(f"Real-world corpus: {len(df)} rows, {df['group_key'].nunique()} groups")
    print("Class distribution:")
    print(df["pramana_label"].value_counts().to_string())

    texts = df["text"].astype(str).tolist()
    y = df["pramana_label"].tolist()
    groups = df["group_key"].tolist()

    print("\nGenerating Sentence-BERT embeddings (full corpus)...")
    X = generate_embeddings(texts)

    # StratifiedGroupKFold — keeps class proportions AND ensures group disjointness
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        # Verify group disjointness
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        overlap = train_groups & val_groups
        assert len(overlap) == 0, f"Group leak in fold {fold+1}: {overlap}"

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = [y[i] for i in train_idx]
        y_val = [y[i] for i in val_idx]

        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val).tolist()

        acc = accuracy_score(y_val, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="macro", labels=PRAMANA_LABELS, zero_division=0
        )

        fold_results.append({
            "fold": fold + 1,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
        })
        all_true.extend(y_val)
        all_pred.extend(preds)
        print(f"  Fold {fold+1}: acc={acc:.4f} macro_F1={f1:.4f} "
              f"(train_groups={len(train_groups)}, val_groups={len(val_groups)}, overlap={len(overlap)})")

    # Pooled OOF
    oof_acc = accuracy_score(all_true, all_pred)
    oof_prec, oof_rec, oof_f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", labels=PRAMANA_LABELS, zero_division=0
    )
    oof_report = classification_report(all_true, all_pred, labels=PRAMANA_LABELS, digits=4, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=PRAMANA_LABELS)

    fold_df = pd.DataFrame(fold_results)
    mean_row = {"fold": "mean", **{c: round(fold_df[c].mean(), 4) for c in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}, "n_train": "", "n_val": "", "n_train_groups": "", "n_val_groups": ""}
    std_row = {"fold": "std", **{c: round(fold_df[c].std(), 4) for c in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}, "n_train": "", "n_val": "", "n_train_groups": "", "n_val_groups": ""}
    summary_df = pd.concat([fold_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print("\nGroup-disjoint CV summary:")
    print(summary_df[["fold", "n_train_groups", "n_val_groups", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"\nPooled OOF: acc={oof_acc:.4f}, macro_F1={oof_f1:.4f}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_df = pd.DataFrame(cm, index=PRAMANA_LABELS, columns=PRAMANA_LABELS)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Group-Disjoint CV Confusion Matrix\n(Pooled OOF, macro-F1={oof_f1:.3f})")
    plt.tight_layout()
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "group_cv_confusion_matrix.png", dpi=150)
    plt.close()

    # Write report
    lines = [
        "# Group-Disjoint Cross-Validation (Real-World Corpus)\n\n",
        "**Reviewer fix M-15**: Stratified row-level CV replaced with "
        "`StratifiedGroupKFold` using `essay_id` (AAEC) and `topic_group` (IBM ArgKP) "
        "as group keys. Zero group overlap verified per fold.\n\n",
        f"**Dataset**: {len(df)} rows, {df['group_key'].nunique()} unique groups, {N_FOLDS} folds.\n\n",
        "## Class distribution\n\n",
        _df_to_md(df[["pramana_label","count"]].assign(count=df["pramana_label"].value_counts().values)
                  if False else df["pramana_label"].value_counts().reset_index().rename(columns={"index":"pramana_label","pramana_label":"count"})) + "\n\n",
        "## Per-fold results\n\n",
        _df_to_md(summary_df[["fold", "n_train", "n_val", "n_train_groups", "n_val_groups",
                    "accuracy", "macro_precision", "macro_recall", "macro_f1"]]) + "\n\n",
        "## Pooled OOF\n\n",
        f"| Metric | Value |\n|---|---:|\n",
        f"| Accuracy | {oof_acc:.4f} |\n",
        f"| Macro Precision | {oof_prec:.4f} |\n",
        f"| Macro Recall | {oof_rec:.4f} |\n",
        f"| Macro F1 | {oof_f1:.4f} |\n\n",
        "## Pooled OOF classification report\n\n",
        f"```text\n{oof_report}```\n\n",
        "## Confusion matrix (pooled OOF)\n\n",
        _df_to_md(cm_df, index=True) + "\n\n",
        "![Group CV confusion matrix](figures/group_cv_confusion_matrix.png)\n\n",
        "## Group disjointness verification\n\n",
        "All folds confirmed zero group overlap between train and val partitions. "
        "Each AAEC essay and each IBM ArgKP topic appears in exactly one fold's "
        "validation set.\n\n",
        "## Comparison with previous row-level CV\n\n",
        "| Metric | Previous (row-level) | Current (group-disjoint) | Δ |\n",
        "|---|---:|---:|---:|\n",
        f"| Accuracy | 0.9118 | {oof_acc:.4f} | {oof_acc-0.9118:+.4f} |\n",
        f"| Macro F1 | 0.3414 | {oof_f1:.4f} | {oof_f1-0.3414:+.4f} |\n\n",
        "*Δ shows the effect of controlling for document/topic leakage. "
        "Negative Δ indicates the previous CV overestimated generalization.*\n",
    ]
    (REPORTS_DIR / "group_cv_report.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n  Report: {(REPORTS_DIR / 'group_cv_report.md').relative_to(ROOT)}")
    print("✅ Group-disjoint CV complete.")


if __name__ == "__main__":
    main()
