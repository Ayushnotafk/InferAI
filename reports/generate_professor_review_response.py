"""
Assemble the consolidated professor review response report.

Reads outputs from other report scripts when available; regenerates lightweight
summaries where needed.

Usage (after running individual report generators):
    python reports/generate_professor_review_response.py

Or run the full pipeline:
    python reports/run_all_evaluation_reports.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports._common import ensure_reports_dir

REPORT_LINKS = {
    "dataset": "dataset_statistics.md",
    "lexical": "lexical_diversity_report.md",
    "kappa": "kappa_report.md",
    "heldout": "heldout_evaluation.md",
    "confusion": "confusion_matrix_analysis.md",
    "calibration": "calibration_report.md",
    "leakage": "data_leakage_report.md",
    "shap": "shap_limitations.md",
}


def _read_if_exists(name: str) -> str:
    path = _ROOT / "reports" / name
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return f"_(Report `{name}` not yet generated — run the corresponding script.)_\n"


def _metrics_snippet() -> str:
    p = _ROOT / "models" / "evaluation" / "metrics.json"
    if not p.is_file():
        return "_Train the model to populate metrics._\n"
    m = json.loads(p.read_text(encoding="utf-8"))
    held = m.get("held_out_test_set") or {}
    return (
        f"- Random holdout accuracy: **{m.get('test_accuracy', 'n/a')}**\n"
        f"- Held-out test accuracy: **{held.get('accuracy', 'n/a')}**\n"
        f"- Macro F1 (random holdout): **{m.get('macro_f1', 'n/a')}**\n"
    )


def main() -> None:
    ensure_reports_dir()

    lines: list[str] = []
    lines.append("# InferAI — Professor Review Response\n\n")
    lines.append(
        "Consolidated research evaluation addressing dataset transparency, lexical diversity, "
        "annotation agreement, held-out performance, calibration, leakage auditing, and "
        "explainability scope. **Core model architecture unchanged** (Sentence-BERT + logistic "
        "regression + hybrid rules).\n\n"
    )

    lines.append("## Executive summary\n\n")
    lines.append(_metrics_snippet())
    lines.append(
        "\nKey takeaway: high in-distribution accuracy reflects template-heavy training data; "
        "held-out curated evaluation (~76% accuracy) exposes distribution shift. Reporting now "
        "includes length-aware lexical metrics, bootstrap κ, per-class held-out metrics, "
        "calibration diagnostics, and cross-split leakage checks.\n\n"
    )

    sections = [
        (
            "1. Dataset audit",
            "Addresses corpus sizes, class/strength distributions, deduplication counts, and train/val/test partitioning.",
            REPORT_LINKS["dataset"],
        ),
        (
            "2. Lexical diversity analysis",
            "Extends raw TTR with Guiraud's R and MATTR-50; explains why TTR alone misleads across corpora of different sizes.",
            REPORT_LINKS["lexical"],
        ),
        (
            "3. Annotation agreement",
            "Cohen's κ, agreement %, bootstrap 95% CI, per-class rates, with caveat on rule-assisted annotator 2.",
            REPORT_LINKS["kappa"],
        ),
        (
            "4. Held-out performance",
            "Per-class precision/recall/F1/support, macro/weighted F1, balanced accuracy on `test_set.csv`.",
            REPORT_LINKS["heldout"],
        ),
        (
            "5. Confusion matrix interpretation",
            "Automatic identification of most/least confused class pairs and recall extremes.",
            REPORT_LINKS["confusion"],
        ),
        (
            "6. Calibration analysis",
            "Reliability diagram, per-class calibration curves, ECE, and Brier score on held-out data.",
            REPORT_LINKS["calibration"],
        ),
        (
            "7. Leakage audit",
            "Exact and near-duplicate checks (RapidFuzz + embedding cosine) across train/val/test.",
            REPORT_LINKS["leakage"],
        ),
        (
            "8. Explainability discussion",
            "Clarifies embedding-level SHAP scope; distinguishes from token-level and template explanations.",
            REPORT_LINKS["shap"],
        ),
    ]

    for title, blurb, link in sections:
        lines.append(f"## {title}\n\n")
        lines.append(f"{blurb}\n\n")
        lines.append(f"**Full report:** [`reports/{link}`]({link})\n\n")
        # Include first substantive paragraph from child report if present
        child = _read_if_exists(link)
        if not child.startswith("_("):
            # Skip title line
            parts = [p.strip() for p in child.split("\n\n") if p.strip() and not p.startswith("#")]
            if parts:
                snippet = parts[0][:500]
                if len(parts[0]) > 500:
                    snippet += "…"
                lines.append(f"> {snippet}\n\n")

    lines.append("## 9. Limitations\n\n")
    lines.append(
        "- **Synthetic skew:** Merged training data is dominated by template-generated Anumana examples after deduplication.\n"
        "- **Small real-world corpus:** 200 curated training + 200 held-out test limits statistical power.\n"
        "- **Simulated annotator 2:** κ is optimistic relative to fully independent human annotation.\n"
        "- **Embedding SHAP:** Not token-faithful; hybrid rules not covered by SHAP.\n"
        "- **Uncalibrated probabilities:** Confidence scores require cautious interpretation (see calibration report).\n"
        "- **English-only short spans:** Nyāya taxonomy mapping is approximate for modern rhetorical text.\n\n"
    )

    lines.append("## 10. Future work\n\n")
    lines.append(
        "- Expand independently annotated real-world data; rebalance class priors.\n"
        "- Token-level attributions and faithfulness user studies.\n"
        "- Post-hoc probability calibration on validation fold.\n"
        "- Cross-lingual and longer argumentative discourse support.\n"
        "- Fine-tune or lightly adapt Sentence-BERT on in-domain Nyāya-labeled text.\n\n"
    )

    lines.append("## Report index\n\n")
    lines.append("| Report | Path |\n")
    lines.append("| --- | --- |\n")
    for key, fname in REPORT_LINKS.items():
        lines.append(f"| {key} | `reports/{fname}` |\n")
    lines.append("| Final evaluation (legacy) | `reports/final_evaluation_report.md` |\n")

    out = _ROOT / "reports" / "professor_review_response.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
