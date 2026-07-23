"""
Generate reports/oversampling_comparison.md comparing three training strategies
on dataset/test_set.csv.

Usage:
    python reports/generate_oversampling_comparison.py

Expects snapshot files under reports/model_eval_snapshots/ for original and
class-weight models. Evaluates the current (oversampled) model on test_set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports._eval_test_set import PRAMANA_LABELS, evaluate_test_set

SNAPSHOT_DIR = _ROOT / "reports" / "model_eval_snapshots"
ORIGINAL_SNAPSHOT = SNAPSHOT_DIR / "original_test_set.json"
CLASS_WEIGHT_SNAPSHOT = SNAPSHOT_DIR / "class_weight_test_set.json"
OVERSAMPLE_STATS = _ROOT / "dataset" / "processed" / "oversample_stats.json"
REPORT_OUT = _ROOT / "reports" / "oversampling_comparison.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cm_markdown(classes: list[str], cm: list[list[int]]) -> str:
    lines = ["| True \\ Pred | " + " | ".join(classes) + " |", "| --- | " + " | ".join("---" for _ in classes) + " |"]
    for i, true_lbl in enumerate(classes):
        row = " | ".join(str(int(cm[i][j])) for j in range(len(classes)))
        lines.append(f"| {true_lbl} | {row} |")
    return "\n".join(lines)


def _per_class_table(metrics: dict) -> str:
    lines = ["| Class | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: |"]
    for cls in PRAMANA_LABELS:
        pc = metrics["per_class"][cls]
        lines.append(f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} |")
    return "\n".join(lines)


def _pick_best(models: dict[str, dict]) -> str:
    """Rank by macro F1, then accuracy."""
    ranked = sorted(
        models.items(),
        key=lambda x: (x[1]["macro_f1"], x[1]["accuracy"]),
        reverse=True,
    )
    return ranked[0][0]


def main() -> None:
    if not ORIGINAL_SNAPSHOT.is_file() or not CLASS_WEIGHT_SNAPSHOT.is_file():
        raise FileNotFoundError("Missing model snapshot JSON files under reports/model_eval_snapshots/")

    original = _load_json(ORIGINAL_SNAPSHOT)
    class_weight = _load_json(CLASS_WEIGHT_SNAPSHOT)
    oversampled = evaluate_test_set()

    oversample_meta = _load_json(OVERSAMPLE_STATS) if OVERSAMPLE_STATS.is_file() else {}

    lines: list[str] = []
    lines.append("# Oversampling vs Original vs Class-Weight — Comparison\n\n")
    lines.append(
        "Held-out evaluation on `dataset/test_set.csv` (n=200). "
        "Training strategies share the same merged deduplicated corpus before resampling.\n\n"
    )

    if oversample_meta:
        lines.append("## Class counts (oversampling)\n\n")
        lines.append("### Before oversampling\n\n")
        lines.append("| Class | Count |\n| --- | ---: |\n")
        for cls in PRAMANA_LABELS:
            lines.append(f"| {cls} | {oversample_meta['before'][cls]} |\n")
        lines.append("\n### After oversampling\n\n")
        lines.append("| Class | Count |\n| --- | ---: |\n")
        for cls in PRAMANA_LABELS:
            lines.append(f"| {cls} | {oversample_meta['after'][cls]} |\n")
        lines.append(
            f"\nAnumana rows kept unchanged. Minority classes upsampled to ~{oversample_meta.get('target', 500)} "
            f"with replacement (`random_state=42`).\n\n"
        )

    lines.append("## Aggregate metrics (test_set.csv)\n\n")
    lines.append("| Model | Accuracy | Macro precision | Macro recall | Macro F1 |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for name, m in [
        ("Original", original),
        ("Class-weight (`balanced`)", class_weight),
        ("Oversampled minority", oversampled),
    ]:
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | "
            f"{m['macro_recall']:.4f} | {m['macro_f1']:.4f} |\n"
        )
    lines.append("\n")

    for title, m in [
        ("Original model", original),
        ("Class-weight model", class_weight),
        ("Oversampled model", oversampled),
    ]:
        lines.append(f"## {title}\n\n")
        lines.append("### Per-class metrics\n\n")
        lines.append(_per_class_table(m) + "\n\n")
        lines.append("### Confusion matrix\n\n")
        lines.append(_cm_markdown(m["classes"], m["confusion_matrix"]) + "\n\n")

    models = {
        "Original": original,
        "Class-weight": class_weight,
        "Oversampled": oversampled,
    }
    best = _pick_best(models)

    lines.append("## Recommendation\n\n")
    lines.append(
        f"Based on held-out **macro F1** (primary) and **accuracy** (secondary), "
        f"the **{best}** configuration performs best on `dataset/test_set.csv`.\n\n"
    )

    if best == "Original":
        lines.append(
            "- Original training without `class_weight` or oversampling achieves the strongest "
            "balance on the curated test set despite Anumana skew in merged data.\n"
        )
    elif best == "Class-weight":
        lines.append(
            "- Balanced class weights improve minority recall but may hurt aggregate metrics; "
            "use if recall on Upamana/Pratyaksha is prioritized over overall accuracy.\n"
        )
    else:
        lines.append(
            "- Minority oversampling to ~500 examples per class improves representation without "
            "the aggressive boundary shift seen under `class_weight='balanced'`.\n"
        )

    lines.append("\n### Summary comparison\n\n")
    lines.append("| Criterion | Original | Class-weight | Oversampled |\n")
    lines.append("| --- | --- | --- | --- |\n")
    lines.append(
        f"| Accuracy | {original['accuracy']:.4f} | {class_weight['accuracy']:.4f} | {oversampled['accuracy']:.4f} |\n"
    )
    lines.append(
        f"| Macro F1 | {original['macro_f1']:.4f} | {class_weight['macro_f1']:.4f} | {oversampled['macro_f1']:.4f} |\n"
    )
    for cls in ("Shabda", "Upamana", "Pratyaksha"):
        lines.append(
            f"| {cls} recall | {original['per_class'][cls]['recall']:.4f} | "
            f"{class_weight['per_class'][cls]['recall']:.4f} | "
            f"{oversampled['per_class'][cls]['recall']:.4f} |\n"
        )

    REPORT_OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_OUT}")
    print(f"Recommended model: {best}")


if __name__ == "__main__":
    main()
