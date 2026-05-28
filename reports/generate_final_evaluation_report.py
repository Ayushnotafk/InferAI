"""
Assemble ``reports/final_evaluation_report.md`` from saved metrics and a
short qualitative pass over ``dataset/test_set.csv``.

Run after training:

    python reports/generate_final_evaluation_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings


def _load_metrics() -> dict:
    p = _ROOT / "models" / "evaluation" / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}. Train the model first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _misclassified_examples(limit: int = 6) -> list[dict]:
    model = joblib.load(_ROOT / "models" / "infer_model.pkl")
    le = joblib.load(_ROOT / "models" / "label_encoder.pkl")
    tdf = pd.read_csv(_ROOT / "dataset" / "test_set.csv")
    texts = tdf["text"].astype(str).tolist()
    y_true = le.transform(tdf["pramana_label"].astype(str))
    X = generate_embeddings(texts)
    y_pred = model.predict(X)
    rows: list[dict] = []
    for i in range(len(texts)):
        if y_pred[i] != y_true[i]:
            rows.append(
                {
                    "text": texts[i],
                    "true": le.inverse_transform([y_true[i]])[0],
                    "pred": le.inverse_transform([y_pred[i]])[0],
                }
            )
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    metrics = _load_metrics()
    report_path = _ROOT / "reports" / "final_evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cr_path = _ROOT / "models" / "evaluation" / "classification_report.txt"
    cr_txt = cr_path.read_text(encoding="utf-8") if cr_path.is_file() else "(missing classification_report.txt)"

    failures = _misclassified_examples()

    lines: list[str] = []
    lines.append("# InferAI — Final Evaluation Report\n")
    lines.append("This report summarizes quantitative metrics produced during training and a small qualitative review on the held-out `dataset/test_set.csv`.\n")

    lines.append("## Quantitative summary\n")
    lines.append(f"- **Train accuracy:** {metrics.get('train_accuracy')}\n")
    lines.append(f"- **Test accuracy (20% random holdout from training table):** {metrics.get('test_accuracy')}\n")
    lines.append(f"- **Macro precision:** {metrics.get('macro_precision')}\n")
    lines.append(f"- **Macro recall:** {metrics.get('macro_recall')}\n")
    lines.append(f"- **Macro F1:** {metrics.get('macro_f1')}\n")

    held = metrics.get("held_out_test_set")
    if isinstance(held, dict) and "accuracy" in held:
        lines.append("\n## Held-out curated test set\n")
        lines.append(f"- **Accuracy on `dataset/test_set.csv`:** {held.get('accuracy')}\n")
        lines.append("\n### Classification report (held-out)\n")
        lines.append("```text\n")
        lines.append(held.get("classification_report", "").strip() + "\n")
        lines.append("```\n")

    lines.append("\n## Sklearn report (random holdout)\n")
    lines.append("```text\n")
    lines.append(cr_txt.strip() + "\n")
    lines.append("```\n")

    lines.append("\n## Confusion matrices (artifacts)\n")
    lines.append("- `models/evaluation/confusion_matrix.png` — random holdout split\n")
    lines.append("- `models/evaluation/confusion_matrix_test_set.png` — curated test set (if generated)\n")

    lines.append("\n## Qualitative notes\n")
    lines.append(
        "The logistic regression head operates on Sentence-BERT embeddings; errors often occur when "
        "rhetorical style mixes multiple pramāṇa cues in one short span, or when implicit authority language "
        "appears without explicit attribution.\n"
    )

    lines.append("\n## Example failure cases (first few mislabels on test set)\n")
    if not failures:
        lines.append("_No misclassifications found in the first scanned portion of the test set._\n")
    else:
        for i, ex in enumerate(failures, 1):
            lines.append(f"{i}. **True:** `{ex['true']}` → **Predicted:** `{ex['pred']}`  \n")
            lines.append(f"   - Text: \"{ex['text'][:320].replace('`', '')}{'…' if len(ex['text']) > 320 else ''}\"\n")

    lines.append("\n## Observations\n")
    lines.append(
        "- Hybrid fusion (`classification/hybrid_reasoning.py`) can stabilize borderline cases at inference time, "
        "but the supervised head still reflects whatever distribution is present in `infer_dataset.csv`.\n"
        "- Composite strength (`reasoning_strength/composite.py`) intentionally decouples rhetorical strength from raw softmax peaks.\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
